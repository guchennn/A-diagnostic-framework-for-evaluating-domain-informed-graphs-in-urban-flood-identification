"""CPU-only contracts for the independently selected attribute series."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "independent_attributes", ROOT / "scripts/run_independent_attribute_controls.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_matrix_matches_historical_independent_series():
    config = runner.configuration()
    all_blocks = [b for city in config["cities"] for b in runner.blocks(config, city)]
    assert len(all_blocks) == 130
    assert len(all_blocks) * len(config["variants"]) == 520
    assert len(all_blocks) * len(config["variants"]) * len(config["outer_seeds"]) == 1560
    assert {b["feature_set"] for b in all_blocks if b["budget"] == "100"} == {
        "F16-noD8",
        "F14-noNeigh",
    }
    assert {b["trajectory"] for b in all_blocks if b["budget"] == "FULL"} == {"NA"}
    assert {b["feature_set"] for b in all_blocks if b["budget"] == "FULL"} == set(
        runner.FEATURE_LABELS
    )


def test_subset_preserves_frozen_order_and_does_not_mutate():
    raw = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = runner.subset({"raw": raw}, ["c", "a"])
    assert result["raw"].columns.tolist() == ["a", "c"]
    assert raw.columns.tolist() == ["a", "b", "c"]
    with pytest.raises(ValueError, match="feature names"):
        runner.subset({"raw": raw}, ["missing"])


def test_full_task_keeps_budget_draw_and_seed_namespace():
    config = runner.configuration()
    grid = yaml.safe_load((ROOT / "configs/model_grid.yaml").read_text("utf-8"))
    block = next(b for b in runner.blocks(config, "beijing") if b["budget"] == "FULL")
    task = runner.task_key(config, grid, block, config["variants"][0], "SMALL", "NEURAL-1")
    assert (task.budget, task.trajectory, task.feature_set) == ("FULL", "NA", "FULL-COMMON")
    meta = runner.seed_record(task, grid)
    assert meta["torch_seed"] == int(meta["seed_uint128"]) % (2**63)
    assert "FULL" in meta["seed_namespace"]


def test_all_variants_select_independently_before_refitting(tmp_path, monkeypatch):
    config = runner.configuration()
    grid = yaml.safe_load((ROOT / "configs/model_grid.yaml").read_text("utf-8"))
    block = runner.blocks(config, "beijing")[0]
    fold = {"raw": pd.DataFrame({"a": [1], "b": [2]})}
    monkeypatch.setattr(runner.e1, "_inner_fold_data", lambda *_: fold)
    monkeypatch.setattr(runner.e1, "_outer_fold_data", lambda *_: fold)
    monkeypatch.setattr(runner, "succeeded", lambda *_: False)
    inner, selected, outer = [], [], []

    def fit_inner(root, destination, task, data, model_grid, *, prepared_folds):
        assert len(prepared_folds) == 3
        assert all(f["raw"].columns.tolist() == ["a"] for f in prepared_folds)
        inner.append(task)
        return "success"

    def select(root, destination, unit, *, task_keys, models, alternative_order):
        assert len({t.realization_id for t in task_keys}) == 1
        selected.append(task_keys[0].realization_id)
        own = task_keys[0].graph_id == "G0"
        return {
            "models": {
                models[0]: {
                    "config": "MEDIUM" if own else "SMALL",
                    "best_epochs": [7, 9, 11] if own else [3, 5, 7],
                }
            }
        }

    def fit_outer(destination, task, data, model_grid, *, epochs):
        assert len(selected) == 4
        assert (task.config, epochs) == (("MEDIUM", 9) if task.graph_id == "G0" else ("SMALL", 5))
        outer.append(task)
        return "success"

    monkeypatch.setattr(runner.e1, "run_inner_task", fit_inner)
    monkeypatch.setattr(runner.e1, "select_outer_models", select)
    monkeypatch.setattr(runner.e1, "run_outer_fit", fit_outer)
    runner.run_block(tmp_path, config, {}, grid, block, ["a"])
    assert (len(inner), len(selected), len(outer)) == (8, 4, 12)
    records = list(runner.run_path(tmp_path, config).rglob("training_metadata.json"))
    assert len(records) == 12
    assert {json.loads(p.read_text())["selected_epochs"] for p in records} == {5, 9}


def test_missing_full_inputs_fail_without_substitution(tmp_path, monkeypatch):
    config = runner.configuration()
    config["cities"] = ["beijing"]
    config["conditions"] = [config["conditions"][1]]
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/model_grid.yaml").write_text("protocol_id: test")
    monkeypatch.setattr(runner.e1, "_read_city", lambda *_: {})

    def missing(data, task, inner):
        assert task.budget == "FULL" and task.trajectory == "NA"
        raise ValueError("inner training P or U is incomplete")

    monkeypatch.setattr(runner.e1, "_inner_fold_data", missing)
    with pytest.raises(ValueError, match="incomplete"):
        runner.check_inputs(tmp_path, config)


def test_resume_rejects_modified_payload(tmp_path):
    payload = tmp_path / "oof.parquet"
    payload.write_bytes(b"original")
    receipt = tmp_path / "status.json"
    receipt.write_text(json.dumps({"status": "success", "oof_sha256": runner.sha256_file(payload)}))
    assert runner.succeeded(receipt, payload.name, "oof_sha256")
    payload.write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload changed"):
        runner.succeeded(receipt, payload.name, "oof_sha256")
