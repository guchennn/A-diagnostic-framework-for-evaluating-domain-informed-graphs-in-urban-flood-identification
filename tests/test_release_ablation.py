"""Synthetic selection and prepared-fold tests; no city training or GPU use."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reproduce_ablation", ROOT / "scripts/reproduce_ablation.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.fixture
def inventory(tmp_path):
    rows = []
    for budget in ("100", "FULL"):
        for fold in range(5):
            for trajectory in range(1, 6) if budget == "100" else ("NA",):
                for feature in replay.SETS.values():
                    for model in ("GraphSAGE-G2", "XGBoost"):
                        neural = model == "GraphSAGE-G2"
                        rows.append(
                            {
                                "budget": budget,
                                "outer_fold": str(fold),
                                "trajectory": str(trajectory),
                                "feature_set": feature,
                                "city": "zhengzhou",
                                "model": model,
                                "config": "SMALL",
                                "best_inner_epochs": json.dumps([3, 5, 7] if neural else [0, 0, 0])
                                if budget == "100"
                                else "",
                                "refit_epochs": str(5 if neural else 0) if budget == "100" else "",
                            }
                        )
    path = tmp_path / "selected.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_complete_inventory_and_full_gap(inventory):
    frame = replay.load_selections(inventory)
    assert len(frame) == 180
    assert replay.plan(frame, "100")["fits"] == 750
    full = replay.plan(frame, "FULL")
    assert full["units"] == 15
    assert full["missing_neural_epoch_records"] == 15
    assert full["status"] == "blocked"
    assert set(frame.loc[frame.budget.eq("FULL"), "trajectory"]) == {"NA"}


@pytest.mark.parametrize("damage", ["duplicate", "missing", "epochs", "city", "config"])
def test_corrupt_selection_rejected(inventory, damage):
    frame = pd.read_csv(inventory, dtype=str, keep_default_na=False)
    if damage == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif damage == "missing":
        frame = frame.iloc[1:]
    elif damage == "epochs":
        frame.loc[0, "refit_epochs"] = "99"
    else:
        frame.loc[0, damage] = "invalid"
    frame.to_csv(inventory, index=False)
    with pytest.raises(ValueError):
        replay.load_selections(inventory)


@pytest.mark.parametrize("feature,count", [("F18", 18), ("F16-noD8", 16), ("F14-noNeigh", 14)])
def test_prepared_fold_replay_inherits_schedule_and_pairs_seeds(
    inventory, monkeypatch, feature, count
):
    frame = replay.load_selections(inventory)
    rows = frame.loc[
        frame.budget.eq("100")
        & frame.outer_fold.eq("0")
        & frame.trajectory.eq("1")
        & frame.feature_set.eq(feature)
    ]
    rows = rows.copy()
    rows.loc[rows.model.eq("GraphSAGE-G2"), "config"] = "MEDIUM"
    raw = pd.DataFrame(np.arange(54).reshape(3, 18), columns=replay.EXPECTED_FEATURE_NAMES)
    banks = pd.DataFrame(
        {"grid_id": ["a", "b"] * 3, "bank_id": [1, 1, 2, 2, 3, 3], "sample_role": ["P", "B"] * 3}
    )
    fold = {
        "raw": raw,
        "test_node_ids": ["a", "b"],
        "banks": banks,
        "grid": pd.DataFrame({"row": [0, 21], "column": [0, 21]}, index=["a", "b"]),
    }
    units, calls = [], []

    def outer(data, unit):
        units.append(unit)
        return fold

    def predict(prepared, task, grid, epochs):
        calls.append((task, epochs))
        assert list(prepared["raw"].columns) == replay.feature_names(ROOT)[feature]
        assert prepared["raw"].shape[1] == count
        assert task.feature_set == "FULL-COMMON"
        assert task.model == ("XGBoost" if task.graph_id == "NA" else "GraphSAGE-G2")
        return np.array([0.2, 0.8]), {}

    monkeypatch.setattr(replay.e1, "_outer_fold_data", outer)
    monkeypatch.setattr(replay.e1, "_predict_outer", predict)
    grid = yaml.safe_load((ROOT / "configs/model_grid.yaml").read_text())
    scores, audit = replay.refit_unit({}, rows, replay.feature_names(ROOT)[feature], grid)
    assert units[0].budget == "100" and units[0].trajectory == "1"
    assert len(calls) == 10 and len(scores) == 72 and len(audit) == 10
    assert raw.shape == (3, 18)
    for graph in ("G0", "G1", "G2"):
        tasks = [(task, epochs) for task, epochs in calls if task.graph_id == graph]
        assert {task.seed for task, _ in tasks} == set(replay.SEEDS)
        assert {(task.config, epochs) for task, epochs in tasks} == {("MEDIUM", 5)}
    # Graph identity and ablation identity are absent from the existing E1 optimizer namespace.
    for seed in replay.SEEDS:
        seeds = [r["seed_128"] for r in audit if r["seed_slot"] == seed and r["method"] != "XGB"]
        assert len(set(seeds)) == 1
    for _, group in scores.groupby(["method", "seed", "bank_id"]):
        assert group.score.tolist() == [0.2, 0.8]


def test_full_blocked_before_gate_or_city_read(inventory, monkeypatch, tmp_path):
    monkeypatch.setattr(
        replay.reviewer_revision,
        "prepare_plan",
        lambda root: pytest.fail("FULL must stop before GPU gates"),
    )
    with pytest.raises(ValueError, match="FULL replay blocked"):
        replay.run(ROOT, inventory, tmp_path / "out", "FULL")


def test_failed_gate_prevents_city_read_and_output(inventory, monkeypatch, tmp_path):
    monkeypatch.setattr(replay.reviewer_revision, "prepare_plan", lambda root: {"test": True})

    def reject(root, plan):
        assert plan == {"test": True}
        raise ValueError("stale performance-blind probe")

    monkeypatch.setattr(replay.reviewer_revision, "require_revision_environment", reject)
    monkeypatch.setattr(replay.e1, "_read_city", lambda *a: pytest.fail("city read before gate"))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="stale performance-blind"):
        replay.run(ROOT, inventory, output, "100")
    assert not output.exists()


def test_run_writes_complete_mock_replay(inventory, monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    (root / "configs").mkdir(parents=True)
    for name in ("feature_sets_v1_2.yaml", "model_grid.yaml"):
        (root / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
    events = []
    monkeypatch.setattr(replay.reviewer_revision, "prepare_plan", lambda root: {"test": True})
    monkeypatch.setattr(
        replay.reviewer_revision,
        "require_revision_environment",
        lambda *args: events.append("gate"),
    )

    def read(*args):
        assert events == ["gate"]
        events.append("read")
        return {}

    def refit(data, rows, names, grid):
        assert events == ["gate", "read"]
        assert len(rows) == 2
        return pd.DataFrame({"score": [0.5]}), [{"mock": True}]

    monkeypatch.setattr(replay.e1, "_read_city", read)
    monkeypatch.setattr(replay, "refit_unit", refit)
    output = root / "reproduced" / "mock"
    result = replay.run(root, inventory, output, "100")
    assert result["completed_units"] == 75
    assert result["status"] == "selection-replay-complete"
    assert result["numerical_replication_verified"] is False
    assert len(list(output.glob("*.parquet"))) == 75
    assert json.loads((output / "manifest.json").read_text()) == result
    with pytest.raises(FileExistsError):
        replay.run(root, inventory, output, "100")


@pytest.mark.parametrize(
    "budget,status,units", [("100", "selection-ready", 75), ("FULL", "blocked", 15)]
)
def test_cli_uses_published_default_from_other_directory(tmp_path, budget, status, units):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/reproduce_ablation.py"), "plan", "--budget", budget],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["status"] == status
    assert result["units"] == units


def test_missing_selection_file_is_required(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay.load_selections(tmp_path / "missing.csv")
