from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import yaml
from shapely.geometry import box

from floodgnn_ceus import evaluation_design

ROOT = Path(__file__).parents[1]


def aligned_grid() -> gpd.GeoDataFrame:
    columns = [19, 20, 25]
    return gpd.GeoDataFrame(
        {
            "grid_id": [f"g{column}" for column in columns],
            "row": [0, 0, 0],
            "column": columns,
        },
        geometry=[box(column * 200, 0, (column + 1) * 200, 200) for column in columns],
        crs=32649,
    )


def test_greedy_blocks_and_open_buffer_are_deterministic() -> None:
    blocks = pd.DataFrame(
        {
            "block_row": [0, 0, 1, 1],
            "block_col": [0, 1, 0, 1],
            "positive_count": [1, 1, 1, 1],
            "node_count": [1, 1, 1, 1],
        }
    )
    assigned = evaluation_design.assign_blocks(blocks, folds=2, tolerance=1e-12)
    assert assigned["fold_id"].tolist() == [0, 1, 0, 1]

    block_assignment = pd.DataFrame({"block_row": [0, 0], "block_col": [0, 1], "fold_id": [0, 1]})
    partition = evaluation_design.materialize_partition(
        aligned_grid(), block_assignment, held_out_fold=0, buffer_m=1000
    )
    assert dict(zip(partition["grid_id"], partition["partition"], strict=True)) == {
        "g19": "test",
        "g20": "buffer",
        "g25": "train",
    }
    assert partition["grid_id"].is_unique


def test_reveals_are_nested_and_u_removes_only_current_reveals() -> None:
    candidates = [f"g{index:04d}" for index in range(250)]
    first = evaluation_design.random_order(candidates, "protocol/city/0/uniform/1")
    second = evaluation_design.random_order(reversed(candidates), "protocol/city/0/uniform/1")
    assert first == second
    assert set(first[:20]) < set(first[:50]) < set(first[:100]) < set(first[:200])

    master = ["revealed", "unrevealed-positive", *[f"u{index:04d}" for index in range(2001)]]
    selected = evaluation_design.select_u(master, {"revealed"}, size=2000)
    assert "revealed" not in selected
    assert "unrevealed-positive" in selected
    assert len(selected) == 2000
    with pytest.raises(ValueError, match="fewer than 2000"):
        evaluation_design.select_u(master[:100], set(), size=2000)


def test_banks_are_fixed_one_to_ten_samples() -> None:
    positives = ["p0", "p1"]
    background = [f"b{index:02d}" for index in range(30)]
    bank = evaluation_design.sample_bank(
        positives,
        background,
        ratio=10,
        seed_name="protocol/city/STRICT-LOC/0/outer/bank/1",
    )
    repeated = evaluation_design.sample_bank(
        positives,
        reversed(background),
        ratio=10,
        seed_name="protocol/city/STRICT-LOC/0/outer/bank/1",
    )
    assert bank.equals(repeated)
    assert bank.groupby("sample_role").size().to_dict() == {"background": 20, "positive": 2}
    assert set(positives).isdisjoint(bank.query("sample_role == 'background'")["grid_id"])
    with pytest.raises(ValueError, match="bank background"):
        evaluation_design.sample_bank(positives, background[:19], ratio=10, seed_name="x")


def test_label_interface_is_allowlisted_and_strict_is_a_subset() -> None:
    ledger = pd.DataFrame(
        {
            "grid_id": ["g0", "g0", "g1"],
            "label_layer": ["STRICT-LOC", "ALL-LOC", "ALL-LOC"],
            "raw_id": ["secret", "secret", "secret"],
            "toponym": ["private", "private", "private"],
            "flood_prob": [0.9, 0.9, 0.1],
        }
    )
    labels = evaluation_design.label_sets_from_ledger(ledger)
    assert labels == {"STRICT-LOC": {"g0"}, "ALL-LOC": {"g0", "g1"}}
    assert set(labels) == {"STRICT-LOC", "ALL-LOC"}

    invalid = ledger.loc[ledger["label_layer"] == "STRICT-LOC"]
    with pytest.raises(ValueError, match="subset"):
        evaluation_design.label_sets_from_ledger(invalid)


def test_feasibility_marks_a_budget_without_resampling() -> None:
    status = evaluation_design.budget_feasibility([183, 173, 185, 170, 190], 200)
    assert status == {
        "status": "not-estimable",
        "reason": "outer_train_positive_count_below_budget",
        "minimum_available": 170,
        "required": 200,
    }
    assert evaluation_design.budget_feasibility([183, 173, 185, 170, 190], 100)["status"] == (
        "success"
    )


def test_frozen_configs_capture_the_complete_design_contract() -> None:
    split = yaml.safe_load((ROOT / "configs" / "split_config.yaml").read_text(encoding="utf-8"))
    labels = yaml.safe_load((ROOT / "configs" / "label_schema.yaml").read_text(encoding="utf-8"))

    assert split["seed_protocol_id"] == "FloodGNN-CEUS-ZZ-BJ-v1.0"
    assert split["spatial"] == {
        "block_cells": 20,
        "block_size_m": 4000,
        "outer_folds": 5,
        "inner_folds": 3,
        "buffer_m": 1000,
        "buffer_predicate": "polygon_distance_m < buffer_m",
        "loss_node_weight": 0.2,
        "loss_comparison_tolerance": 1e-12,
    }
    assert split["reveals"]["STRICT-LOC"] == {
        "budgets": [20, 50, 100, 200, "FULL"],
        "trajectory_ids": ["1", "2", "3", "4", "5"],
        "full_trajectory_id": "NA",
    }
    assert split["reveals"]["ALL-LOC"] == {
        "analysis": "E3a",
        "budgets": [100],
        "trajectory_ids": ["1", "2", "3"],
    }
    assert split["unlabeled_sample_size"] == 2000
    assert split["banks"] == {"count": 3, "positive_background_ratio": 10}
    assert split["rng"] == {
        "bit_generator": "PCG64DXSM",
        "candidate_order": "grid_id asc",
        "seed_digest": "SHA-256 first 16 bytes unsigned big-endian",
    }
    assert labels["model_interface_columns"] == ["grid_id", "label_layer"]
    assert labels["layers"]["STRICT-LOC"]["subset_of"] == "ALL-LOC"
    assert labels["privacy_status"] == "processing-only"


def test_design_report_discloses_limited_label_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    for name in ("split_config.yaml", "label_schema.yaml"):
        (config_root / name).write_bytes((ROOT / "configs" / name).read_bytes())

    city_report = {"statuses": [], "outer_folds": []}
    monkeypatch.setattr(
        evaluation_design,
        "_city_design",
        lambda root, city_id, split: ({"city_id": city_id, **city_report}, []),
    )

    result = evaluation_design.prepare_evaluation_design(tmp_path)
    report = json.loads(
        (tmp_path / "artifacts" / "reports" / "evaluation_design.json").read_text("utf-8")
    )

    assert report["labels_read"] is True
    assert report["label_columns_read"] == ["grid_id", "label_layer"]
    assert result["labels_read"] is True
