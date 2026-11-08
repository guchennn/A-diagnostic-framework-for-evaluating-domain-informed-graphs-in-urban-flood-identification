"""Protocol-critical tests for the topology runner; no GPU is used."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "topology_runner", Path(__file__).resolve().parents[1] / "scripts/run_topology_controls.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def geometry():
    return pd.DataFrame(
        {"grid_id": ["a", "b", "c", "d"], "row": [0, 0, 1, 1], "column": [0, 1, 0, 1]}
    )


def edges():
    return pd.DataFrame({"source_grid_id": ["a", "b"], "target_grid_id": ["c", "d"]})


def test_cache_identity_pairs_outer_but_separates_inner_and_realizations():
    key = runner.cache_key
    assert key("beijing", 0, "1", None, "train", 1) == key("beijing", 0, "5", None, "train", 1)
    assert key("beijing", 0, "1", 0, "train", 1) != key("beijing", 0, "5", 0, "train", 1)
    assert key("beijing", 0, "1", None, "train", 1) != key("beijing", 0, "1", None, "train", 2)


def test_graph_check_rejects_distance_and_degree_changes():
    original = edges()
    swapped = pd.DataFrame({"source_grid_id": ["a", "b"], "target_grid_id": ["d", "c"]})
    config = {"min_changed_fraction": 0.0, "preserve_global_distance_counts": True}
    with pytest.raises(ValueError, match="rook/diagonal"):
        runner.graph_invariants(original, swapped, geometry(), config)
    swapped["source_grid_id"] = ["a", "a"]
    with pytest.raises(ValueError, match="degree differs"):
        runner.graph_invariants(original, swapped, geometry(), config)
    assert runner.graph_invariants(original, original, geometry(), config)["exact_in_out_degree"]


def test_unchanged_graph_is_rejected_when_change_required():
    with pytest.raises(ValueError, match="insufficient changed"):
        runner.graph_invariants(edges(), edges(), geometry(), {"min_changed_fraction": 0.08})


def test_frozen_json_is_not_replaced(tmp_path):
    p = tmp_path / "lock.json"
    runner.immutable_json(p, {"value": 1})
    runner.immutable_json(p, {"value": 1})
    with pytest.raises(ValueError, match="refusing"):
        runner.immutable_json(p, {"value": 2})
    assert json.loads(p.read_text()) == {"value": 1}


def test_successful_receipt_requires_payload_hash(tmp_path):
    payload = tmp_path / "inner_scores.parquet"
    payload.write_bytes(b"original")
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps({"status": "success", "result_sha256": runner.sha256_file(payload)})
    )
    assert runner.succeeded(status, payload.name, "result_sha256")
    payload.write_bytes(b"modified")
    with pytest.raises(ValueError, match="payload changed"):
        runner.succeeded(status, payload.name, "result_sha256")


def test_graph_injection_preserves_partition_and_original_edges(tmp_path, monkeypatch):
    original = pd.concat(
        [
            edges().assign(graph_id="G2", partition="train"),
            edges().assign(graph_id="G2", partition="test"),
        ],
        ignore_index=True,
    )
    fold = {
        "cut_edges": original,
        "grid": geometry().set_index("grid_id"),
        "train_node_ids": ["a", "b", "c", "d"],
        "validation_node_ids": ["a", "b", "c", "d"],
    }
    observed = []

    def fake_partition(root, config, graph, geo, key, *, create):
        observed.append((key, set(geo.columns), set(graph.partition)))
        return graph[["source_grid_id", "target_grid_id"]]

    monkeypatch.setattr(runner, "partition_graph", fake_partition)
    result = runner.inject_graphs(tmp_path, {}, fold, "beijing", 0, "1", 0, 2, create=False)
    assert len(result["cut_edges"]) == 8
    pd.testing.assert_frame_equal(fold["cut_edges"], original)
    assert all(columns == {"grid_id", "row", "column"} for _, columns, _ in observed)
    assert [partition for _, _, partition in observed] == [{"train"}, {"test"}]
    assert set(result["cut_edges"].graph_id) == {"G2", "G7"}


def test_plan_and_task_identity_keep_graph_repeats_distinct():
    config = runner.configuration()
    grid = {"protocol_id": "frozen-protocol"}
    assert len(list(runner.units(config, "beijing"))) == 75
    variant = config["variants"][0]
    a = runner.task_key(config, grid, "beijing", 0, "1", variant, 1, "SMALL", "NEURAL-1")
    b = runner.task_key(config, grid, "beijing", 0, "1", variant, 2, "SMALL", "NEURAL-1")
    assert a.realization_id != b.realization_id
    assert a.seed == b.seed
    assert a.config == b.config
    assert a.graph_id == b.graph_id == "G7"
    assert a.budget == "100"


def test_public_freeze_resolves_relative_local_lock(tmp_path, monkeypatch):
    local_path = "artifacts/runs/baseline/local_environment_lock.json"
    path = tmp_path / local_path
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    monkeypatch.setattr(runner, "identity", lambda *_: {"sources": "bound"})
    monkeypatch.setattr(
        runner.revision, "freeze_local_environment", lambda *_: {"lock_path": local_path}
    )
    runner.freeze(tmp_path, {"run_id": "test"})
    lock = json.loads((tmp_path / "artifacts/runs/test/local_environment_lock.json").read_text())
    assert lock["local_revision_lock"] == local_path
    assert lock["local_revision_lock_sha256"] == runner.sha256_file(path)


def test_public_run_path_uses_local_runs(tmp_path):
    assert runner.run_path(tmp_path, {"run_id": "test"}) == tmp_path / "artifacts/runs/test"
