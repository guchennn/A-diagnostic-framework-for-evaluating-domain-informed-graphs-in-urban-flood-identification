import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from floodgnn_ceus.models import fixed_filter_features


def test_fixed_filter_default_and_two_steps_preserve_36_dimensions():
    x = np.arange(54, dtype=float).reshape(3, 18)
    edges = np.array([[0, 1], [1, 2]])
    first = np.vstack([x[0], (x[0] + x[1]) / 2, (x[1] + x[2]) / 2])
    second = np.vstack([first[0], (first[0] + first[1]) / 2, (first[1] + first[2]) / 2])
    np.testing.assert_array_equal(fixed_filter_features(x, edges), np.c_[x, first])
    np.testing.assert_array_equal(fixed_filter_features(x, edges, steps=2), np.c_[x, second])


@pytest.mark.parametrize("steps", [1, 2])
def test_g0_duplicates_self_features(steps):
    x = np.arange(54, dtype=float).reshape(3, 18)
    np.testing.assert_array_equal(
        fixed_filter_features(x, np.empty((2, 0), dtype=int), steps=steps), np.c_[x, x]
    )


def test_direction_changes_filter_and_undirected_g1_is_symmetric():
    x = np.array([[0.0], [4.0]])
    np.testing.assert_array_equal(fixed_filter_features(x, np.array([[0], [1]]))[:, 1], [0, 2])
    np.testing.assert_array_equal(fixed_filter_features(x, np.array([[1], [0]]))[:, 1], [2, 4])
    np.testing.assert_array_equal(
        fixed_filter_features(x, np.array([[0, 1], [1, 0]]))[:, 1], [2, 2]
    )


@pytest.mark.parametrize("steps", [0, 3, -1, 1.5, True])
def test_invalid_filter_depth_is_rejected(steps):
    with pytest.raises(ValueError, match="steps"):
        fixed_filter_features(np.zeros((2, 18)), np.empty((2, 0), dtype=int), steps=steps)


def test_revision_plan_is_blind_and_separate_from_old_caches():
    from floodgnn_ceus.train import prepare_plan

    plan = prepare_plan(Path(__file__).resolve().parents[1])
    assert plan["outer_units"] == 50
    assert plan["inner_candidate_tasks"] == 1250
    assert plan["inner_fold_fits"] == 3750
    assert plan["outer_refits"] == 1550
    assert plan["labels_read"] is False
    assert plan["preregistered"] is False
    assert plan["run_namespace"].startswith("reviewer-revision-v1-")


def test_missing_revision_environment_blocks_before_loading_data(tmp_path, monkeypatch):
    from floodgnn_ceus import train as revision

    monkeypatch.setattr(
        revision.e1, "_read_city", lambda *args: pytest.fail("data read before gate")
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        revision.run_batch(tmp_path, phase="inner", start=0, limit=1)


@pytest.mark.parametrize("layers", [0, 3, -1, True, 1.5])
def test_invalid_graphsage_depth_fails_before_torch_import(layers):
    from floodgnn_ceus.models import train_neural

    with pytest.raises(ValueError, match="layers"):
        train_neural(
            "GraphSAGE-G2",
            {},
            None,
            None,
            None,
            None,
            None,
            None,
            seed=1,
            max_epochs=1,
            patience=1,
            layers=layers,
        )


@pytest.fixture
def revision_root(tmp_path):
    source = Path(__file__).resolve().parents[1]
    for relative in (
        "configs/matched_models.yaml",
        "configs/model_grid.yaml",
        "uv.lock",
        "pyproject.toml",
        "artifacts/protocol_lock.json",
        "src/floodgnn_ceus/models.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "artifacts/protocol_lock.json":
            target.write_text('{"effective_protocol_id": "synthetic-test-protocol"}')
        else:
            shutil.copyfile(source / relative, target)
    return tmp_path


def test_source_change_cannot_reuse_cached_revision(revision_root):
    from floodgnn_ceus import train as revision

    first = revision.prepare_plan(revision_root)
    path = revision_root / "src/floodgnn_ceus/models.py"
    path.write_text(path.read_text("utf-8") + "\n# changed code\n", encoding="utf-8")
    second = revision.prepare_plan(revision_root)
    assert first["run_namespace"] != second["run_namespace"]


def test_archived_environment_lock_does_not_authorize_local_revision(revision_root, monkeypatch):
    from floodgnn_ceus import train as revision

    (revision_root / "artifacts/environment_lock.json").write_text('{"status":"success"}')
    monkeypatch.setattr(revision.e1, "_read_city", lambda *args: pytest.fail("read before gate"))
    with pytest.raises(FileNotFoundError, match="local_environment_lock"):
        revision.run_batch(revision_root, limit=1)


def test_local_lock_and_probe_require_matching_inputs(revision_root, monkeypatch):
    from floodgnn_ceus import train as revision
    from floodgnn_ceus.io import sha256_file

    monkeypatch.setattr(revision, "_environment", lambda: {"test": "gate-only, no model"})
    files = [{"path": "fixture", "sha256": "original"}]
    monkeypatch.setattr(revision, "_input_identity", lambda root: files)
    revision.freeze_local_environment(revision_root)
    plan = revision.prepare_plan(revision_root)
    run_root = revision._run_root(revision_root, plan)
    lock = json.loads((run_root / "local_environment_lock.json").read_text("utf-8"))
    with pytest.raises(FileNotFoundError, match="resource_probe"):
        revision.require_revision_environment(revision_root, plan)
    receipt = {
        "status": "success",
        "epochs": 1,
        "cell": revision.PROBE_CELL,
        "performance_blind": True,
        "metrics_computed": False,
        "predictions_written": False,
        "source_sha256": plan["source_sha256"],
        "input_sha256": lock["input_sha256"],
        "lock_sha256": sha256_file(run_root / "local_environment_lock.json"),
    }
    revision.e1._write_json(run_root / "resource_probe.json", receipt)
    revision.require_revision_environment(revision_root, plan)
    receipt["metrics_computed"] = True
    revision.e1._write_json(run_root / "resource_probe.json", receipt)
    with pytest.raises(ValueError, match="probe"):
        revision.require_revision_environment(revision_root, plan)
    files[0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="inputs"):
        revision.require_revision_environment(revision_root, plan)


def test_independent_variant_selection_and_original_astar_candidate_set(revision_root):
    import pandas as pd
    import yaml

    from floodgnn_ceus import train as revision
    from floodgnn_ceus.io import sha256_file

    config = revision._config(revision_root)
    grid = yaml.safe_load((revision_root / "configs/model_grid.yaml").read_text("utf-8"))
    unit = revision._units(config)[0]
    run_root = revision_root / "new-results"
    for variant in config["variants"]:
        for candidate in revision._candidates(config, variant):
            task = revision._task(grid, unit, variant, candidate, "HPO-1")
            path = revision.e1._status_path(run_root / "inner", task)
            path.parent.mkdir(parents=True, exist_ok=True)
            score = 0.5 if candidate == "MEDIUM" else 0.3
            if variant["id"] == "FF-G1-1":
                score = 0.99  # Excluded from A*: it is a separate post-review control.
            rows = [
                {
                    **task.as_dict(),
                    "inner_fold": f,
                    "score": score,
                    "best_epoch": 10 + f,
                    "status": "success",
                }
                for f in range(3)
            ]
            result_path = path.with_name("inner_scores.parquet")
            pd.DataFrame(rows).to_parquet(result_path, index=False)
            revision.e1._write_json(
                path, {"status": "success", "result_sha256": sha256_file(result_path)}
            )
    selected = revision.select_revision_unit(revision_root, run_root, unit)
    assert len(selected["variants"]) == 13
    assert selected["best_alternative"]["id"] == "XGBoost"
    assert selected["variants"]["FF-G1-1"]["config"] == "SMALL"
    assert selected["variants"]["FF-G2-1"]["config"] == "MEDIUM"
    assert selected["variants"]["FF-G2-1"]["refit_epochs"] == 11
    assert selected["variants"]["SAGE-G2-1"]["layers"] == 1
    # The selection path never opens any outer predictions.
    (run_root / "outer").mkdir()
    (run_root / "outer" / "oof.parquet").write_text("invalid outer data must never be read")
    assert revision.select_revision_unit(revision_root, run_root, unit) == selected
    result_path.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="hash"):
        revision.select_revision_unit(revision_root, run_root, unit)


def test_revision_xgboost_parameters_are_explicit_and_keep_split_fix(revision_root):
    import yaml

    from floodgnn_ceus import train as revision

    config = revision._config(revision_root)
    grid = yaml.safe_load((revision_root / "configs/model_grid.yaml").read_text("utf-8"))
    for variant in config["variants"]:
        if variant["model"] in ("XGBoost", "Neighbor-XGB"):
            changed = revision._variant_grid(grid, variant, config)
            for candidate in config["configs"]:
                params = changed["models"][variant["model"]][candidate]
                assert (params["reg_lambda"], params["reg_alpha"], params["gamma"]) == (1, 0, 0)
                assert "min_child_weight" not in params  # Existing predictor forces zero.
    assert config["xgboost_fixed"]["min_child_weight"] == 0


@pytest.mark.parametrize(
    "model,graph,depth",
    [
        ("Fixed-filter", "G0", 1),
        ("Fixed-filter", "G1", 1),
        ("Fixed-filter", "G2", 1),
        ("Fixed-filter", "G2", 2),
        ("GraphSAGE-G2", "G0", 2),
        ("GraphSAGE-G2", "G1", 2),
        ("GraphSAGE-G2", "G2", 1),
        ("GraphSAGE-G2", "G2", 2),
    ],
)
def test_inner_outer_routing_uses_requested_graph_and_depth(model, graph, depth, monkeypatch):
    import pandas as pd
    import yaml

    from floodgnn_ceus import e1
    from floodgnn_ceus import train as revision
    from floodgnn_ceus.models import NeuralResult
    from floodgnn_ceus.protocol import EXPECTED_FEATURE_NAMES
    from floodgnn_ceus.training import fit_preprocessor

    root = Path(__file__).resolve().parents[1]
    grid = yaml.safe_load((root / "configs/model_grid.yaml").read_text("utf-8"))
    grid.update(fixed_filter_steps=depth, graphsage_layers=depth)
    unit = revision.E1OuterUnit("zhengzhou", 0, "100", "1")
    task = revision._task(
        grid, unit, {"id": "test", "model": model, "graph": graph}, "SMALL", "HPO-1"
    )
    ids = ["a", "b", "c", "d"]
    raw = pd.DataFrame(np.arange(72).reshape(4, 18), index=ids, columns=EXPECTED_FEATURE_NAMES)
    banks = pd.DataFrame(
        [
            {"bank_id": bank, "grid_id": node, "node_index": index, "sample_role": role}
            for bank in (1, 2, 3)
            for node, index, role in (("c", 2, "positive"), ("d", 3, "background"))
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "source_grid_id": source,
                "target_grid_id": target,
                "partition": partition,
                "graph_id": g,
                "realization_id": "NA",
            }
            for g, pairs in (("G1", [("a", "b"), ("c", "d")]), ("G2", [("b", "a"), ("d", "c")]))
            for (source, target), partition in zip(pairs, ("train", "test"), strict=True)
        ]
    )
    fold = {
        "raw": raw,
        "banks": banks,
        "grid": pd.DataFrame(index=ids),
        "positions": dict(zip(ids, range(4), strict=True)),
        "positive_ids": ["a"],
        "u_ids": ["b"],
        "train_node_ids": ids[:2],
        "validation_node_ids": ids[2:],
        "node_ids": ids,
        "test_node_ids": ids[2:],
        "cut_edges": edges,
        "edges": edges,
    }
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return NeuralResult(np.array([0.8, 0.2]), 1, 1)

    monkeypatch.setattr(e1, "train_neural", capture)
    e1._score_inner_fold(fold, task, grid)
    e1._predict_outer(fold, task, grid, epochs=1)
    transformed = fit_preprocessor(raw, ids[:2], scale=True).transform(raw).to_numpy()
    expected_edges = {
        "G0": np.empty((2, 0), dtype=int),
        "G1": np.array([[0], [1]]),
        "G2": np.array([[1], [0]]),
    }[graph]
    for args, kwargs in calls:
        np.testing.assert_array_equal(kwargs["prediction_edge_index"], expected_edges)
        if model == "Fixed-filter":
            expected = fixed_filter_features(transformed[2:], expected_edges, steps=depth)
            np.testing.assert_array_equal(kwargs["prediction_x"], expected)
            assert args[2].shape[1] == 36
        else:
            assert kwargs["layers"] == depth
            np.testing.assert_array_equal(args[3], expected_edges)
