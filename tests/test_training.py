from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from floodgnn_ceus.models import fixed_filter_features, neighbor_features
from floodgnn_ceus.training import (
    OOF_COLUMNS,
    TASK_FIELDS,
    TaskKey,
    balanced_sample_weights,
    best_alternative,
    fit_audit,
    fit_preprocessor,
    outer_refit_epochs,
    seed_from_name,
    task_action,
    write_oof,
)


def test_preprocessor_fits_only_supplied_training_rows() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, 1000.0],
            "b": [2.0, 4.0, -1000.0],
        },
        index=["p", "u", "test"],
    )

    preprocessor = fit_preprocessor(frame, ["p", "u"], scale=True)
    transformed = preprocessor.transform(frame)

    assert preprocessor.medians.tolist() == [1.0, 3.0]
    assert transformed.loc["p", "a"] == pytest.approx(0.0)
    assert transformed.loc["u", "a"] == pytest.approx(0.0)
    assert transformed.loc["test", "a"] > 100.0


def test_preprocessor_rejects_all_missing_training_column() -> None:
    frame = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]}, index=["p", "test"])

    with pytest.raises(ValueError, match="all-missing"):
        fit_preprocessor(frame, ["p"], scale=False)


def test_fit_audit_hashes_rows_preprocessing_and_matrices_without_values() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 4.0]}, index=["p", "u"])
    preprocessor = fit_preprocessor(frame, ["p", "u"], scale=True)
    matrix = preprocessor.transform(frame).to_numpy()

    train_edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    evaluation_edges = np.array([[2], [3]], dtype=np.int64)
    first = fit_audit(
        ["p", "u"],
        preprocessor,
        evaluation_cut_edge_index=evaluation_edges,
        fit_matrix=matrix,
        train_cut_edge_index=train_edges,
    )
    second = fit_audit(
        ["p", "u"],
        preprocessor,
        evaluation_cut_edge_index=evaluation_edges.copy(),
        fit_matrix=matrix.copy(),
        train_cut_edge_index=train_edges.copy(),
    )

    assert first == second
    assert first["fit_row_count"] == 2
    assert first["preprocessing"] == preprocessor.record()
    assert first["matrices"]["fit_matrix"]["shape"] == [2, 2]
    assert first["matrices"]["train_cut_edge_index"]["shape"] == [2, 2]
    changed = fit_audit(
        ["p", "u"],
        preprocessor,
        evaluation_cut_edge_index=np.array([[2], [4]], dtype=np.int64),
        fit_matrix=matrix,
        train_cut_edge_index=train_edges,
    )
    assert (
        first["matrices"]["evaluation_cut_edge_index"]["sha256"]
        != changed["matrices"]["evaluation_cut_edge_index"]["sha256"]
    )
    assert "values" not in first["matrices"]["fit_matrix"]


def test_training_weights_balance_positive_and_u_groups() -> None:
    weights = balanced_sample_weights(positive_count=20, unlabeled_count=2000)

    assert weights[:20].sum() == pytest.approx(0.5)
    assert weights[20:].sum() == pytest.approx(0.5)
    assert len(weights) == 2020


def test_task_key_preserves_all_frozen_dimensions_and_na() -> None:
    values = {field: "NA" for field in TASK_FIELDS}
    values.update(
        protocol_id="lock-id",
        experiment_id="E1",
        city="zhengzhou",
        outer_fold="0",
        budget="20",
        trajectory="1",
        model="KDE",
    )
    key = TaskKey(**values)

    assert list(key.as_dict()) == list(TASK_FIELDS)
    assert key.relative_path().as_posix().startswith("experiment=E1/city=zhengzhou/task=")
    assert seed_from_name("same") == seed_from_name("same")
    assert seed_from_name("same") != seed_from_name("other")


def test_fixed_filter_uses_incoming_g2_neighbors_and_self() -> None:
    x = np.array([[2.0], [4.0], [10.0]])
    edges = np.array([[0, 1], [1, 2]])

    filtered = fixed_filter_features(x, edges)

    assert filtered[:, 1].tolist() == [2.0, 3.0, 7.0]


def test_neighbor_features_use_only_supplied_partition_edges() -> None:
    raw = np.array([[1.0], [3.0], [100.0]])
    imputed = raw.copy()
    edges = np.array([[0, 1], [1, 0]])

    derived = neighbor_features(raw, imputed, edges)

    assert derived.shape == (3, 6)
    assert derived[0].tolist() == [1.0, 3.0, 0.0, -2.0, 1.0, 1.0]
    assert derived[2].tolist() == [100.0, 100.0, 0.0, 0.0, 0.0, 0.0]


def test_best_alternative_uses_complete_inner_scores_and_fixed_tie_order() -> None:
    rows = []
    for model in ("KDE", "XGBoost", "LightGBM", "Neighbor-XGB", "MLP", "Fixed-filter"):
        for inner_fold in range(3):
            rows.append(
                {
                    "model": model,
                    "config": "NA" if model == "KDE" else "SMALL",
                    "inner_fold": inner_fold,
                    "score": 0.5,
                    "status": "success",
                }
            )
    scores = pd.DataFrame(rows)

    selected = best_alternative(scores)

    assert selected == {"model": "KDE", "config": "NA", "score": 0.5}


def test_best_alternative_rejects_an_incomplete_required_family() -> None:
    rows = []
    for model in ("KDE", "XGBoost", "LightGBM", "Neighbor-XGB", "MLP", "Fixed-filter"):
        folds = range(2) if model == "MLP" else range(3)
        for inner_fold in folds:
            rows.append(
                {
                    "model": model,
                    "config": "NA" if model == "KDE" else "SMALL",
                    "inner_fold": inner_fold,
                    "score": 0.5,
                    "status": "success",
                }
            )

    with pytest.raises(ValueError, match="MLP"):
        best_alternative(pd.DataFrame(rows))


def test_best_alternative_prefers_small_on_an_exact_config_tie() -> None:
    rows = []
    for model in ("KDE", "XGBoost", "LightGBM", "Neighbor-XGB", "MLP", "Fixed-filter"):
        configs = ("NA",) if model == "KDE" else ("SMALL", "MEDIUM")
        for config in configs:
            for inner_fold in range(3):
                rows.append(
                    {
                        "model": model,
                        "config": config,
                        "inner_fold": inner_fold,
                        "score": 0.5,
                        "status": "success",
                    }
                )

    selected = best_alternative(pd.DataFrame(rows))

    assert selected["config"] == "NA"
    complete = pd.DataFrame(rows)
    complete.loc[complete["model"].eq("KDE"), "score"] = 0.4
    assert best_alternative(complete)["config"] == "SMALL"


def test_outer_epoch_and_resume_contracts_are_explicit() -> None:
    assert outer_refit_epochs([11, 3, 7]) == 7
    assert task_action(None) == "run"
    assert task_action("failed") == "retry"
    assert task_action("success") == "skip"
    assert task_action("not-estimable") == "skip"
    with pytest.raises(ValueError, match="status"):
        task_action("unknown")


def test_task_key_rejects_path_separators() -> None:
    values = {field: "NA" for field in TASK_FIELDS}
    values["protocol_id"] = "bad/path"

    with pytest.raises(ValueError, match="task key"):
        TaskKey(**values)


def test_oof_shards_are_schema_locked_and_immutable(tmp_path: Path) -> None:
    row = {field: "NA" for field in TASK_FIELDS}
    row.update(
        bank_id=1,
        grid_id="node-1",
        sample_role="positive",
        score=0.25,
        status="success",
    )
    frame = pd.DataFrame([row], columns=OOF_COLUMNS)
    path = tmp_path / "oof.parquet"

    assert write_oof(frame, path) == path
    assert write_oof(frame, path) == path
    changed = frame.copy()
    changed.loc[0, "score"] = 0.5
    with pytest.raises(FileExistsError, match="different content"):
        write_oof(changed, path)


def test_full_test_oof_rows_use_explicit_na_bank_dimension(tmp_path: Path) -> None:
    row = {field: "NA" for field in TASK_FIELDS}
    row.update(
        bank_id="NA",
        grid_id="node-1",
        sample_role="NA",
        score=0.25,
        status="success",
    )

    write_oof(pd.DataFrame([row], columns=OOF_COLUMNS), tmp_path / "full_test.parquet")
