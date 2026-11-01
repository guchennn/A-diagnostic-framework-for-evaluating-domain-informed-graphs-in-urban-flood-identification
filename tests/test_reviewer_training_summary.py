"""Synthetic invariants for the complete-run training postprocessor."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "reviewer_training_summary", SCRIPTS / "summarize_reviewer_training.py"
)
summary = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(summary)
finally:
    sys.path.pop(0)


def prediction_rows() -> pd.DataFrame:
    rows = []
    for seed in summary.SEEDS:
        for index in range(20):
            rows.append(
                {
                    "seed": seed,
                    "bank_id": "NA",
                    "grid_id": f"cell-{index:02d}",
                    "sample_role": "NA",
                    "score": 1.0,
                }
            )
        for bank in "123":
            for index in range(11):
                rows.append(
                    {
                        "seed": seed,
                        "bank_id": bank,
                        "grid_id": f"cell-{index:02d}",
                        "sample_role": "positive" if index == 0 else "background",
                        "score": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_tree_seed_copies_are_verified_not_counted_as_independent():
    frame = prediction_rows()
    copies = summary.validated_seed_frames(frame, True, "NEURAL-1")
    assert len(copies) == 3
    for _, rows in copies:
        metrics, *_ = summary.metrics_and_signature(rows)
        assert metrics["sampled_pb_ap"] == pytest.approx(1 / 11)
        assert metrics["full_background_ap"] == pytest.approx(1 / 20)
        assert metrics["domain_count"] == 20
        assert metrics["recall_top_1pct"] == pytest.approx(0.01)
        assert metrics["recall_top_5pct"] == pytest.approx(0.05)
        assert metrics["recall_top_10pct"] == pytest.approx(0.1)
    changed = frame.copy()
    changed.loc[changed.seed.eq("NEURAL-2") & changed.grid_id.eq("cell-00"), "score"] = 2
    with pytest.raises(ValueError, match="copies differ"):
        summary.validated_seed_frames(changed, True, "NEURAL-1")


def test_missing_seed_and_duplicate_rows_fail():
    frame = prediction_rows()
    with pytest.raises(ValueError, match="seed slots"):
        summary.validated_seed_frames(frame.loc[frame.seed.ne("NEURAL-3")], True, "NEURAL-1")
    with pytest.raises(ValueError, match="duplicate"):
        summary.validated_seed_frames(pd.concat([frame, frame.iloc[:1]]), True, "NEURAL-1")


def test_bank_metrics_and_full_background_cannot_be_interchanged():
    frame = prediction_rows().query("seed == 'NEURAL-1'").copy()
    incomplete = frame.loc[~(frame.bank_id.eq("1") & frame.grid_id.eq("cell-10"))]
    with pytest.raises(ValueError, match="1:10"):
        summary.metrics_and_signature(incomplete)
    frame.loc[frame.bank_id.eq("1") & frame.grid_id.eq("cell-00"), "score"] = 0.5
    with pytest.raises(ValueError, match="predictions differ"):
        summary.metrics_and_signature(frame)


def complete_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    units, selections = [], []
    for city in summary.CITIES:
        for fold in range(5):
            for trajectory in "12345":
                for variant in (*summary.VARIANTS, "A*"):
                    for seed in summary.SEEDS:
                        units.append(
                            {
                                "city": city,
                                "outer_fold": fold,
                                "trajectory": trajectory,
                                "variant": variant,
                                "seed": seed,
                                **dict.fromkeys(summary.METRICS, float(fold == 4)),
                            }
                        )
                    if variant != "A*":
                        selections.append(
                            {
                                "city": city,
                                "outer_fold": fold,
                                "trajectory": trajectory,
                                "variant": variant,
                                "actual_fits": 1 if variant in summary.DETERMINISTIC else 3,
                                "selected_as_alternative": variant == "XGBoost",
                            }
                        )
    return pd.DataFrame(units), pd.DataFrame(selections)


def test_complete_design_has_equal_fold_means_and_real_fit_counts():
    units, selections = complete_metrics()
    folds, effects = summary.aggregate_units(units, selections)
    assert folds.loc[folds.variant.eq("XGBoost"), "actual_fits"].eq(5).all()
    assert folds.loc[folds.variant.eq("A*"), "actual_fits"].eq(5).all()
    assert folds.loc[folds.variant.eq("SAGE-G2-2"), "actual_fits"].eq(15).all()
    assert folds.groupby(["city", "variant"]).sampled_pb_ap.mean().eq(0.2).all()
    assert effects.effect.eq(0).all()
    described, lofo = summary.describe_folds(effects, ["city", "contrast", "metric"], "effect")
    assert described.n_folds.eq(5).all()
    assert described.resampling_q025.eq(0).all()
    assert described.resampling_q975.eq(0).all()
    assert len(lofo) == len(described) * 5


def test_partial_outer_matrix_cannot_produce_summary():
    units, selections = complete_metrics()
    with pytest.raises(ValueError, match="complete fold"):
        summary.aggregate_units(units.iloc[1:], selections)


def test_perfect_order_and_boundary_ties_do_not_depend_on_row_order():
    labels = np.r_[np.ones(10), np.zeros(90)]
    scores = np.arange(100)[::-1]
    metrics = summary.ranking_metrics(labels, scores)
    assert metrics["ap"] == 1
    assert metrics["recall_top_1pct"] == 0.1
    assert metrics["recall_top_5pct"] == 0.5
    assert metrics["recall_top_10pct"] == 1
    ties = np.ones(100)
    assert summary.ranking_metrics(labels, ties) == summary.ranking_metrics(labels[::-1], ties)


def test_hash_mismatch_and_failed_task_rejected(tmp_path):
    variant = {"id": "XGBoost", "model": "XGBoost", "graph": "NA"}
    task = summary.task_key(("beijing", 0, "1"), variant, "SMALL", "NEURAL-1", "test")
    result = tmp_path / "outer/tasks" / task.relative_path() / "oof.parquet"
    result.parent.mkdir(parents=True)
    result.write_bytes(b"synthetic bytes; hash check precedes parquet parsing")
    status = result.with_name("status.json")
    payload = {"status": "success", "task": task.as_dict(), "oof_sha256": "invalid"}
    status.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        summary.verify_task(tmp_path, "outer", task, {}, tmp_path)
    payload["status"] = "failed"
    status.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsuccessful"):
        summary.verify_task(tmp_path, "outer", task, {}, tmp_path)


def test_dry_inventory_reads_no_metrics_and_reports_incomplete(tmp_path):
    result = summary.inventory(tmp_path)
    assert result["metrics_read"] is False
    assert result["counts"] == {"inner": {}, "outer": {}}
    assert result["expected_inner"] == 1250
    assert result["expected_outer"] == 1550
    assert result["selection_files"] == 0


def test_external_junction_keeps_logical_repository_provenance_path(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    run = root / "artifacts" / "runs" / "reviewer-revision-v1-test"
    external = tmp_path / "other-volume" / run.name
    original_resolve = Path.resolve

    def simulated_junction(path, *args, **kwargs):
        if path == run:
            return external
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", simulated_junction)
    assert not run.resolve().is_relative_to(root)
    logical = summary.logical_run_path(root, run)
    assert logical == run
    assert (logical / "plan.json").relative_to(root).as_posix() == (
        "artifacts/runs/reviewer-revision-v1-test/plan.json"
    )


def test_relative_run_entry_is_normalized_without_resolving_link(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = Path("artifacts/runs/unused/../reviewer-revision-v1-test")
    assert summary.logical_run_path(tmp_path, relative) == (
        tmp_path / "artifacts/runs/reviewer-revision-v1-test"
    )


@pytest.mark.parametrize(
    "path",
    [
        "outside/reviewer-revision-v1-test",
        "artifacts/reviewer-revision-v1-test",
        "artifacts/runs/../reviewer-revision-v1-test",
        "artifacts/runs/nested/reviewer-revision-v1-test",
        "artifacts/runs/unrelated-run",
    ],
)
def test_run_path_must_use_the_exact_repository_entry(tmp_path, path):
    with pytest.raises(ValueError, match="must use the repository"):
        summary.logical_run_path(tmp_path, tmp_path / path)


@pytest.mark.parametrize("bad_config,bad_score", [("MEDIUM", 0.25), ("SMALL", 0.3)])
def test_saved_selection_is_rechecked_from_inner_scores_before_outer_read(
    tmp_path, monkeypatch, bad_config, bad_score
):
    variant = {"id": "XGBoost", "model": "XGBoost", "graph": "NA"}
    choices = {variant_id: {} for variant_id in summary.VARIANTS}
    choices["XGBoost"] = {
        **variant,
        "config": bad_config,
        "score": bad_score,
        "best_epochs": [0, 0, 0],
        "refit_epochs": 0,
    }
    directory = tmp_path / "selections"
    directory.mkdir()
    (directory / "zhengzhou-f0-b100-t1.json").write_text(
        json.dumps(
            {
                "unit": {"city": "zhengzhou", "outer_fold": 0, "budget": "100", "trajectory": "1"},
                "variants": choices,
                "alternative_order": summary.ALTERNATIVES,
            }
        ),
        encoding="utf-8",
    )

    def synthetic_inner(run, phase, task, hashes, root):
        assert phase == "inner", "outer predictions must not be read before selection validation"
        path = tmp_path / f"{task.config}.parquet"
        pd.DataFrame(
            [
                {
                    **task.as_dict(),
                    "inner_fold": fold,
                    "score": 0.25,
                    "best_epoch": 0,
                    "status": "success",
                }
                for fold in range(3)
            ]
        ).to_parquet(path)
        return path

    monkeypatch.setattr(summary, "verify_task", synthetic_inner)
    with pytest.raises(ValueError, match="saved selection differs"):
        summary.validate_selections(
            tmp_path, tmp_path, {"variants": [variant]}, {"protocol_id": "test"}, {}
        )
