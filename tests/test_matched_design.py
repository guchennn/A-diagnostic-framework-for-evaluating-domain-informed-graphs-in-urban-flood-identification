"""Synthetic invariants for the reviewer's fixed-prediction reanalysis."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "train", Path(__file__).parents[1] / "scripts/metrics.py"
)
revision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(revision)


def test_constant_scores_have_prevalence_ap_and_budget_recall():
    y = np.r_[np.ones(17), np.zeros(83)]
    metrics = revision.ranking_metrics(y, np.ones(100))
    assert metrics["ap"] == pytest.approx(0.17)
    for percentage in (1, 5, 10):
        assert metrics[f"recall_top_{percentage}pct"] == pytest.approx(percentage / 100)


def test_perfect_ranking_and_fractional_budget():
    y = np.r_[np.ones(10), np.zeros(90)]
    metrics = revision.ranking_metrics(y, np.arange(100)[::-1])
    assert metrics == pytest.approx(
        {"ap": 1, "recall_top_1pct": 0.1, "recall_top_5pct": 0.5, "recall_top_10pct": 1}
    )
    small = revision.ranking_metrics(np.array([1, 0, 0]), np.array([3, 2, 1]))
    assert small["recall_top_1pct"] == pytest.approx(0.03)


def test_boundary_ties_are_permutation_invariant():
    labels = np.array([1, 0, 1, 0] + [0] * 96)
    scores = np.array([9, 9, 9, 9] + [0] * 96)
    expected = revision.ranking_metrics(labels, scores)
    order = np.random.default_rng(4).permutation(len(labels))
    assert revision.ranking_metrics(labels[order], scores[order]) == expected
    assert expected["recall_top_1pct"] == pytest.approx(0.25)


def test_missing_and_duplicate_pairs_fail_closed():
    frame = pd.DataFrame({"fold": [0, 0, 1], "method": ["G2", "G1", "G2"], "ap": [1, 0, 1]})
    with pytest.raises(ValueError, match="missing"):
        revision.paired_wide(frame, ["fold"], "method", "ap")
    with pytest.raises(ValueError, match="duplicate"):
        revision.paired_wide(pd.concat([frame, frame.iloc[:1]]), ["fold"], "method", "ap")


def test_equal_fold_weight_and_lofo_not_cell_weight():
    frame = pd.DataFrame({"contrast": "test", "outer_fold": range(5), "effect": [0, 0, 0, 0, 1]})
    summary, lofo = revision.describe_folds(frame, ["contrast"], "effect")
    assert summary["mean"].iloc[0] == pytest.approx(0.2)
    assert summary["positive_folds"].iloc[0] == 1
    assert lofo.loc[lofo.omitted_fold.eq(4), "mean"].iloc[0] == 0
    with pytest.raises(ValueError, match="exactly folds"):
        revision.describe_folds(frame.iloc[:4], ["contrast"], "effect")


def test_shared_full_and_scarce_fold_variation_cancels_in_interaction():
    full, scarce = [], []
    features = {"full": "F18", "noD8": "F16-noD8", "noNeigh": "F14-noNeigh"}
    for fold in range(5):
        for regime, feature in features.items():
            for method in ("G0", "G1", "G2", "XGB"):
                value = 0.2 + (0.01 * fold if method == "G2" and regime != "full" else 0)
                full.append(
                    {"outer_fold": fold, "regime": regime, "method": method, "pb_auprc": value}
                )
                scarce.append({"outer_fold": fold, "method": f"{method}|{feature}", "pb_ap": value})
    contrasts = revision.joint_feature_contrasts(pd.DataFrame(full), pd.DataFrame(scarce))
    assert np.allclose(contrasts.effect, 0)
    interactions = contrasts.loc[contrasts.contrast.str.startswith("interaction")]
    assert interactions.full_component.var() > 0
    summary, _ = revision.describe_folds(interactions, ["baseline", "contrast"], "effect")
    assert (summary[["mean", "resampling_q025", "resampling_q975"]] == 0).all().all()


@pytest.mark.parametrize("scores", [[0, np.nan], [0, np.inf]])
def test_nonfinite_scores_rejected(scores):
    with pytest.raises(ValueError, match="finite"):
        revision.ranking_metrics(np.array([0, 1]), np.array(scores))
