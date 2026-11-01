"""Synthetic invariants for weighted evaluation and matched support decomposition."""

import importlib.util
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from floodgnn_ceus.inference import weighted_auprc

SCRIPTS = Path(__file__).parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "final_review", SCRIPTS / "analyze_final_review_evaluation.py"
)
analysis = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def test_constant_scores_weighted_prevalence_and_midpoints():
    y = np.array([1, 1, 0, 0, 0, 0])
    stats = analysis.positive_statistics(y, np.ones(6), 10 * 2 / 4)
    assert stats[:, 0].sum() == pytest.approx(1 / 11)
    np.testing.assert_array_equal(stats[:, 1:], 0.5)
    natural = analysis.positive_statistics(y, np.ones(6))
    assert natural[:, 0].sum() == pytest.approx(1 / 3)
    np.testing.assert_array_equal(natural[:, 1:], stats[:, 1:])


def test_weighted_ap_ties_matches_independent_threshold_reference():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([3, 3, 2, 1, 2, 1])
    weight = 2.7
    stats = analysis.positive_statistics(y, scores, weight)
    expected = 0.0
    for threshold in np.unique(scores):
        above = scores >= threshold
        tp, fp = y[above].sum(), (1 - y[above]).sum()
        expected += y[scores == threshold].sum() / y.sum() * tp / (tp + weight * fp)
    assert stats[:, 0].sum() == pytest.approx(expected)
    pair_auc = np.mean(
        [float(p > b) + 0.5 * float(p == b) for p in scores[y == 1] for b in scores[y == 0]]
    )
    assert stats[:, 2].mean() == pytest.approx(pair_auc)


def test_perfect_order_and_invalid_inputs():
    result = analysis.positive_statistics(np.array([0, 0, 1, 1]), np.arange(4), 15)
    assert result[:, 0].sum() == 1
    assert result[:, 2].mean() == 1
    with pytest.raises(ValueError, match="finite"):
        analysis.positive_statistics(np.array([0, 1]), np.array([1, np.nan]))
    with pytest.raises(ValueError, match="both binary"):
        analysis.positive_statistics(np.ones(2), np.arange(2))


def synthetic_units():
    rows = []
    for fold, trajectory, seed, variant in product(
        range(5), "12345", analysis.SEEDS, (*analysis.VARIANTS, "A*")
    ):
        rows.append(
            {
                "city": "synthetic",
                "outer_fold": fold,
                "trajectory": trajectory,
                "seed": seed,
                "variant": variant,
                **dict.fromkeys(analysis.ALL_METRICS, fold / 10),
            }
        )
    return pd.DataFrame(rows)


def test_equal_fold_aggregation_does_not_inflate_seed_independence():
    units = synthetic_units()
    folds, effects = analysis.aggregate_metrics(units)
    summary, lofo = analysis.describe_folds(folds, ["city", "variant", "metric"], "value")
    assert summary.n_folds.eq(5).all()
    assert np.allclose(summary["mean"], 0.2)
    assert effects.value.eq(0).all()
    assert len(lofo) == len(summary) * 5
    with pytest.raises(ValueError, match="complete fold"):
        analysis.aggregate_metrics(units.drop(index=0))
    with pytest.raises(ValueError, match="duplicate"):
        analysis.aggregate_metrics(pd.concat([units, units.iloc[:1]]))


def test_structure_empty_group_zero_ap_missing_rank_and_exact_additivity():
    diagnostics, memberships = {}, {}
    for fold, trajectory, seed in product(range(5), "12345", analysis.SEEDS):
        memberships["synthetic", fold] = np.array(["Z0", "Zplus"])
        variants = {}
        for variant in analysis.STRUCTURE_VARIANTS:
            value = 0.2 if variant == "FF-G2-1" else 0.1
            variants[variant] = {
                condition: np.full((2, 3), value) for condition in analysis.CONDITIONS
            }
        diagnostics["synthetic", fold, trajectory, seed] = variants
    folds, measurements = analysis.structure_tables(diagnostics, memberships)
    empty = folds.loc[folds.group.eq("Zcut")]
    assert empty.loc[empty.metric.eq("ap_contribution"), "value"].eq(0).all()
    assert empty.loc[empty.metric.ne("ap_contribution"), "value"].isna().all()
    summary, _ = analysis.summarize_structure(folds)
    assert (
        summary.loc[summary.group.eq("Zcut") & summary.metric.eq("ap_contribution"), "status"]
        .eq("complete")
        .all()
    )
    assert (
        summary.loc[summary.group.eq("Zcut") & summary.metric.ne("ap_contribution"), "status"]
        .eq("missing_fold")
        .all()
    )
    assert (
        summary.loc[summary.group.eq("Z0") & summary.metric.ne("ap_contribution"), "status"]
        .eq("sparse")
        .all()
    )
    chosen = measurements.loc[measurements.contrast.eq("FF-G2-1 - FF-G0-1")]
    assert np.allclose(
        chosen.groupby(["outer_fold", "trajectory", "seed", "condition"]).ap_contribution.sum(), 0.2
    )
    del diagnostics["synthetic", 0, "1", "NEURAL-1"]["FF-G0-1"]
    with pytest.raises(ValueError, match="missing paired"):
        analysis.structure_tables(diagnostics, memberships)


def test_weighted_ap_is_not_finite_bank_expectation():
    # One positive at the middle score, one high and one low background; sample one background.
    full = analysis.positive_statistics(np.array([0, 1, 0]), np.array([0, 1, 2]), 0.5)
    bank_aps = [
        analysis.positive_statistics(np.array([1, 0]), np.array([1, b]))[:, 0].sum() for b in [0, 2]
    ]
    assert full[:, 0].sum() == pytest.approx(2 / 3)
    assert np.mean(bank_aps) == pytest.approx(3 / 4)


def test_weighted_contributions_match_frozen_evaluator_with_ties():
    rng = np.random.default_rng(11)
    labels = np.r_[np.ones(13, dtype=int), np.zeros(287, dtype=int)]
    scores = rng.integers(0, 10, len(labels)) / 10
    background_weight = 10 * labels.sum() / (1 - labels).sum()
    weights = np.where(labels == 1, 1, background_weight)
    stats = analysis.positive_statistics(labels, scores, background_weight)
    assert stats[:, 0].sum() == pytest.approx(weighted_auprc(labels, scores, weights))


def test_weighted_condition_preserves_full_domain_ranks_and_positive_alignment():
    rows = []
    for bank, indices in [("NA", range(30)), ("1", range(22)), ("2", range(22)), ("3", range(22))]:
        for index in indices:
            rows.append(
                {
                    "bank_id": bank,
                    "grid_id": f"cell-{index:02d}",
                    "score": index % 4,
                    "sample_role": "NA"
                    if bank == "NA"
                    else "positive"
                    if index < 2
                    else "background",
                }
            )
    frame = pd.DataFrame(rows).sample(frac=1, random_state=3)
    metrics, details = analysis.evaluate_rows(frame, frozenset({"cell-00", "cell-01"}))
    assert metrics["background_weight_1to10"] == pytest.approx(20 / 28)
    np.testing.assert_array_equal(
        details["full_domain"][:, 1:], details["full_weighted_1to10"][:, 1:]
    )
    assert details["sampled_1to10"].shape == (2, 3)


@pytest.mark.parametrize("wide", [True, False])
def test_reference_summary_consumes_actual_producer_schema(tmp_path, wide):
    from summarize_reviewer_training import aggregate_units

    units = pd.concat(
        [synthetic_units().assign(city=city) for city in ("zhengzhou", "beijing")],
        ignore_index=True,
    )
    selections = (
        units.loc[units.variant.ne("A*")]
        .drop_duplicates(["city", "outer_fold", "trajectory", "variant"])
        .assign(actual_fits=3)
    )
    selections["selected_as_alternative"] = selections.variant.eq("MLP")
    produced, _ = aggregate_units(units, selections)
    current, _ = analysis.aggregate_metrics(units)
    directory = tmp_path / "archive"
    directory.mkdir()
    if wide:
        reference = produced
        filename = "method_folds.csv"
    else:
        reference = produced.melt(
            id_vars=["city", "outer_fold", "variant"],
            value_vars=list(analysis.METRICS),
            var_name="metric",
            value_name="value",
        )
        filename = "methods_folds.csv"
    path = directory / filename
    reference.to_csv(path, index=False)
    hashes = {}
    qa = analysis.check_previous(tmp_path, current, pd.DataFrame(), hashes, Path("archive"))
    assert qa["previous_metric_fold_values"] == 700
    assert qa["maximum_metric_error"] < 1e-12
    assert set(hashes) == {f"archive/{filename}"}
    reference.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="700 paired"):
        analysis.check_previous(tmp_path, current, pd.DataFrame(), {}, Path("archive"))
    pd.concat([reference, reference.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        analysis.check_previous(tmp_path, current, pd.DataFrame(), {}, Path("archive"))
    reference.loc[0, "sampled_pb_ap" if wide else "value"] += 0.01
    reference.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree"):
        analysis.check_previous(tmp_path, current, pd.DataFrame(), {}, Path("archive"))


def test_no_reference_does_not_read_historical_outputs(tmp_path):
    hashes = {}
    assert analysis.check_previous(tmp_path, pd.DataFrame(), pd.DataFrame(), hashes) == {
        "previous_metric_fold_values": 0,
        "previous_structure_fold_values": 0,
    }
    assert hashes == {}
    with pytest.raises(FileNotFoundError):
        analysis.check_previous(tmp_path, pd.DataFrame(), pd.DataFrame(), {}, Path("missing"))


def test_explicit_structure_summary_preserves_sign_and_pairing(tmp_path):
    rows = []
    for city, fold, group, comparison, metric in product(
        ("zhengzhou", "beijing"),
        range(5),
        ("Z0", "Zcut", "Zplus"),
        ("G2-G0", "G2-Fixed-filter"),
        (
            "ap_contribution",
            "fullbank_midrank_gain",
            "background_mean_gain",
            "background_median_gain",
        ),
    ):
        rows.append(
            {
                "city": city,
                "outer_fold": fold,
                "group": group,
                "comparison": comparison,
                "metric": metric,
                "analysis": "A",
                "value": 0.02,
            }
        )
    previous = pd.DataFrame(rows)
    path = tmp_path / "metric_folds.csv"
    previous.to_csv(path, index=False)
    current = previous.copy()
    current["contrast"] = current.comparison.map(
        {
            "G2-G0": "SAGE-G2-2 - SAGE-G0-2",
            "G2-Fixed-filter": "FF-G2-1 - SAGE-G2-2",
        }
    )
    current.loc[current.comparison.eq("G2-Fixed-filter"), "value"] *= -1
    current["metric"] = current.metric.replace({"fullbank_midrank_gain": "candidate_midrank_gain"})
    current["condition"] = "sampled_1to10"
    qa = analysis.check_previous(
        tmp_path,
        pd.DataFrame(),
        current,
        {},
        reference_structure_summary=Path("metric_folds.csv"),
    )
    assert qa["previous_structure_fold_values"] == 240
    with pytest.raises(ValueError, match="240"):
        analysis.check_previous(
            tmp_path,
            pd.DataFrame(),
            current.iloc[:-1],
            {},
            reference_structure_summary=Path("metric_folds.csv"),
        )
    current.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="disagree"):
        analysis.check_previous(
            tmp_path,
            pd.DataFrame(),
            current,
            {},
            reference_structure_summary=Path("metric_folds.csv"),
        )
