"""Complete-domain AP accounting and within-fold depth contrast invariants."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "major_diagnostics", SCRIPTS / "analyze_major_revision_diagnostics.py"
)
analysis = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def test_full_domain_ties_accounting_and_group_precision():
    y = np.array([1, 0, 1, 0, 0])
    a = analysis.positive_statistics(y, np.array([3, 3, 2, 1, 0]))
    b = analysis.positive_statistics(y, np.zeros(5))
    np.testing.assert_array_equal(b[:, 1:], 0.5)
    members = pd.DataFrame({"A": ["Zplus", "Z0"], "L": ["1", "0"], "K": ["1", "0"]})
    rows = pd.DataFrame(analysis.aggregate_groups(a - b, members))
    for _, part in rows.groupby("analysis"):
        assert part.loc[part.metric.eq("ap_contribution"), "value"].sum() == pytest.approx(
            a[:, 0].sum() - b[:, 0].sum()
        )
    group = rows.loc[rows.analysis.eq("L") & rows.group.eq("1")].set_index("metric").value
    assert group.ap_contribution == pytest.approx(
        group.positive_weight_share * group.mean_precision_change
    )
    empty = rows.loc[rows.analysis.eq("L") & rows.group.eq("2")].set_index("metric").value
    assert empty.ap_contribution == 0
    assert np.isnan(empty.background_mean_gain)


def test_depth_difference_pairs_same_fold_before_mean():
    rows = []
    for fold, one, two in [(0, 0.5, 0.1), (1, -0.2, 0.3)]:
        for group, value, count in [("1", one, 7), ("2", two, 4)]:
            rows.append(
                {
                    "city": "synthetic",
                    "outer_fold": fold,
                    "condition": "full_domain",
                    "contrast": "a-b",
                    "metric": "background_mean_gain",
                    "analysis": "L",
                    "group": group,
                    "value": value,
                    "positive_cells": count,
                }
            )
    result = analysis.paired_depth(pd.DataFrame(rows))
    np.testing.assert_allclose(result.value, [0.4, -0.5])
    assert result.positive_cells.tolist() == [4, 4]


def test_archived_evaluation_is_explicit(tmp_path):
    assert analysis.check_previous(tmp_path, pd.DataFrame(), pd.DataFrame(), {}) == {
        "reference_verified": False,
    }
    with pytest.raises(FileNotFoundError):
        analysis.check_previous(tmp_path, pd.DataFrame(), pd.DataFrame(), {}, Path("missing"))


def test_archived_evaluation_checks_metrics_and_nan_membership(tmp_path):
    methods = pd.DataFrame(
        {
            "city": ["toy"],
            "outer_fold": [0],
            "variant": ["SAGE-G2-2"],
            "full_background_ap": [0.3],
            "full_background_pb_auc": [0.6],
        }
    )
    old = methods.melt(id_vars=["city", "outer_fold", "variant"], var_name="metric")
    directory = tmp_path / "archive"
    directory.mkdir()
    old.to_csv(directory / "methods_folds.csv", index=False)
    folds = pd.DataFrame(
        {
            "city": ["toy", "toy"],
            "outer_fold": [0, 0],
            "condition": ["full_domain"] * 2,
            "contrast": ["SAGE-G2-2 - SAGE-G0-2"] * 2,
            "group": ["Zplus", "Zcut"],
            "metric": ["background_mean_gain"] * 2,
            "analysis": ["A"] * 2,
            "value": [0.1, np.nan],
        }
    )
    folds.to_csv(directory / "structure_folds.csv", index=False)
    hashes = {}
    qa = analysis.check_previous(tmp_path, methods, folds, hashes, Path("archive"))
    assert qa["reference_verified"]
    assert qa["global_max_error"] == qa["support_max_error"] == 0
    assert len(hashes) == 2
    folds.loc[1, "value"] = 0.0
    with pytest.raises(ValueError, match="support results differ"):
        analysis.check_previous(tmp_path, methods, folds, {}, Path("archive"))
    old.loc[0, "value"] = np.nan
    old.to_csv(directory / "methods_folds.csv", index=False)
    with pytest.raises(ValueError, match="global metrics differ"):
        analysis.check_previous(tmp_path, methods, folds, {}, Path("archive"))
