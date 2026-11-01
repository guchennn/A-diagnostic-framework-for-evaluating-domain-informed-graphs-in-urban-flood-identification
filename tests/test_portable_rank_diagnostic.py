"""Independent invariants for the releaseable, synthetic ranking example."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "portable_rank", Path(__file__).parents[1] / "scripts/diagnose_graph_ranking.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_constant_and_perfect_rank_diagnostics():
    result = MODULE.diagnose_bank([1, 1, 0, 0], [4, 3, 2, 1], [1, 1, 1, 1], ["a", "b", "", ""])
    assert result["ap_a"] == 1
    assert result["ap_b"] == 0.5
    assert result["mean_positive_midrank_change"] == 0.25
    assert result["additivity_residual"] == 0
    assert sum(r["ap_contribution_a_minus_b"] for r in result["groups"]) == 0.5


def test_tie_blocks_match_independent_threshold_definition():
    y = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.8, 0.8, 0.7, 0.4, 0.4, 0.1])
    precision, rank = MODULE.score_components(y, scores)
    expected_p = [y[scores >= s].mean() for s in scores]
    expected_r = [((scores < s).sum() + 0.5 * (scores == s).sum()) / len(y) for s in scores]
    np.testing.assert_allclose(precision, expected_p)
    np.testing.assert_allclose(rank, expected_r)


def test_group_allocation_is_not_group_only_ap():
    result = MODULE.diagnose_bank([1, 0, 1, 0], [4, 3, 2, 1], [1, 2, 3, 4], ["a", "", "b", ""])
    assert result["groups"][0]["ap_contribution_a_minus_b"] == pytest.approx(0.25)
    assert result["groups"][1]["ap_contribution_a_minus_b"] == pytest.approx(1 / 12)


def test_missing_positive_group_is_rejected():
    with pytest.raises(ValueError, match="exhaustive"):
        MODULE.diagnose_bank([1, 0], [2, 1], [1, 2], ["", ""])


@pytest.mark.parametrize("missing", [None, np.nan, np.float32("nan")])
def test_missing_group_is_checked_before_string_conversion(missing):
    with pytest.raises(ValueError, match="exhaustive"):
        MODULE.diagnose_bank([1, 0], [2, 1], [1, 2], [missing, "background"])


def test_duplicate_pairing_is_rejected(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("bank_id,item_id,label,group,score_a,score_b\nb,1,1,g,1,2\nb,1,0,,2,1\n")
    with pytest.raises(ValueError, match="unique"):
        MODULE.run_csv(source)


def test_opposite_signs_and_equal_ap_have_distinct_group_accounting():
    banks = MODULE.run_csv(Path(__file__).parents[1] / "examples/ranking_diagnostic/synthetic.csv")[
        "banks"
    ]
    first = banks["opposite_signs"]
    second = banks["same_ap_different_allocation"]
    for bank in (first, second):
        assert bank["ap_a"] == pytest.approx(41 / 56)
        assert bank["ap_b"] == pytest.approx(11 / 21)
        assert bank["ap_difference"] == pytest.approx(5 / 24)
    group = first["groups"][0]
    assert group["group"] == "A"
    assert group["ap_contribution_a_minus_b"] == pytest.approx(11 / 168)
    assert group["midrank_change_a_minus_b"] == pytest.approx(-3 / 16)
    assert second["groups"][0]["ap_contribution_a_minus_b"] == pytest.approx(5 / 24)
    assert second["groups"][0]["midrank_change_a_minus_b"] == pytest.approx(1 / 8)
