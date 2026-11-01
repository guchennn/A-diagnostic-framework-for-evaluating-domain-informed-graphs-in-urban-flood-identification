"""New-run structural decomposition: exact AP additivity, ties and fold pairing."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("analyze_structure", SCRIPTS / "analyze_structure.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def bank():
    return pd.DataFrame(
        {
            "city": ["toy"] * 4,
            "budget": ["100"] * 4,
            "trajectory": ["1"] * 4,
            "outer_fold": [0] * 4,
            "seed": ["NEURAL-1"] * 4,
            "bank_id": [1] * 4,
            "grid_id": ["a", "b", "c", "d"],
            "sample_role": ["positive", "positive", "background", "background"],
            "A": ["Zplus"] * 4,
            "L": ["1", "2", "1", "2"],
            "K": ["1", "2", "1", "2"],
            "score_G0": [0.5] * 4,
            "score_G2": [3.0, 4.0, 1.0, 2.0],
            "score_Fixed-filter": [2.0, 2.0, 1.0, 1.0],
        }
    )


def test_midpoint_not_average_rank_and_exact_ap_additivity():
    paired = MODULE.midpoint_ranks(bank())
    assert paired.rank_G0.tolist() == [0.5] * 4
    assert paired.rank_G2.tolist() == [0.625, 0.875, 0.125, 0.375]
    cells, totals = MODULE.compute_cells(paired)
    for (analysis, comparison), group in cells.groupby(["analysis", "comparison"]):
        expected = totals.loc[totals.comparison.eq(comparison), "ap_difference"].item()
        assert group.ap_contribution.sum() == pytest.approx(expected, abs=1e-12)
    gain = cells.loc[
        cells.analysis.eq("A") & cells.group.eq("Zplus") & cells.comparison.eq("G2-G0"), "rank_gain"
    ].item()
    assert gain == 0.25


def test_tie_audit_constant_bank_and_largest_positive_gap():
    ties = MODULE.tie_audit(bank()).set_index("method")
    assert ties.loc["G0", "constant_bank"]
    assert ties.loc["G0", "maximum_tie_size"] == 4
    assert ties.loc["G0", "max_positive_upper_minus_midpoint"] == 0.5
    assert not ties.loc["G2", "constant_bank"]
    assert ties.loc["G2", "max_positive_upper_minus_midpoint"] == 0.125


def test_background_averages_repeated_positives_before_group_fraction():
    paired = pd.concat([bank(), bank().assign(bank_id=2)], ignore_index=True)
    structure = bank()[["city", "outer_fold", "grid_id", "K"]]
    folds = MODULE.background_folds(paired, structure)
    row = folds.loc[
        folds.analysis.eq("A")
        & folds.group.eq("Zplus")
        & folds.outer_fold.eq(0)
        & folds.comparison.eq("G2-G0")
        & folds.metric.eq("background_improved_fraction")
    ].iloc[0]
    assert row.positive_cells == 2
    assert row.value == 1


def test_pair_fold_contrast_and_sparse_interval_gate():
    rows = []
    for fold in range(5):
        for group in ["1", "2"]:
            rows.append(
                {
                    "city": "toy",
                    "analysis": "L",
                    "group": group,
                    "outer_fold": fold,
                    "comparison": "G2-G0",
                    "positive_cells": 6,
                    "metric": "fullbank_midrank_gain",
                    "value": fold / 10 if group == "1" else (4 - fold) / 20,
                }
            )
    frame = pd.DataFrame(rows).sample(frac=1, random_state=2)
    contrast = MODULE.paired_range_contrasts(frame)
    assert contrast.sort_values("outer_fold").value.tolist() == pytest.approx(
        [-0.2, -0.05, 0.1, 0.25, 0.4]
    )
    summary, lofo = MODULE.summarize_folds(contrast)
    assert summary["mean"].item() == pytest.approx(0.1)
    assert lofo.loc[lofo.omitted_fold.eq(0), "mean"].item() == pytest.approx(0.175)
    contrast.loc[contrast.outer_fold.eq(0), "positive_cells"] = 4
    summary, _ = MODULE.summarize_folds(contrast)
    assert summary.status.item() == "sparse"
    assert np.isnan(summary.resampling_q025.item())
    contrast.loc[contrast.outer_fold.eq(0), "value"] = np.nan
    summary, _ = MODULE.summarize_folds(contrast)
    assert summary.status.item() == "missing_fold"
    assert np.isnan(summary["mean"].item())


def test_empty_group_ap_is_zero_and_still_has_descriptive_interval():
    frame = pd.DataFrame(
        {
            "city": ["toy"] * 5,
            "analysis": ["A"] * 5,
            "group": ["Zcut"] * 5,
            "comparison": ["G2-G0"] * 5,
            "outer_fold": range(5),
            "positive_cells": [0] * 5,
            "metric": ["ap_contribution"] * 5,
            "value": [0.0] * 5,
        }
    )
    summary, _ = MODULE.summarize_folds(frame)
    assert summary.status.item() == "complete"
    assert summary.resampling_q025.item() == 0


@pytest.fixture
def portable_structure_root(tmp_path):
    from floodgnn_ceus.supplementary_mechanisms import cut_edges

    for city in ("zhengzhou", "beijing"):
        base = tmp_path / "data" / "interim" / city
        (base / "graphs").mkdir(parents=True)
        (base / "evaluation_design").mkdir()
        grid = pd.DataFrame({"grid_id": ["a", "b", "c"]})
        edges = pd.DataFrame(
            {
                "source_grid_id": ["a", "b"],
                "target_grid_id": ["b", "c"],
                "graph_id": ["G2", "G2"],
            }
        )
        parts = grid.assign(outer_fold=0, partition="test", block_row=0, block_col=0)
        grid.to_parquet(base / "final_scorable_grid.parquet")
        edges.to_parquet(base / "graphs/base_edges.parquet")
        parts.to_parquet(base / "evaluation_design/outer_partitions.parquet")
        cut_edges(edges, parts).assign(outer_fold=0).to_parquet(base / "graphs/outer_edges.parquet")
    return tmp_path


def test_recompute_structure_without_archive_and_verify_on_request(portable_structure_root):
    root = portable_structure_root
    hashes = {}
    structure, partitions = MODULE.structure_inputs(root, hashes)
    assert len(structure) == len(partitions) == 6
    assert structure.L.tolist() == ["0", "1", "2"] * 2
    assert len(hashes) == 8
    archive = root / "archive.parquet"
    structure.sample(frac=1).to_parquet(archive)
    rebuilt, _ = MODULE.structure_inputs(root, hashes, Path("archive.parquet"))
    pd.testing.assert_frame_equal(rebuilt, structure)
    assert "archive.parquet" in hashes
    structure.loc[0, "L"] = "2"
    structure.to_parquet(archive)
    with pytest.raises(AssertionError):
        MODULE.structure_inputs(root, {}, Path("archive.parquet"))
    with pytest.raises(FileNotFoundError):
        MODULE.structure_inputs(root, {}, Path("missing.parquet"))


@pytest.mark.parametrize(
    "module_name",
    [
        "analyze_structure",
        "evaluate_predictions",
        "decompose_performance",
    ],
)
@pytest.mark.parametrize("absolute", [False, True])
def test_cli_root_and_new_run_namespace_reach_fit_gate(
    tmp_path, monkeypatch, module_name, absolute
):
    import importlib

    module = importlib.import_module(module_name)
    root = tmp_path / "release"
    root.mkdir()
    run = Path("artifacts/runs/reviewer-revision-v1-abcdef123456")
    supplied = root / run if absolute else run
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", [module_name, "--root", str(root), "--run-root", str(supplied)]
    )

    def reject_unverified(actual_root, actual_run, hashes):
        assert actual_root == root
        assert actual_run == root / run
        raise ValueError("synthetic fit gate rejected")

    monkeypatch.setattr(module, "verify_provenance", reject_unverified)
    with pytest.raises(ValueError, match="synthetic fit gate rejected"):
        module.main()
    assert not (root / "results").exists()
