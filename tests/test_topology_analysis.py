"""Protocol-critical pairing and complete-only topology analysis checks."""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_topology_controls as analysis


@pytest.fixture
def paired():
    config = {
        "cities": ["toy"],
        "outer_folds": list(range(5)),
        "trajectories": ["1", "2"],
        "outer_seeds": ["NEURAL-1", "NEURAL-2"],
        "graph_realizations": [1, 2, 3],
        "variants": [{"model": "Fixed-filter"}],
    }
    rows = []
    for fold, draw, seed in product(range(5), ["1", "2"], config["outer_seeds"]):
        value = 0.4 + fold * 0.01 + int(draw) * 0.02 + config["outer_seeds"].index(seed) * 0.03
        base = {
            "city": "toy",
            "outer_fold": fold,
            "draw": draw,
            "seed": seed,
            "predictor": "Fixed-filter",
        }
        rows.append(
            {
                **base,
                "representation": "D8",
                "realization": 0,
                **dict.fromkeys(analysis.METRICS, value),
            }
        )
        for realization in (1, 2, 3):
            rows.append(
                {
                    **base,
                    "representation": "G7",
                    "realization": realization,
                    **dict.fromkeys(analysis.METRICS, value - realization * 0.01),
                }
            )
    return config, pd.DataFrame(rows)


def test_pair_before_equal_fold_aggregation(paired):
    config, frame = paired
    tables = analysis.aggregate_metrics(frame.sample(frac=1, random_state=7), config)
    summary = tables["paired_summary"]
    assert summary.n_folds.eq(5).all()
    assert summary["mean"].to_numpy() == pytest.approx([0.02] * 4)
    assert summary.mean_pp.to_numpy() == pytest.approx([2.0] * 4)
    assert summary.range_low.to_numpy() == pytest.approx([0.02] * 4)
    assert summary.range_high.to_numpy() == pytest.approx([0.02] * 4)
    assert len(tables["paired_folds"]) == 5 * 4
    for realization, part in tables["graph_realization_contrasts"].groupby("realization"):
        assert part["mean"].to_numpy() == pytest.approx([realization * 0.01] * 4)
        assert part.n_folds.eq(5).all()
    absolute = tables["absolute_summary"].set_index(["representation", "metric"])
    assert absolute.loc[("D8", "full_ap"), "mean"] == pytest.approx(0.465)
    assert absolute.loc[("G7", "full_ap"), "mean"] == pytest.approx(0.445)


@pytest.mark.parametrize("representation", ["D8", "G7"])
def test_missing_paired_fit_is_rejected(paired, representation):
    config, frame = paired
    frame = frame.drop(frame.index[frame.representation.eq(representation)][0])
    with pytest.raises(ValueError, match="missing or unexpected fits"):
        analysis.aggregate_metrics(frame, config)


def test_duplicate_and_unexpected_fit_rejected(paired):
    config, frame = paired
    with pytest.raises(ValueError, match="duplicate fit identity"):
        analysis.aggregate_metrics(pd.concat([frame, frame.iloc[[0]]]), config)
    frame.loc[0, "seed"] = "UNPAIRED"
    with pytest.raises(ValueError, match="missing or unexpected fits"):
        analysis.aggregate_metrics(frame, config)


def test_nonfinite_metrics_cannot_be_averaged_away(paired):
    config, frame = paired
    frame.loc[0, "full_ap"] = float("nan")
    with pytest.raises(ValueError, match="non-finite fit metric"):
        analysis.aggregate_metrics(frame, config)


def test_existing_metrics_preserve_ties_and_weighting():
    frame = pd.DataFrame(
        [
            {
                "bank_id": bank,
                "grid_id": str(i),
                "sample_role": "positive" if i < 2 else "unlabelled",
                "score": 0.5,
            }
            for bank, i in product(["NA", "1", "2", "3"], range(4))
        ]
    )
    values = analysis.evaluate(frame, {"0", "1"})
    assert values["full_ap"] == pytest.approx(0.5)
    assert values["sampled_ap"] == pytest.approx(0.5)
    assert values["weighted_ap"] == pytest.approx(1 / 11)
    assert values["recall_10"] == pytest.approx(0.1)


def test_incomplete_receipts_do_not_compute_or_publish_metrics(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/topology_controls_v1.yaml").write_text("run_id: toy", encoding="utf-8")
    output = tmp_path / "results"
    monkeypatch.setattr(analysis, "verify_provenance", lambda *_: (tmp_path, {}, []))

    def incomplete(*_):
        raise ValueError("missing receipt")

    def forbidden(*_):
        pytest.fail("metrics must not run before completion")

    monkeypatch.setattr(analysis, "verify_reference", lambda *_: {})
    monkeypatch.setattr(analysis, "collect_receipts", incomplete)
    monkeypatch.setattr(analysis, "metrics_from_records", forbidden)
    with pytest.raises(ValueError, match="missing receipt"):
        analysis.analyze(tmp_path, output, tmp_path / "reference")
    assert not output.exists()


def test_evaluation_uses_strict_label_bank_only(tmp_path):
    from types import SimpleNamespace

    directory = tmp_path / "data/interim/toy/evaluation_design"
    directory.mkdir(parents=True)
    pd.DataFrame(
        {"outer_fold": [0] * 4, "partition": ["test"] * 4, "grid_id": list("abcd")}
    ).to_parquet(directory / "outer_partitions.parquet")
    banks = pd.DataFrame(
        [
            {
                "scope": "outer",
                "outer_fold": 0,
                "label_layer": layer,
                "bank_id": bank,
                "grid_id": grid_id,
                "sample_role": "positive" if grid_id in positives else "background",
            }
            for layer, positives in [("STRICT-LOC", "ab"), ("ALL-LOC", "abc")]
            for bank, grid_id in product([1, 2, 3], "abcd")
        ]
    )
    banks.to_parquet(directory / "evaluation_banks.parquet")
    frame = pd.DataFrame(
        [
            {
                "bank_id": bank,
                "grid_id": grid_id,
                "sample_role": "positive" if grid_id in "ab" else "background",
                "score": 0.5,
                "status": "success",
            }
            for bank, grid_id in product(["NA", "1", "2", "3"], "abcd")
        ]
    )
    payload = tmp_path / "oof.parquet"
    frame.to_parquet(payload)
    record = ({"city": "toy", "outer_fold": 0}, SimpleNamespace(as_dict=dict), payload)
    result = analysis.metrics_from_records(tmp_path, [record])
    assert result.full_ap.iloc[0] == pytest.approx(0.5)
    frame.loc[frame.bank_id.eq("1"), "score"] = 0.6
    frame.to_parquet(payload)
    with pytest.raises(ValueError, match="bank scores differ"):
        analysis.metrics_from_records(tmp_path, [record])


def test_g6_aggregates_are_separate_and_provenance_labeled(tmp_path):
    directory = tmp_path / "results/minor_revision_final_20261029"
    directory.mkdir(parents=True)
    rows = [
        {"city": city, "series": "main", "contrast": contrast, "metric": metric, "mean": 0.01}
        for city, contrast, metric in product(
            ["zhengzhou", "beijing"],
            ["FF-G2-1 - FF-G6-1", "SAGE-G2-2 - SAGE-G6-2"],
            analysis.METRICS,
        )
    ]
    pd.DataFrame(rows).to_csv(directory / "contrasts_summary.csv", index=False)
    tables, hashes = analysis.existing_g6_tables(tmp_path)
    assert set(tables) == {"existing_G6_comparisons_summary"}
    result = tables["existing_G6_comparisons_summary"]
    assert result["mean"].eq(0.01).all()
    assert result.source_sha256.nunique() == 1
    assert "not refit or pooled" in result.provenance.iloc[0]
    assert len(hashes) == 1


def test_reference_run_outside_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="inside root"):
        analysis.verify_reference(tmp_path, tmp_path.parent / "other")


def test_public_metrics_match_direct_threshold_definition_with_ties():
    rng = np.random.default_rng(731)
    for _ in range(10):
        ids = [f"cell{i:03}" for i in range(50)]
        positive = set(ids[:10])
        score = np.round(rng.random(50), 1)
        y = np.arange(50) < 10
        full = pd.DataFrame(
            {
                "grid_id": ids,
                "score": score,
                "sample_role": np.where(y, "positive", "background"),
                "bank_id": "NA",
            }
        )
        frame = pd.concat([full.assign(bank_id=b) for b in ("NA", "1", "2", "3")])
        result = analysis.evaluate(frame, positive)
        for key, weight in (("full_ap", 1), ("sampled_ap", 1), ("weighted_ap", 2.5)):
            expected = np.mean(
                [
                    sum(y & (score >= t))
                    / (sum(y & (score >= t)) + weight * sum(~y & (score >= t)))
                    for t in score[y]
                ]
            )
            assert result[key] == pytest.approx(expected, abs=1e-14)
