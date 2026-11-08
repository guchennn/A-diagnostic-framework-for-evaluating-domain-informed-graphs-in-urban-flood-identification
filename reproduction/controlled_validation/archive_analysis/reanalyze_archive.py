"""Read-only reanalysis of archived synthetic PU fits; no training or score reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from cache_helpers import EXPECTED_CONFIG, load_cache, positive_terms

PAIRS = [("useful", "own"), ("useful", "wrong"), ("wrong", "own")]
GROUPS = ["0", "1", "2"]


def reweight(terms: np.ndarray, weight: float) -> np.ndarray:
    """Exact tied-threshold AP terms under uniform background weight."""
    precision = terms[:, 0] * len(terms)
    assert np.all((precision > 0) & (precision <= 1 + 1e-12))
    return precision / (precision + weight * (1 - precision)) / len(terms)


def verify_formula() -> dict:
    """Independent score-threshold calculation, including positive/background ties."""
    cases = [
        ([1, 0, 1, 0, 0], [0.9, 0.8, 0.3, 0.2, 0.1]),
        ([1, 0, 1, 0, 1, 0], [0.8, 0.8, 0.5, 0.5, 0.5, 0.1]),
        ([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]),
    ]
    residuals = []
    for y, scores in cases:
        y, scores = np.asarray(y), np.asarray(scores)
        m, b = int(y.sum()), int((y == 0).sum())
        terms = positive_terms(y, scores)
        for weight in [0.1, 1.0, 10 * m / b]:
            direct = 0.0
            for threshold in np.unique(scores[y == 1]):
                at_threshold = int(((scores == threshold) & (y == 1)).sum())
                tp = int(((scores >= threshold) & (y == 1)).sum())
                fp = int(((scores >= threshold) & (y == 0)).sum())
                direct += (at_threshold / m) * tp / (tp + weight * fp)
            residuals.append(abs(reweight(terms, weight).sum() - direct))
    assert max(residuals) < 1e-12
    return {
        "toy_cases": len(cases),
        "weight_cases": len(residuals),
        "max_weighted_ap_error": max(residuals),
    }


def aggregate(frame: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    return (
        frame.groupby(keys, dropna=False)[value]
        .agg(
            mean_pp="mean",
            min_study_pp="min",
            max_study_pp="max",
            studies="count",
            positive_studies=lambda x: int((x > 0).sum()),
            negative_studies=lambda x: int((x < 0).sum()),
        )
        .reset_index()
    )


def main(root: Path, output: Path) -> None:
    ROOT, OUT = root.resolve(), output.resolve()
    BASE = ROOT / "results/synthetic_pu_regime_20261029"
    CONTROL = ROOT / "results/synthetic_pu_regime_20261029_controls"
    if not (BASE / "config.json").is_file():
        raise FileNotFoundError(
            "Original synthetic config/cache artifacts required under --project-root/results; see README."
        )
    CONFIG = json.loads((BASE / "config.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    checks = verify_formula()
    for key in [
        "grid_rows",
        "grid_columns",
        "test_rows",
        "trajectories",
        "graph_seeds",
        "own_seeds",
        "studies",
        "grids_per_study",
    ]:
        assert CONFIG[key] == EXPECTED_CONFIG[key], key
    assert CONFIG["upstream_coefficients"] == [0.0, 0.5, 1.0]
    test_n = (CONFIG["test_rows"][1] - CONFIG["test_rows"][0]) * CONFIG["grid_columns"]
    inputs = [
        Path(__file__).resolve(),
        BASE / "config.json",
        CONTROL / "config.json",
        Path(__file__).with_name("cache_helpers.py"),
        CONTROL / "ap_grids.csv",
        CONTROL / "location_summary.csv",
    ]
    rows, support_rows, count_rows, residuals = [], [], [], []
    for beta in CONFIG["upstream_coefficients"]:
        for study in range(CONFIG["studies"]):
            for grid in range(CONFIG["grids_per_study"]):
                result = load_cache(ROOT, beta, study, grid)
                inputs.append(
                    (ROOT / "results/synthetic_pu_regime_20261029_cache")
                    / f"{beta}_{study}_{grid}.pkl"
                )
                if beta == 0:
                    inputs.append(
                        (ROOT / "results/synthetic_pu_regime_20261029_controls_cache")
                        / f"{beta}_{study}_{grid}.pkl"
                    )
                m = result["positives"]
                b = test_n - m
                assert m > 0 and b > 0 and len(result["groups"]) == m
                assert set(result["groups"]) == set(GROUPS)
                assert set(result["support"]) == {"root", "retained", "cut"}
                names = ["own", "useful", "wrong"] + (["truncated"] if beta > 0 else [])
                count_rows.append(
                    {
                        "beta": beta,
                        "study": study,
                        "grid": grid,
                        "positives": m,
                        "background": b,
                        "weight_1to10": 10 * m / b,
                        "truncated_available": beta > 0,
                    }
                )
                for evaluation, weight in [("full", 1.0), ("1to10", 10 * m / b)]:
                    means = {}
                    for name in names:
                        fits = [
                            result["stats"][(name, t, s)]
                            for t in range(CONFIG["trajectories"])
                            for s in range(CONFIG["graph_seeds"])
                        ]
                        assert all(f.shape == (m, 2) for f in fits)
                        means[name] = np.mean(
                            [np.column_stack([reweight(f, weight), f[:, 1]]) for f in fits], axis=0
                        )
                    for a, ref in PAIRS + ([("truncated", "useful")] if beta > 0 else []):
                        diff = means[a] - means[ref]
                        context = {
                            "beta": beta,
                            "study": study,
                            "grid": grid,
                            "evaluation": evaluation,
                            "contrast": f"{a} - {ref}",
                        }
                        contributions = []
                        for group in [*GROUPS, "total"]:
                            mask = (
                                result["groups"] == group if group != "total" else np.ones(m, bool)
                            )
                            contribution = diff[mask, 0].sum() * 100
                            if group != "total":
                                contributions.append(contribution)
                            rows.append(
                                dict(
                                    **context,
                                    group=group,
                                    ap_contribution_pp=contribution,
                                    rank_change_pp=diff[mask, 1].mean() * 100,
                                )
                            )
                        residuals.append(abs(sum(contributions) - diff[:, 0].sum() * 100))
                    if beta > 0:
                        diff = means["truncated"] - means["useful"]
                        cut, retained = result["support"] == "cut", result["support"] == "retained"
                        for metric, column, scale in [
                            ("threshold_precision", 0, 100 * m),
                            ("background_rank", 1, 100),
                        ]:
                            cut_value = diff[cut, column].mean() * scale
                            retained_value = diff[retained, column].mean() * scale
                            support_rows.append(
                                {
                                    "beta": beta,
                                    "study": study,
                                    "grid": grid,
                                    "evaluation": evaluation,
                                    "metric": metric,
                                    "cut_pp": cut_value,
                                    "retained_pp": retained_value,
                                    "cut_minus_retained_pp": cut_value - retained_value,
                                }
                            )
    grids = pd.DataFrame(rows)
    keys = ["beta", "evaluation", "contrast", "group"]
    studies = (
        grids.groupby([*keys, "study"])[["ap_contribution_pp", "rank_change_pp"]]
        .mean()
        .reset_index()
    )
    totals = studies[studies.group == "total"]
    ap_summary = aggregate(totals, keys[:-1], "ap_contribution_pp")
    group_summary = aggregate(studies, keys, "ap_contribution_pp")
    rank_summary = aggregate(studies, keys, "rank_change_pp")
    interaction = studies.pivot(
        index=["beta", "contrast", "group", "study"],
        columns="evaluation",
        values="ap_contribution_pp",
    ).reset_index()
    interaction["weight_interaction_pp"] = interaction["1to10"] - interaction["full"]
    interaction["same_sign_across_weights"] = np.sign(interaction["1to10"]) == np.sign(
        interaction["full"]
    )
    interaction_summary = aggregate(
        interaction, ["beta", "contrast", "group"], "weight_interaction_pp"
    )
    weight_directions = (
        interaction.groupby(["beta", "contrast", "group"])
        .agg(studies=("study", "count"), same_sign_studies=("same_sign_across_weights", "sum"))
        .reset_index()
    )
    reference = studies.pivot(
        index=["beta", "evaluation", "group", "study"],
        columns="contrast",
        values="ap_contribution_pp",
    ).reset_index()
    reference["reference_effect_pp"] = reference["useful - own"] - reference["useful - wrong"]
    reference["identity_residual_pp"] = reference["reference_effect_pp"] - reference["wrong - own"]
    reference_summary = aggregate(reference, ["beta", "evaluation", "group"], "reference_effect_pp")
    direction = studies.copy()
    direction["ap_rank_same_sign"] = np.sign(direction.ap_contribution_pp) == np.sign(
        direction.rank_change_pp
    )
    direction_summary = (
        direction.groupby(keys)
        .agg(studies=("study", "count"), same_sign_studies=("ap_rank_same_sign", "sum"))
        .reset_index()
    )
    support = pd.DataFrame(support_rows)
    support_studies = (
        support.groupby(["beta", "evaluation", "metric", "study"])[
            ["cut_pp", "retained_pp", "cut_minus_retained_pp"]
        ]
        .mean()
        .reset_index()
    )
    support_summary = aggregate(
        support_studies, ["beta", "evaluation", "metric"], "cut_minus_retained_pp"
    )
    # Explicit missing beta=0 truncation; no fabricated zero effect or replication count.
    absent = pd.DataFrame(
        [
            {"beta": 0.0, "evaluation": e, "metric": m, "studies": 0, "availability": "not cached"}
            for e in ["full", "1to10"]
            for m in ["threshold_precision", "background_rank"]
        ]
    )
    support_summary["availability"] = "cached"
    support_summary = pd.concat([support_summary, absent], ignore_index=True)
    existing = pd.read_csv(CONTROL / "ap_grids.csv")
    reconstructed = grids[(grids.group == "total") & (grids.evaluation == "full")]
    comparison = existing.merge(
        reconstructed, on=["beta", "study", "grid", "contrast"], validate="one_to_one"
    )
    assert len(comparison) == len(existing) == 450
    checks["archived_ap_max_error_pp"] = float(
        np.max(np.abs(comparison.ap_gain * 100 - comparison.ap_contribution_pp))
    )
    checks["max_additivity_residual_pp"] = max(residuals)
    checks["max_reference_identity_residual_pp"] = float(reference.identity_residual_pp.abs().max())
    checks["cache_grids"] = len(count_rows)
    rank_pivot = studies.pivot(
        index=["beta", "contrast", "group", "study"], columns="evaluation", values="rank_change_pp"
    )
    checks["rank_change_across_weights_max_error_pp"] = float(
        (rank_pivot["1to10"] - rank_pivot["full"]).abs().max()
    )
    assert checks["rank_change_across_weights_max_error_pp"] == 0
    checks["paired_fits_per_model_per_grid"] = CONFIG["trajectories"] * CONFIG["graph_seeds"]
    assert (
        max(
            checks[k]
            for k in [
                "archived_ap_max_error_pp",
                "max_additivity_residual_pp",
                "max_reference_identity_residual_pp",
            ]
        )
        < 1e-10
    )
    tables = {
        "grid_contrasts": grids,
        "study_contrasts": studies,
        "ap_summary": ap_summary,
        "group_summary": group_summary,
        "rank_summary": rank_summary,
        "weight_interactions_study": interaction,
        "weight_interactions_summary": interaction_summary,
        "reference_interactions_study": reference,
        "reference_interactions_summary": reference_summary,
        "direction_agreement": direction_summary,
        "weight_direction_agreement": weight_directions,
        "support_grid": support,
        "support_study": support_studies,
        "support_summary": support_summary,
        "cache_counts": pd.DataFrame(count_rows),
    }
    for name, frame in tables.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    manifest = [
        {
            "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in inputs
    ]
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(ap_summary.to_string(index=False))
    print(support_summary.to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.project_root, args.output_dir)
