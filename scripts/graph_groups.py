"""Descriptive reviewer audit of frozen structure, background ranks, and label flow."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from floodgnn_ceus.io import sha256_file

BANK_KEYS = ["city", "budget", "trajectory", "outer_fold", "seed", "bank_id"]
NODE_KEYS = ["city", "budget", "outer_fold", "grid_id", "A", "L"]
METHODS = ["G0", "G2", "Fixed-filter"]
COMPARISONS = [("G2", "G0"), ("Fixed-filter", "G0"), ("G2", "Fixed-filter")]
GROUPS = {"A": ["Z0", "Zcut", "Zplus"], "L": ["0", "1", "2"]}
FEATURES = ["log1p_population_density", "road_length_density", "built_up_fraction"]
METRICS = ["mean_gain", "median_gain", "improved_fraction", "unchanged_fraction"]


def background_midrank(positive: np.ndarray, background: np.ndarray) -> np.ndarray:
    """P(background score < positive score) + half the exact-tie probability."""
    positive, background = np.asarray(positive, float), np.asarray(background, float)
    if positive.ndim != 1 or background.ndim != 1 or not len(background):
        raise ValueError("one-dimensional scores and a nonempty background are required")
    if not np.isfinite(positive).all() or not np.isfinite(background).all():
        raise ValueError("scores must be finite")
    ordered = np.sort(background)
    lower = np.searchsorted(ordered, positive, side="left")
    upper = np.searchsorted(ordered, positive, side="right")
    return (lower + upper) / (2 * len(ordered))


def positive_gains(paired: pd.DataFrame) -> pd.DataFrame:
    """Average banks, seeds, then trajectories for each unique recorded positive."""
    required = BANK_KEYS + ["grid_id", "A", "L", "sample_role"]
    if paired[required].isna().any().any() or paired.duplicated(BANK_KEYS + ["grid_id"]).any():
        raise ValueError("missing pairing metadata or duplicate bank node")
    if set(paired.sample_role) != {"positive", "background"}:
        raise ValueError("positive and background roles are required")
    if not np.isfinite(paired[[f"score_{m}" for m in METHODS]].to_numpy()).all():
        raise ValueError("all three paired methods must have finite scores")
    positives = paired.loc[paired.sample_role.eq("positive")]
    for column in ["outer_fold", "A", "L"]:
        if positives.groupby(["city", "grid_id"])[column].nunique().ne(1).any():
            raise ValueError("positive fold or structure membership differs")
    rows = []
    memberships = {}
    for key, bank in paired.groupby(BANK_KEYS, sort=True):
        pos = bank.loc[bank.sample_role.eq("positive")]
        membership_key = (key[0], key[1], key[3])
        members = frozenset(pos.grid_id)
        if not members or memberships.setdefault(membership_key, members) != members:
            raise ValueError("positive membership differs between paired runs or banks")
        bg = bank.loc[bank.sample_role.eq("background")]
        ranks = {
            method: background_midrank(pos[f"score_{method}"], bg[f"score_{method}"])
            for method in METHODS
        }
        result = pos[NODE_KEYS + ["trajectory", "seed", "bank_id"]].copy()
        for treatment, reference in COMPARISONS:
            result[f"{treatment}-{reference}"] = ranks[treatment] - ranks[reference]
        rows.append(result)
    result = pd.concat(rows, ignore_index=True)
    comparisons = [f"{a}-{b}" for a, b in COMPARISONS]
    for retained in [["trajectory", "seed"], ["trajectory"], []]:
        result = result.groupby(NODE_KEYS + retained, sort=True)[comparisons].mean().reset_index()
    return result


def fold_summaries(gains: pd.DataFrame, minimum: int) -> pd.DataFrame:
    rows = []
    for (city, budget), domain in gains.groupby(["city", "budget"]):
        for analysis, groups in GROUPS.items():
            for group, fold, (treatment, reference) in itertools.product(
                groups, range(5), COMPARISONS
            ):
                part = domain.loc[domain[analysis].eq(group) & domain.outer_fold.eq(fold)]
                values = part[f"{treatment}-{reference}"]
                rows.append(
                    {
                        "city": city,
                        "budget": budget,
                        "analysis": analysis,
                        "group": group,
                        "outer_fold": fold,
                        "comparison": f"{treatment}-{reference}",
                        "positive_cells": len(values),
                        "status": "complete" if len(values) >= minimum else "sparse",
                        "mean_gain": values.mean(),
                        "median_gain": values.median() if len(values) else np.nan,
                        "improved_fraction": (values > 0).mean() if len(values) else np.nan,
                        "unchanged_fraction": (values == 0).mean() if len(values) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def equal_fold_summary(folds: pd.DataFrame, minimum: int) -> pd.DataFrame:
    rows = []
    keys = ["city", "budget", "analysis", "group", "comparison"]
    for key, part in folds.groupby(keys, sort=True):
        if part.outer_fold.duplicated().any():
            raise ValueError("duplicate fold summary")
        part = part.set_index("outer_fold").reindex(range(5))
        complete = part[METRICS].notna().all().all()
        dense = part.positive_cells.ge(minimum).all()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "positive_cells": int(part.positive_cells.sum()),
                "nonempty_folds": int(part.mean_gain.notna().sum()),
                "minimum_positive_cells_per_fold": part.positive_cells.min(),
                "status": "missing_fold" if not complete else "complete" if dense else "sparse",
                **{f"equal_fold_{m}": part[m].mean() if complete else np.nan for m in METRICS},
            }
        )
    return pd.DataFrame(rows)


def paired_l1_l2(folds: pd.DataFrame, minimum: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join L1 and L2 by fold identity; never subtract independently sorted arrays."""
    keys = ["city", "budget", "comparison", "outer_fold"]
    left = folds.loc[folds.analysis.eq("L") & folds.group.eq("1")]
    right = folds.loc[folds.analysis.eq("L") & folds.group.eq("2")]
    joined = left.merge(right, on=keys, how="outer", suffixes=("_L1", "_L2"), validate="1:1")
    paired = joined[keys].copy()
    paired["positive_cells_L1"] = joined.positive_cells_L1
    paired["positive_cells_L2"] = joined.positive_cells_L2
    paired["positive_cells"] = joined[["positive_cells_L1", "positive_cells_L2"]].min(
        axis=1, skipna=False
    )
    for metric in METRICS:
        paired[metric] = joined[f"{metric}_L1"] - joined[f"{metric}_L2"]
    paired["analysis"], paired["group"] = "L", "L1-L2"
    paired["status"] = np.where(paired.positive_cells.ge(minimum), "complete", "sparse")
    summary = equal_fold_summary(paired, minimum).drop(columns="positive_cells")
    return paired.drop(columns="positive_cells"), summary


def covariate_summary(structure: pd.DataFrame, positives: pd.DataFrame) -> pd.DataFrame:
    """One record per domain cell; these associations do not identify reporting bias."""
    positive_ids = positives[["city", "grid_id"]].drop_duplicates().assign(recorded_positive=True)
    domain = structure.merge(positive_ids, on=["city", "grid_id"], how="left", validate="1:1")
    rows = []
    for city, part in domain.groupby("city"):
        for population in ["all_domain_cells", "recorded_STRICT_LOC_positive"]:
            selected = (
                part
                if population == "all_domain_cells"
                else part.loc[part.recorded_positive.eq(True)]
            )
            for analysis, groups in GROUPS.items():
                for group, feature in itertools.product(groups, FEATURES):
                    values = selected.loc[selected[analysis].eq(group), feature]
                    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
                    rows.append(
                        {
                            "city": city,
                            "population": population,
                            "analysis": analysis,
                            "group": group,
                            "feature": feature,
                            "n_cells": len(values),
                            "n_observed": int(values.notna().sum()),
                            "median": median,
                            "q1": q1,
                            "q3": q3,
                            "iqr": q3 - q1,
                        }
                    )
    return pd.DataFrame(rows)


def label_flow(report: dict) -> pd.DataFrame:
    rows = []
    for city in report["cities"]:
        summary = city["label_remap"]["summary"]
        counts = summary["strict_primary_reason_counts"]
        if sum(counts.values()) != summary["raw_row_count"]:
            raise ValueError("mutually exclusive primary reasons must conserve input rows")
        if counts["included"] != summary["label_cell_counts"]["STRICT-LOC"]:
            raise ValueError("included rows must equal STRICT-LOC cells")
        remaining = summary["raw_row_count"]
        for step, reason in enumerate(["outside_AOI", "not_all_type_1", "duplicate_grid"]):
            excluded = counts[reason]
            rows.append(
                {
                    "city": city["city_id"],
                    "step": step + 1,
                    "primary_reason": reason,
                    "input_rows": remaining,
                    "excluded_rows": excluded,
                    "retained_rows": remaining - excluded,
                }
            )
            remaining -= excluded
    return pd.DataFrame(rows)


def run(root: Path, output: Path) -> None:
    runtime = root / "artifacts/runtime/supp-mechanism-ab-20261007-r4"
    sources = [
        runtime / "paired_bank_rows.parquet",
        runtime / "test_node_structure.parquet",
        root / "artifacts/reports/dsm_final_aoi.json",
        root / "configs/supplementary_mechanisms_v1.yaml",
    ]
    paired, structure = [pd.read_parquet(p) for p in sources[:2]]
    config = yaml.safe_load(sources[3].read_text(encoding="utf-8"))
    minimum = config["rank_min_unique_positives_per_fold"]
    expected = set(
        itertools.product(
            config["cities"],
            [config["budget"]],
            config["trajectories"],
            config["outer_folds"],
            config["seeds"],
            config["bank_ids"],
        )
    )
    if set(paired[BANK_KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("saved predictions do not cover the frozen run/bank Cartesian product")
    if len(paired) * len(METHODS) != config["expected_rows"]:
        raise ValueError("saved paired-row count differs from the frozen source")
    if structure.duplicated(["city", "grid_id"]).any():
        raise ValueError("domain cells must be unique")
    for axis, groups in GROUPS.items():
        if not set(structure[axis]).issubset(groups):
            raise ValueError("unknown frozen structure group")
    verified = paired.merge(
        structure[["city", "grid_id", "outer_fold", "A", "L"]],
        on=["city", "grid_id", "outer_fold", "A", "L"],
        how="left",
        validate="m:1",
        indicator=True,
    )
    if not verified._merge.eq("both").all():
        raise ValueError("paired scores disagree with frozen structure")
    features = []
    for city in config["cities"]:
        path = root / f"data/interim/{city}/features_18.parquet"
        sources.append(path)
        features.append(pd.read_parquet(path, columns=["grid_id"] + FEATURES).assign(city=city))
    structure = structure.merge(
        pd.concat(features), on=["city", "grid_id"], how="left", validate="1:1", indicator=True
    )
    if not structure._merge.eq("both").all():
        raise ValueError("domain feature coverage is incomplete")
    gains = positive_gains(paired)
    folds = fold_summaries(gains, minimum)
    contrast_folds, contrasts = paired_l1_l2(folds, minimum)
    report = json.loads(sources[2].read_text(encoding="utf-8"))
    for city in report["cities"]:
        expected_n = city["label_remap"]["summary"]["label_cell_counts"]["STRICT-LOC"]
        if gains.loc[gains.city.eq(city["city_id"]), "grid_id"].nunique() != expected_n:
            raise ValueError("scored recorded positives do not match the STRICT-LOC count")
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "covariates.csv": covariate_summary(structure, gains),
        "rank_folds.csv": folds,
        "rank_summary.csv": equal_fold_summary(folds, minimum),
        "l1_l2_paired_folds.csv": contrast_folds,
        "l1_l2_summary.csv": contrasts,
        "label_flow.csv": label_flow(report),
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    readme = """# Reviewer structure and label audit

This is a descriptive reanalysis of frozen budget-100, FULL-COMMON, uniform-reveal,
STRICT-LOC predictions: 5 outer folds, 5 trajectories, 3 seeds and 3 evaluation banks.
No model is fitted or selected, and no threshold or group is chosen from outcomes.

For each positive and method, background midrank is
`(number of background scores below it + 0.5 * number tied) / background count`.
The reference is the entire background in the same evaluation bank, not its structural
subgroup; positive scores do not enter the reference distribution. Exact float ties are used.
A constant-score model has midrank 0.5, and perfect positive/background separation has 1.
Gains are treatment minus reference; Fixed-filter is FF. Banks, then seeds, then trajectories
are averaged for each unique positive before fold group summaries. Thus improvement fraction
and median describe each positive's average gain, not the fraction of stochastic runs improved.
No positive is treated as multiple independent observations. Summary metrics weight all five
outer folds equally, including the fold medians and fold improvement fractions; they are not
pooled-positive medians or fractions. Empty folds suppress the five-fold summary. The original
minimum of five unique positives per group per fold marks sparse groups; fold values remain
descriptive. No confidence intervals, p-values, or independence claims are produced.

L1-L2 contrasts subtract within the same city, budget, comparison and fold before five-fold
averaging. Their minimum count is the minimum group support across the paired folds. These
compare associations in fixed structural groups, not interventions on propagation depth.
L0 has no incoming test edge; L1 has incoming edges but no two-edge upstream path; L2 has a
two-edge upstream path. A groups distinguish natural zero indegree (Z0), pruning-induced
zero indegree (Zcut), and retained incoming support (Zplus).

Covariates report unique cells, medians, quartiles and IQR separately for all domain cells
and recorded STRICT-LOC positives. Population is the existing log1p_population_density feature;
road_length_density and built_up_fraction retain their original units and transformation.
These descriptive distributions can reveal covariate composition differences, but do not
estimate missing reports, prove unbiased ascertainment, or establish physical mechanisms.
The full domain includes recorded positives. The outputs contain aggregate records only.

Label flow uses mutually exclusive strict_primary_reason_counts in dsm_final_aoi.json.
The display order is outside AOI, not all type 1, then duplicate grid. Overlapping qc_flag_counts
are not used. The ledger is not redistributed. Source and output hashes are in manifest.json.

Run: `.venv/Scripts/python.exe scripts/graph_groups.py`
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    sources.append(Path(__file__).resolve())
    manifest = {
        "scope": "descriptive frozen-score reviewer reanalysis; no model selection",
        "unique_domain_cells": len(structure),
        "unique_positive_cells": len(gains),
        "minimum_positive_cells_per_fold": minimum,
        "sources": [
            {"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)} for p in sources
        ],
        "outputs": [
            {"path": name, "sha256": sha256_file(output / name)} for name in [*tables, "README.md"]
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "tables": {n: len(t) for n, t in tables.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/reviewer_structure_20261011"))
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
