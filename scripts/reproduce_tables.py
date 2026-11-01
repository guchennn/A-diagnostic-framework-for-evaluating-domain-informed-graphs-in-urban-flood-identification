"""Recompute the main and ablation tables from released fit/fold aggregates."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    "sampled_pb_ap",
    "full_background_ap",
    "full_weighted_1to10_ap",
    "recall_top_10pct",
]


def check_values(actual, expected, keys, columns):
    """Check values by explicit keys, including missing or duplicated rows."""
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"Duplicate rows for {keys}")
    left = actual.set_index(keys)[columns].sort_index()
    right = expected.set_index(keys)[columns].sort_index()
    if not left.index.equals(right.index):
        raise ValueError(f"Row keys differ for {keys}")
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-12)


def reproduce(source, output):
    units = pd.read_csv(source / "main_units.csv")
    unit_keys = ["city", "outer_fold", "trajectory", "seed", "variant"]
    if units.duplicated(unit_keys).any():
        raise ValueError("Duplicate fitted-model rows")
    sizes = units.groupby(["city", "outer_fold", "variant"]).size()
    if not sizes.eq(15).all():
        raise ValueError("Expected five revelation trajectories and three refit slots per fold")
    trajectory = units.groupby(["city", "outer_fold", "variant", "trajectory"])[METRICS].mean()
    folds = trajectory.groupby(["city", "outer_fold", "variant"]).mean().reset_index()
    check_values(
        folds, pd.read_csv(source / "main_folds.csv"), ["city", "outer_fold", "variant"], METRICS
    )
    main = folds.groupby(["city", "variant"])[METRICS].mean().reset_index()
    expected = pd.read_csv(source / "main_absolute.csv")
    check_values(main, expected, ["city", "variant"], METRICS)
    main = main.merge(expected[["city", "variant", "label"]], on=["city", "variant"])

    ablation_folds = pd.read_csv(source / "ablation_absolute_folds.csv")
    keys = ["city", "budget", "method", "feature_set"]
    if ablation_folds.duplicated(keys + ["outer_fold"]).any():
        raise ValueError("Duplicate ablation fold rows")
    if not ablation_folds.groupby(keys).size().eq(5).all():
        raise ValueError("Expected five spatial folds per ablation setting")
    ablation = ablation_folds.groupby(keys)["sampled_pb_ap"].mean().reset_index()
    check_values(
        ablation, pd.read_csv(source / "ablation_absolute_summary.csv"), keys, ["sampled_pb_ap"]
    )
    pivot = ablation.pivot(
        index=["city", "budget", "feature_set"], columns="method", values="sampled_pb_ap"
    )
    table3 = pivot[["G0", "G2"]].rename(columns={"G0": "own_AP", "G2": "D8_AP"})
    table3["D8_minus_own"] = table3.D8_AP - table3.own_AP

    output.mkdir(parents=True, exist_ok=True)
    main.to_csv(output / "table2_absolute.csv", index=False)
    folds.to_csv(output / "main_fold_metrics.csv", index=False)
    table3.reset_index().to_csv(output / "table3_ablation.csv", index=False)
    ablation.to_csv(output / "ablation_all_models.csv", index=False)
    print("Verified all main fit-to-fold means and ablation fold-to-city means.")
    print(f"Wrote tables to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/absolute_performance")
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced/tables")
    args = parser.parse_args()
    reproduce(args.source, args.output)


if __name__ == "__main__":
    main()
