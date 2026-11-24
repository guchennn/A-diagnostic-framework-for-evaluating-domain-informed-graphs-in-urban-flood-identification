"""Recompute the published aggregates without city records or prediction files."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["city", "predictor", "feature_set", "budget"]
METRICS = ["own", "adjacency", "d8", "d8_minus_own", "d8_minus_adjacency"]
FILES = ["unit_metrics.csv", "fold_values.csv", "summary.csv", "table6.csv"]
TOLERANCE = 5e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual, expected, context: str) -> None:
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, f"shape mismatch: {context}")
    require(np.isfinite(actual).all(), f"non-finite values: {context}")
    require(
        np.allclose(actual, expected, atol=TOLERANCE, rtol=0),
        f"numeric mismatch: {context}",
    )


def indexed(frame: pd.DataFrame, keys: list[str], expected: set, name: str):
    require(not frame.duplicated(keys).any(), f"duplicate keys: {name}")
    actual = set(frame[keys].itertuples(index=False, name=None))
    require(actual == expected, f"incomplete or unexpected matrix: {name}")
    return frame.set_index(keys).sort_index()


def verify(directory: Path) -> dict:
    receipt = json.loads((directory / "verification.json").read_text("utf-8"))
    require(receipt["status"] == "complete", "experiment is not complete")
    frames = {}
    for name in FILES:
        path = directory / name
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == receipt["output_sha256"][name],
            f"hash mismatch: {name}",
        )
        frames[name] = pd.read_csv(
            path, dtype={"budget": str, "trajectory": str, "seed": str}, keep_default_na=False
        )
    conditions = set(
        product(
            ("beijing", "zhengzhou"),
            ("FF", "SAGE"),
            ("F18", "F16-noD8", "F14-noNeigh"),
            ("100", "FULL"),
        )
    )
    unit_keys = [*KEYS, "outer_fold", "trajectory", "seed"]
    expected_units = {
        (*condition, fold, draw, seed)
        for condition in conditions
        for fold, draw, seed in product(
            range(5),
            ("1", "2", "3", "4", "5") if condition[3] == "100" else ("NA",),
            ("NEURAL-1", "NEURAL-2", "NEURAL-3"),
        )
    }
    units = indexed(frames[FILES[0]], unit_keys, expected_units, "paired units")
    close(units[METRICS], units[METRICS], "finite unit metrics")
    for reference in ("own", "adjacency"):
        close(units[f"d8_minus_{reference}"], units.d8 - units[reference], reference)
    require(
        ((units[["own", "adjacency", "d8"]] >= 0) & (units[["own", "adjacency", "d8"]] <= 1))
        .all()
        .all(),
        "unit AP must be in [0, 1]",
    )
    fold_keys = [*KEYS, "outer_fold"]
    expected_folds = {(*condition, fold) for condition in conditions for fold in range(5)}
    folds = indexed(frames[FILES[1]], fold_keys, expected_folds, "fold values")
    recomputed = units.reset_index().groupby(fold_keys)[METRICS].mean().sort_index()
    close(folds[METRICS], recomputed[METRICS], "within-fold means")
    summary = indexed(
        frames[FILES[2]],
        [*KEYS, "quantity"],
        {(*condition, metric) for condition in conditions for metric in METRICS},
        "summary",
    )
    table = indexed(frames[FILES[3]], KEYS, conditions, "Table 6")
    draws = np.random.default_rng(20261011).integers(0, 5, (10000, 5))
    for condition in sorted(conditions):
        for metric in METRICS:
            values = recomputed.loc[condition, metric].to_numpy(float)
            low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
            positive, negative = int((values > 0).sum()), int((values < 0).sum())
            row = summary.loc[(*condition, metric)]
            close(
                row[["mean", "range_low", "range_high"]],
                [values.mean(), low, high],
                f"summary {condition} {metric}",
            )
            close(row[[f"fold{i}" for i in range(5)]], values, "summary fold values")
            require(
                row.positive_folds == positive and row.negative_folds == negative,
                f"fold sign counts: {condition} {metric}",
            )
            published = table.loc[condition]
            if metric in ("own", "adjacency", "d8"):
                close(published[f"{metric}_ap_percent"], 100 * values.mean(), "Table 6 AP")
            else:
                close(
                    published[[f"{metric}_{suffix}" for suffix in ("pp", "low_pp", "high_pp")]],
                    100 * np.array([values.mean(), low, high]),
                    "Table 6 contrast",
                )
                require(
                    published[f"{metric}_positive_folds"] == positive, "Table 6 positive fold count"
                )
    return {
        "status": "passed",
        "conditions": len(conditions),
        "paired_units": len(units),
        "fold_rows": len(folds),
        "summary_rows": len(summary),
        "tolerance": TOLERANCE,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().directory)))
