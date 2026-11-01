"""Recompute paired, unit-weight ranking diagnostics from a common evaluation bank.

Input CSV columns: bank_id,item_id,label,group,score_a,score_b.
Group is required for positives; background rows may use an empty group.
No training, location data, or private CEUS files are needed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def score_components(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return threshold precision and midpoint ECDF, treating exact ties together."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positive_counts = np.bincount(inverse, weights=labels, minlength=len(counts))
    above = np.cumsum(counts[::-1])[::-1]
    positive_above = np.cumsum(positive_counts[::-1])[::-1]
    precision = (positive_above / above)[inverse]
    midpoint = ((np.cumsum(counts) - 0.5 * counts) / len(scores))[inverse]
    return precision, midpoint


def diagnose_bank(labels, scores_a, scores_b, groups) -> dict:
    """Account for AP on the complete bank; do not recompute AP within groups."""
    labels = np.asarray(labels)
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    raw_groups = np.asarray(groups, dtype=object)
    groups = raw_groups.astype(str)
    if any(v.ndim != 1 for v in (labels, a, b, groups)):
        raise ValueError("bank arrays must be one-dimensional")
    if not len(labels) or any(len(v) != len(labels) for v in (a, b, groups)):
        raise ValueError("bank arrays must have the same nonzero length")
    if not np.isin(labels, [0, 1]).all() or set(labels) != {0, 1}:
        raise ValueError("bank must contain binary recorded positives and background")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("all scores must be finite")
    positive = labels == 1
    if any(
        value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        for value in raw_groups[positive]
    ):
        raise ValueError("every positive needs one exhaustive structural group")
    if any(not g.strip() for g in groups[positive]):
        raise ValueError("every positive needs one exhaustive structural group")
    pa, ra = score_components(labels, a)
    pb, rb = score_components(labels, b)
    ap_a, ap_b = float(pa[positive].mean()), float(pb[positive].mean())
    rows = []
    for group in sorted(set(groups[positive])):
        member = positive & (groups == group)
        rows.append(
            {
                "group": group,
                "positive_count": int(member.sum()),
                "positive_share": float(member.sum() / positive.sum()),
                "ap_contribution_a_minus_b": float(
                    (pa[member] - pb[member]).sum() / positive.sum()
                ),
                "midrank_change_a_minus_b": float((ra[member] - rb[member]).mean()),
            }
        )
    residual = sum(row["ap_contribution_a_minus_b"] for row in rows) - (ap_a - ap_b)
    if abs(residual) > 1e-12:
        raise ArithmeticError("group AP contributions fail additivity")
    return {
        "bank_size": len(labels),
        "positive_count": int(positive.sum()),
        "ap_a": ap_a,
        "ap_b": ap_b,
        "ap_difference": ap_a - ap_b,
        "mean_positive_midrank_change": float((ra[positive] - rb[positive]).mean()),
        "additivity_residual": residual,
        "groups": rows,
    }


def run_csv(source: Path) -> dict:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"bank_id", "item_id", "label", "group", "score_a", "score_b"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        banks: dict[str, list[dict]] = {}
        seen = set()
        for row in reader:
            key = (row["bank_id"], row["item_id"])
            if not all(key) or key in seen:
                raise ValueError("bank_id/item_id must be nonempty and unique within banks")
            seen.add(key)
            if "weight" in row and float(row["weight"]) != 1:
                raise ValueError("this minimal example supports unit evaluation weights only")
            banks.setdefault(key[0], []).append(row)
    if not banks:
        raise ValueError("CSV contains no evaluation banks")
    return {
        "convention": "unit weights; exact ties; whole-bank AP; midpoint ECDF",
        "scope": "per-bank descriptive results; no independent-replicate inference",
        "banks": {
            key: diagnose_bank(
                [int(row["label"]) for row in rows],
                [float(row["score_a"]) for row in rows],
                [float(row["score_b"]) for row in rows],
                [row["group"] for row in rows],
            )
            for key, rows in sorted(banks.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
