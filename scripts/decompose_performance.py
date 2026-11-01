"""Complete-domain L/K diagnostics on frozen, independently selected primary predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_structure import structure_inputs
from evaluate_predictions import positive_statistics, summarize_structure
from summarize_training import (
    CONTRASTS,
    check_identity,
    logical_run_path,
    metrics_and_signature,
    require,
    validate_selections,
    validated_seed_frames,
    verify_provenance,
)

from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.supplementary_mechanisms import GROUPS

PAIRS = tuple(pair for pair in CONTRASTS if pair[1] != "A*") + (("FF-G2-1", "SAGE-G0-2"),)
VARIANTS = {variant for pair in PAIRS for variant in pair}
METRICS = (
    "ap_contribution",
    "candidate_midrank_gain",
    "background_mean_gain",
    "background_median_gain",
    "background_improved_fraction",
    "background_unchanged_fraction",
    "positive_weight_share",
    "mean_precision_change",
)


def aggregate_groups(differences: np.ndarray, memberships: pd.DataFrame) -> list[dict]:
    """Account on complete-domain thresholds; no within-group reranking."""
    require(differences.shape == (len(memberships), 3), "positive diagnostics misaligned")
    rows = []
    for analysis, groups in {**GROUPS, "overall": ["all"]}.items():
        for group in groups:
            mask = (
                np.ones(len(memberships), bool)
                if analysis == "overall"
                else (memberships[analysis].to_numpy() == group)
            )
            selected = differences[mask]
            n, total = len(selected), len(differences)
            bg = selected[:, 2]
            values = (
                selected[:, 0].sum(),
                selected[:, 1].mean() if n else np.nan,
                bg.mean() if n else np.nan,
                np.median(bg) if n else np.nan,
                (bg > 0).mean() if n else np.nan,
                (bg == 0).mean() if n else np.nan,
                n / total,
                selected[:, 0].mean() * total if n else np.nan,
            )
            rows.extend(
                {
                    "analysis": analysis,
                    "group": group,
                    "positive_cells": n,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in zip(METRICS, values, strict=True)
            )
        contributions = [
            r["value"]
            for r in rows
            if r["analysis"] == analysis and r["metric"] == "ap_contribution"
        ]
        require(
            np.isclose(sum(contributions), differences[:, 0].sum(), atol=1e-12),
            "AP additivity failed",
        )
    return rows


def paired_depth(folds: pd.DataFrame) -> pd.DataFrame:
    keys = ["city", "outer_fold", "condition", "contrast", "metric"]
    part = folds.loc[
        folds.analysis.eq("L")
        & folds.group.isin(["1", "2"])
        & ~folds.metric.isin(["ap_contribution", "positive_weight_share"])
    ]
    wide = part.loc[part.group.eq("1")].merge(
        part.loc[part.group.eq("2")], on=keys, validate="1:1", suffixes=("_1", "_2")
    )
    return wide[keys].assign(
        analysis="L",
        group="L1-L2",
        value=wide.value_1 - wide.value_2,
        positive_cells=wide[["positive_cells_1", "positive_cells_2"]].min(axis=1),
    )


def summarize(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, omitted = [], []
    for analysis, part in folds.groupby("analysis"):
        summary, lofo = summarize_structure(part)
        summaries.append(summary.assign(analysis=analysis))
        omitted.append(lofo.assign(analysis=analysis))
    return pd.concat(summaries, ignore_index=True), pd.concat(omitted, ignore_index=True)


def check_previous(
    root: Path,
    methods: pd.DataFrame,
    folds: pd.DataFrame,
    hashes: dict,
    reference_summary: Path | None = None,
) -> dict:
    """Check overlapping metrics only when an archived evaluation is supplied."""
    if reference_summary is None:
        return {"reference_verified": False}
    old_path = root / reference_summary / "methods_folds.csv"
    old = pd.read_csv(old_path)
    check = methods.melt(id_vars=["city", "outer_fold", "variant"], var_name="metric").merge(
        old,
        on=["city", "outer_fold", "variant", "metric"],
        validate="1:1",
        suffixes=("_new", "_old"),
    )
    require(len(check) == len(methods) * 2, "missing global metric comparisons")
    global_error = float(abs(check.value_new - check.value_old).max())
    require(
        np.isfinite(check[["value_new", "value_old"]]).all().all() and global_error < 1e-12,
        "global metrics differ",
    )
    hashes[old_path.relative_to(root).as_posix()] = sha256_file(old_path)
    old_path = root / reference_summary / "structure_folds.csv"
    old = pd.read_csv(old_path)
    old = old.loc[old.condition.eq("full_domain")]
    check = old.merge(
        folds.loc[folds.analysis.isin(["A", "overall"])],
        on=["city", "outer_fold", "condition", "contrast", "group", "metric"],
        validate="1:1",
        suffixes=("_old", "_new"),
    )
    require(len(check) == len(old), "missing original support comparisons")
    support_error = float(abs(check.value_new - check.value_old).max())
    require(
        len(check) > 0
        and np.allclose(check.value_new, check.value_old, atol=1e-12, rtol=0, equal_nan=True),
        "existing support results differ",
    )
    hashes[old_path.relative_to(root).as_posix()] = sha256_file(old_path)
    return {
        "reference_verified": True,
        "global_max_error": global_error,
        "support_max_error": support_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path, default=Path("results/major_revision_diagnostics_20261012")
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-structure", type=Path, help="Archived node-label parquet")
    parser.add_argument(
        "--reference-summary", type=Path, help="Archived final evaluation directory"
    )
    args = parser.parse_args()
    root, hashes = args.root.resolve(), {}
    output = root / args.output
    require(not output.exists(), "output must be new")
    run = logical_run_path(root, root / args.run_root)
    config, grid = verify_provenance(root, run, hashes)
    _, outer, _ = validate_selections(root, run, config, grid, hashes)
    print("Verified complete primary-run provenance and selections.", flush=True)
    structure, _ = structure_inputs(root, hashes, args.reference_structure)
    lookup = structure.set_index(["city", "outer_fold", "grid_id"])
    diagnostics, memberships, signatures, method_records = {}, {}, {}, []
    for index, (task, path) in enumerate(outer):
        if task.realization_id not in VARIANTS:
            continue
        frame = pd.read_parquet(path)
        check_identity(frame, task)
        for seed, rows in validated_seed_frames(frame, False, task.seed):
            metrics, signature, positive, _ = metrics_and_signature(rows)
            unit = (task.city, int(task.outer_fold), task.trajectory)
            require(signatures.setdefault(unit, signature) == signature, "unpaired domain")
            full = rows.loc[rows.bank_id.eq("NA")].sort_values("grid_id")
            labels = full.grid_id.isin(positive).to_numpy(np.int8)
            stats = positive_statistics(labels, full.score.to_numpy())
            require(
                np.isclose(stats[:, 0].sum(), metrics["full_background_ap"], atol=1e-12),
                "full AP mismatch",
            )
            ids = full.loc[labels.astype(bool), "grid_id"].tolist()
            member = lookup.loc[[(unit[0], unit[1], node) for node in ids], list(GROUPS)]
            member = member.reset_index(drop=True)
            if unit[:2] in memberships:
                pd.testing.assert_frame_equal(member, memberships[unit[:2]])
            memberships[unit[:2]] = member
            diagnostics.setdefault((*unit, seed), {})[task.realization_id] = stats
            method_records.append(
                {
                    "city": unit[0],
                    "outer_fold": unit[1],
                    "variant": task.realization_id,
                    "full_background_ap": stats[:, 0].sum(),
                    "full_background_pb_auc": stats[:, 2].mean(),
                }
            )
        if index % 100 == 0:
            print(f"Read {index + 1}/{len(outer)} task positions.", flush=True)
    collected, measurement_records = {}, []
    for (city, fold, trajectory, seed), variants in diagnostics.items():
        require(set(variants) == VARIANTS, "incomplete neural matrix")
        for a, b in PAIRS:
            difference = variants[a] - variants[b]
            key = (city, fold, f"{a} - {b}")
            collected.setdefault(key, []).append(difference)
            for row in aggregate_groups(difference, memberships[city, fold]):
                if row["metric"] == "ap_contribution":
                    measurement_records.append(
                        {
                            "city": city,
                            "outer_fold": fold,
                            "trajectory": trajectory,
                            "seed": seed,
                            "contrast": key[2],
                            **row,
                        }
                    )
    records = []
    for (city, fold, contrast), repeats in collected.items():
        require(len(repeats) == 15, "expected five trajectories and three seeds")
        for row in aggregate_groups(np.mean(repeats, axis=0), memberships[city, fold]):
            records.append(
                {
                    "city": city,
                    "outer_fold": fold,
                    "contrast": contrast,
                    "condition": "full_domain",
                    **row,
                }
            )
    folds = pd.DataFrame(records)
    summary, lofo = summarize(folds)
    depth = paired_depth(folds)
    depth_summary, depth_lofo = summarize(depth)
    methods = (
        pd.DataFrame(method_records)
        .groupby(["city", "outer_fold", "variant"], as_index=False)
        .mean()
    )
    qa = check_previous(root, methods, folds, hashes, args.reference_summary)
    output.mkdir(parents=True)
    tables = {
        "group_folds.csv": folds,
        "group_summary.csv": summary,
        "group_lofo.csv": lofo,
        "l1_l2_folds.csv": depth,
        "l1_l2_summary.csv": depth_summary,
        "l1_l2_lofo.csv": depth_lofo,
        "ap_measurements.csv": pd.DataFrame(measurement_records),
        "methods_folds.csv": methods,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    manifest = {
        "status": "complete",
        "run": run.name,
        "contrasts": PAIRS,
        "paired_units": len(diagnostics),
        **qa,
        "source_hashes": hashes,
        "script_sha256": sha256_file(Path(__file__)),
        "outputs": {name: sha256_file(output / name) for name in tables},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Complete-domain structural diagnostics\n\n"
        "Nine pre-existing neural contrasts use all held-out eligible cells exactly once per fit. "
        "A/L/K definitions are unchanged; K is distinct incoming support within two hops. "
        "Positive threshold precision, candidate midpoint and background midpoint use the complete "
        "domain with half credit for ties. Per-positive differences average 15 paired fits before "
        "group medians/fractions. Group AP terms sum to each global difference. Mean precision "
        "change = contribution / positive weight share; empty groups contribute zero AP and have "
        "undefined rank/precision means. Counts refer to distinct positive cells.\n\n"
        "Summaries weight five folds equally. Common fold resampling (10,000 draws, seed 20261011) "
        "is descriptive; rank/precision ranges require five positives per fold. No significance "
        "or independent-replicate interpretation. L1-L2 is paired within each fold. Existing "
        "global AP/PB-AUC and complete-domain A support outputs are checked to 1e-12 "
        "when --reference-summary is supplied. "
        "Only aggregate outputs are released. No additional training or outer-score selection.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k.endswith("error") or k == "paired_units"}
        )
    )


if __name__ == "__main__":
    main()
