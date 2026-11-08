"""Paired full-domain local increments under own and adjacency references."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[3]
ROOT = REPOSITORY
OUT = REPOSITORY / "reproduced/final_audit/local_references"
RUN_ROOT = None
VERIFY_HISTORICAL = False
sys.path.insert(0, str(REPOSITORY / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "group_migration"))
from analyze_structure import structure_inputs
from audit_migration import load_city
from evaluate_predictions import positive_statistics, summarize_structure
from summarize_training import (
    check_identity,
    logical_run_path,
    metrics_and_signature,
    validate_selections,
    validated_seed_frames,
    verify_provenance,
)

from floodgnn_ceus.io import sha256_file

VARIANTS = {"FF-G0-1": "own", "FF-G1-1": "adjacency", "FF-G2-1": "D8"}
PAIRS = [("D8", "own"), ("D8", "adjacency"), ("adjacency", "own")]


def run() -> None:
    hashes = {}
    run_path = logical_run_path(ROOT, RUN_ROOT)
    config, grid = verify_provenance(ROOT, run_path, hashes)
    _, outer, _ = validate_selections(ROOT, run_path, config, grid, hashes)
    print("Verified frozen provenance and inner-only selections.", flush=True)
    structure, _ = structure_inputs(ROOT, hashes)
    lookup = structure.set_index(["city", "outer_fold", "grid_id"])
    full_frames = []
    for city in ("zhengzhou", "beijing"):
        frame, _, paths = load_city(ROOT, city, VERIFY_HISTORICAL)
        full_frames.append(frame.assign(city=city))
        for path in paths:
            hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    full_lookup = pd.concat(full_frames).set_index(["city", "outer_fold", "grid_id"])
    diagnostics, members, full_members, signatures = {}, {}, {}, {}
    for task, path in outer:
        if task.realization_id not in VARIANTS:
            continue
        frame = pd.read_parquet(path)
        check_identity(frame, task)
        for seed, rows in validated_seed_frames(frame, False, task.seed):
            metric, signature, positive, _ = metrics_and_signature(rows)
            unit = (task.city, int(task.outer_fold), task.trajectory)
            assert signatures.setdefault(unit, signature) == signature
            full = rows.loc[rows.bank_id.eq("NA")].sort_values("grid_id")
            labels = full.grid_id.isin(positive).to_numpy(np.int8)
            values = positive_statistics(labels, full.score.to_numpy())
            assert abs(values[:, 0].sum() - metric["full_background_ap"]) < 1e-12
            ids = full.loc[labels.astype(bool), "grid_id"].tolist()
            groups = lookup.loc[[(unit[0], unit[1], i) for i in ids], "L"].to_numpy(str)
            key = unit[:2]
            if key in members:
                np.testing.assert_array_equal(members[key], groups)
            members[key] = groups
            full_members[key] = full_lookup.loc[
                [(unit[0], unit[1], i) for i in ids], "full_group"
            ].to_numpy(str)
            diagnostics.setdefault((*unit, seed), {})[VARIANTS[task.realization_id]] = values
    assert len(diagnostics) == 150
    collected, fit_rows = {}, []
    max_identity = max_additivity = 0.0
    for (city, fold, draw, seed), vals in diagnostics.items():
        assert set(vals) == set(VARIANTS.values())
        identity = (
            (vals["D8"] - vals["own"])
            - (vals["D8"] - vals["adjacency"])
            - (vals["adjacency"] - vals["own"])
        )
        max_identity = max(max_identity, float(abs(identity).max()))
        groups = members[city, fold]
        for a, b in PAIRS:
            diff = vals[a] - vals[b]
            collected.setdefault((city, fold, f"{a} - {b}"), []).append(diff)
            total = 0.0
            for group in ("0", "1", "2"):
                part = diff[groups == group]
                contribution = part[:, 0].sum()
                total += contribution
                fit_rows.append(
                    {
                        "city": city,
                        "outer_fold": fold,
                        "trajectory": draw,
                        "seed": seed,
                        "contrast": f"{a} - {b}",
                        "group": group,
                        "positive_cells": len(part),
                        "ap_contribution": contribution,
                    }
                )
            max_additivity = max(max_additivity, abs(total - diff[:, 0].sum()))
    folds, cell_counts, averaged = [], [], {}
    for (city, fold, contrast), repeats in collected.items():
        assert len(repeats) == 15
        diff = np.mean(repeats, axis=0)
        averaged[city, fold, contrast] = diff
        groups = members[city, fold]
        for group in ("0", "1", "2", "all"):
            part = diff if group == "all" else diff[groups == group]
            n = len(part)
            metrics = {
                "ap_contribution": part[:, 0].sum(),
                "mean_precision_change": part[:, 0].mean() * len(groups) if n else np.nan,
                "background_mean_gain": part[:, 2].mean() if n else np.nan,
            }
            for metric, value in metrics.items():
                folds.append(
                    {
                        "city": city,
                        "outer_fold": fold,
                        "condition": "full_domain",
                        "contrast": contrast,
                        "group": group,
                        "metric": metric,
                        "positive_cells": n,
                        "value": value,
                    }
                )
            for index, metric in [(0, "precision"), (2, "rank")]:
                cell_counts.append(
                    {
                        "city": city,
                        "outer_fold": fold,
                        "contrast": contrast,
                        "group": group,
                        "metric": metric,
                        "positive_cells": n,
                        "improved": int((part[:, index] > 0).sum()),
                        "worsened": int((part[:, index] < 0).sum()),
                        "unchanged": int((part[:, index] == 0).sum()),
                    }
                )
    fold_df = pd.DataFrame(folds)
    summary, lofo = summarize_structure(fold_df)
    counts = (
        fold_df.groupby(["city", "contrast", "group", "metric"])
        .positive_cells.sum()
        .rename("positive_cells_total")
    )
    summary = summary.merge(counts, on=["city", "contrast", "group", "metric"], validate="1:1")
    for col in ("mean", "resampling_q025", "resampling_q975"):
        summary[col + "_pp"] = summary[col] * 100
    manuscript_error = None
    if VERIFY_HISTORICAL:
        old = pd.read_csv(
            Path(__file__).with_name("published_own_summary.csv"),
            dtype={"group": str},
        )
        check = summary.loc[summary.contrast.eq("D8 - own")].merge(
            old, on=["city", "group", "metric"], suffixes=("_new", "_old"), validate="1:1"
        )
        assert len(check) == len(old)
        manuscript_error = float(abs(check.mean_pp_new - check.mean_pp_old).max())
        assert manuscript_error < 1e-10
    agreement = []
    for (city, fold), groups in members.items():
        own, adj = averaged[city, fold, "D8 - own"], averaged[city, fold, "D8 - adjacency"]
        for group in ("0", "1", "2", "all"):
            mask = np.ones(len(groups), bool) if group == "all" else groups == group
            for index, metric in [(0, "precision"), (2, "rank")]:
                a, b = own[mask, index], adj[mask, index]
                agreement.append(
                    {
                        "city": city,
                        "outer_fold": fold,
                        "group": group,
                        "metric": metric,
                        "positive_cells": int(mask.sum()),
                        "both_positive": int(((a > 0) & (b > 0)).sum()),
                        "both_negative": int(((a < 0) & (b < 0)).sum()),
                        "own_positive_adj_negative": int(((a > 0) & (b < 0)).sum()),
                        "own_negative_adj_positive": int(((a < 0) & (b > 0)).sum()),
                        "any_zero": int(((a == 0) | (b == 0)).sum()),
                    }
                )
    pd.DataFrame(fit_rows).to_csv(OUT / "fit_contributions.csv", index=False)
    sensitivity_rows = []
    for (city, fold, contrast), diff in averaged.items():
        groups = full_members[city, fold]
        for group in ("0", "1", "2", "all"):
            part = diff if group == "all" else diff[groups == group]
            n = len(part)
            values = {
                "ap_contribution": part[:, 0].sum(),
                "mean_precision_change": part[:, 0].mean() * len(groups) if n else np.nan,
                "background_mean_gain": part[:, 2].mean() if n else np.nan,
            }
            for metric, value in values.items():
                sensitivity_rows.append(
                    {
                        "city": city,
                        "outer_fold": fold,
                        "condition": "full_domain",
                        "contrast": contrast,
                        "group": group,
                        "metric": metric,
                        "positive_cells": n,
                        "value": value,
                    }
                )
        assert (
            abs(sum(diff[groups == g, 0].sum() for g in ("0", "1", "2")) - diff[:, 0].sum()) < 1e-12
        )
    sensitivity_folds = pd.DataFrame(sensitivity_rows)
    sensitivity_summary, _ = summarize_structure(sensitivity_folds)
    for col in ("mean", "resampling_q025", "resampling_q975"):
        sensitivity_summary[col + "_pp"] = sensitivity_summary[col] * 100
    sensitivity_folds.to_csv(OUT / "full_graph_group_folds.csv", index=False)
    sensitivity_summary.to_csv(OUT / "full_graph_group_summary.csv", index=False)
    sensitivity_comparison = sensitivity_summary.merge(
        summary,
        on=["city", "condition", "contrast", "group", "metric"],
        suffixes=("_full_group", "_heldout_group"),
        validate="1:1",
    )
    sensitivity_comparison["allocation_difference_pp"] = (
        sensitivity_comparison.mean_pp_full_group - sensitivity_comparison.mean_pp_heldout_group
    )
    sensitivity_comparison.to_csv(OUT / "group_definition_sensitivity.csv", index=False)
    fold_df.to_csv(OUT / "fold_contrasts.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    lofo.to_csv(OUT / "leave_one_fold_out.csv", index=False)
    pd.DataFrame(cell_counts).to_csv(OUT / "cell_direction_by_fold.csv", index=False)
    pd.DataFrame(cell_counts).groupby(["city", "contrast", "group", "metric"], as_index=False).sum(
        numeric_only=True
    ).drop(columns="outer_fold").to_csv(OUT / "cell_direction_counts.csv", index=False)
    pd.DataFrame(agreement).groupby(["city", "group", "metric"], as_index=False).sum(
        numeric_only=True
    ).drop(columns="outer_fold").to_csv(OUT / "reference_direction_agreement.csv", index=False)
    verification = {
        "paired_units": len(diagnostics),
        "models_per_unit": 3,
        "fits_per_fold": 15,
        "manuscript_mean_max_error_pp": manuscript_error,
        "max_reference_identity_residual": max_identity,
        "max_additivity_residual": max_additivity,
        "bootstrap_seed": 20261011,
        "bootstrap_resamples": 10000,
    }
    assert max_identity < 1e-12 and max_additivity < 1e-12
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    for module in (
        "evaluate_predictions.py",
        "analyze_structure.py",
        "summarize_training.py",
    ):
        hashes[f"scripts/{module}"] = sha256_file(REPOSITORY / "scripts" / module)
    for module in ("supplementary_mechanisms.py", "io.py"):
        hashes[f"src/floodgnn_ceus/{module}"] = sha256_file(
            REPOSITORY / "src/floodgnn_ceus" / module
        )
    hashes[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    migration_helper = Path(__file__).resolve().parent.parent / "group_migration/audit_migration.py"
    hashes[str(migration_helper)] = sha256_file(migration_helper)
    (OUT / "input_manifest.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(
        summary[
            [
                "city",
                "contrast",
                "group",
                "metric",
                "mean_pp",
                "resampling_q025_pp",
                "resampling_q975_pp",
                "positive_folds",
            ]
        ].to_string(index=False)
    )
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Frozen native run: artifacts/runs/reviewer-revision-v1-*",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--verify-historical-baseline", action="store_true")
    args = parser.parse_args()
    VERIFY_HISTORICAL = args.verify_historical_baseline
    ROOT = args.project_root.resolve()
    RUN_ROOT = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    required = [RUN_ROOT / "plan.json", RUN_ROOT / "local_environment_lock.json"]
    if VERIFY_HISTORICAL:
        required.append(
            ROOT / "artifacts/runtime/supp-mechanism-ab-20261007-r4/test_node_structure.parquet"
        )
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error(
            "Original local inputs are required; aggregates cannot reconstruct predictions. "
            "See reproduction/final_audit/README.md. Missing: " + "; ".join(missing)
        )
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    run()
