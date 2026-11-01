"""Frozen, descriptive support and propagation-range decomposition of existing scores."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from floodgnn_ceus.graph_support_audit import EDGE_KEYS, upstream_sizes
from floodgnn_ceus.graphs import cut_edges
from floodgnn_ceus.inference import weighted_auprc
from floodgnn_ceus.io import canonical_json_bytes, sha256_file
from floodgnn_ceus.rq3_novelty import _versions

KEYS = ["city", "budget", "trajectory", "outer_fold", "seed", "bank_id", "grid_id"]
CELL_KEYS = KEYS[:-1]
GROUPS = {"A": ["Z0", "Zcut", "Zplus"], "L": ["0", "1", "2"], "K": ["0", "1", "2", "3-4", "5+"]}
METHODS = ["G0", "G2", "Fixed-filter"]
COMPARISONS = [("G2", "G0"), ("Fixed-filter", "G0"), ("G2", "Fixed-filter")]


def ap_terms(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-observation AP terms using full-bank thresholds, including score ties."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ap = weighted_auprc(labels, scores, weights)
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
    positive = np.cumsum((weights * labels)[order])[ends]
    total = np.cumsum(weights[order])[ends]
    precision = np.divide(positive, total, out=np.zeros_like(positive), where=total > 0)
    ordered_precision = np.repeat(precision, np.diff(np.r_[-1, ends]))
    terms = np.empty(len(labels), dtype=float)
    terms[order] = ordered_precision * (weights * labels)[order] / positive[-1]
    if not np.isclose(terms.sum(), ap, atol=1e-12, rtol=0):
        raise ValueError("per-positive AP terms do not recover weighted_auprc")
    return terms


def pair_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate complete three-method pairing before computing full-bank ECDF ranks."""
    required = KEYS + ["method", "sample_role", "outer_block", "score"]
    if frame[required].isna().any().any() or not np.isfinite(frame["score"]).all():
        raise ValueError("missing pairing metadata or nonfinite score")
    if set(frame["method"]) != set(METHODS):
        raise ValueError("all three methods are required")
    if frame.duplicated(KEYS + ["method"]).any():
        raise ValueError("duplicate method rows")
    if set(frame["sample_role"]) != {"positive", "background"}:
        raise ValueError("unexpected sample roles")
    wide = frame.loc[frame["method"].eq("G0"), KEYS + ["sample_role", "outer_block"]]
    for method in METHODS:
        other = frame.loc[frame["method"].eq(method), required].drop(columns="method")
        joined = wide.merge(other, on=KEYS, how="outer", validate="1:1", indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError("method bank membership differs")
        for column in ("sample_role", "outer_block"):
            if not joined[f"{column}_x"].eq(joined[f"{column}_y"]).all():
                raise ValueError(f"method {column} differs")
        wide = joined.drop(columns=["_merge", "sample_role_y", "outer_block_y"]).rename(
            columns={
                "sample_role_x": "sample_role",
                "outer_block_x": "outer_block",
                "score": f"score_{method}",
            }
        )
    positives = wide.loc[wide["sample_role"].eq("positive")]
    if positives.groupby(["city", "grid_id"])["outer_fold"].nunique().ne(1).any():
        raise ValueError("a positive occurs in multiple outer-test folds")
    for method in METHODS:
        wide[f"rank_{method}"] = wide.groupby(CELL_KEYS)[f"score_{method}"].rank(
            method="max", pct=True
        )
    return wide


def graph_structure(
    nodes: list[str], edges: pd.DataFrame, partitions: pd.DataFrame, frozen: pd.DataFrame
) -> pd.DataFrame:
    """Derive support from frozen test graphs and verify their pruning provenance."""
    whole = upstream_sizes(nodes, edges).rename(
        columns={"upstream_1hop": "indegree_full", "upstream_2hop": "K_full"}
    )
    test = partitions.loc[partitions["partition"].eq("test")]
    if set(test["grid_id"]) != set(nodes) or test["grid_id"].duplicated().any():
        raise ValueError("every domain node must occur in exactly one test fold")
    rows = []
    for fold, partition in partitions.groupby("outer_fold"):
        cut = cut_edges(edges, partition)
        actual = frozen.loc[frozen["outer_fold"].eq(fold)]
        cols = EDGE_KEYS + ["partition"]
        pd.testing.assert_frame_equal(
            cut[cols].sort_values(cols).reset_index(drop=True),
            actual[cols].sort_values(cols).reset_index(drop=True),
        )
        local_edges = actual.loc[actual["partition"].eq("test")]
        ids = partition.loc[partition["partition"].eq("test"), "grid_id"].tolist()
        local = upstream_sizes(ids, local_edges).rename(
            columns={"upstream_1hop": "indegree_test", "upstream_2hop": "K_test"}
        )
        # A length-two directed path exists when an immediate predecessor has incoming edges.
        incoming = set(local_edges["target_grid_id"])
        depth_two = set(
            local_edges.loc[local_edges["source_grid_id"].isin(incoming), "target_grid_id"]
        )
        local["L"] = np.where(
            local["grid_id"].isin(depth_two), "2", np.where(local["indegree_test"].gt(0), "1", "0")
        )
        local = local.merge(whole, on="grid_id", validate="1:1")
        if (local["K_test"] > local["K_full"]).any():
            raise ValueError("pruning increased upstream support")
        local["A"] = np.where(
            local["indegree_full"].eq(0),
            "Z0",
            np.where(local["indegree_test"].eq(0), "Zcut", "Zplus"),
        )
        local["K"] = pd.cut(local["K_test"], [-1, 0, 1, 2, 4, np.inf], labels=GROUPS["K"]).astype(
            str
        )
        local["K_added"] = local["K_test"] - local["indegree_test"]
        local["q"] = 1 - local["K_test"] / local["K_full"].replace(0, np.nan)
        local["outer_fold"] = int(fold)
        rows.append(local)
    return pd.concat(rows, ignore_index=True)


def compute_cells(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every predeclared group, including zero-contribution empty groups."""
    rows, totals = [], []
    for key, frame in paired.groupby(CELL_KEYS, sort=True):
        metadata = dict(zip(CELL_KEYS, key, strict=True))
        y = frame["sample_role"].eq("positive").to_numpy(dtype=np.int8)
        weights = np.ones(len(frame))
        terms = {
            method: ap_terms(y, frame[f"score_{method}"].to_numpy(), weights) for method in METHODS
        }
        metrics = {
            method: weighted_auprc(y, frame[f"score_{method}"].to_numpy(), weights)
            for method in METHODS
        }
        for treatment, reference in COMPARISONS:
            comparison = f"{treatment}-{reference}"
            total = metrics[treatment] - metrics[reference]
            totals.append(
                {
                    **metadata,
                    "comparison": comparison,
                    "ap_difference": total,
                    "ap_treatment": metrics[treatment],
                    "ap_reference": metrics[reference],
                }
            )
            gain = frame[f"rank_{treatment}"].to_numpy() - frame[f"rank_{reference}"].to_numpy()
            delta = terms[treatment] - terms[reference]
            for analysis, groups in GROUPS.items():
                contributions = []
                for group in groups:
                    mask = frame[analysis].eq(group).to_numpy()
                    positive = mask & y.astype(bool)
                    contribution = float(delta[positive].sum())
                    contributions.append(contribution)
                    rows.append(
                        {
                            **metadata,
                            "analysis": analysis,
                            "group": group,
                            "comparison": comparison,
                            "positive_measurements": int(positive.sum()),
                            "evaluation_background_measurements": int(
                                (mask & ~y.astype(bool)).sum()
                            ),
                            "positive_weight_share": float(positive.sum() / y.sum()),
                            "rank_gain": float(gain[positive].mean()) if positive.any() else np.nan,
                            "ap_contribution": contribution,
                        }
                    )
                if not np.isclose(sum(contributions), total, atol=1e-12, rtol=0):
                    raise ValueError("group contributions do not sum to full-bank AP difference")
    return pd.DataFrame(rows), pd.DataFrame(totals)


def support_counts(paired: pd.DataFrame, structure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city, fold), domain in structure.groupby(["city", "outer_fold"]):
        bank = paired.loc[paired["city"].eq(city) & paired["outer_fold"].eq(fold)]
        positive_ids = set(bank.loc[bank["sample_role"].eq("positive"), "grid_id"])
        for analysis, groups in GROUPS.items():
            for group in groups:
                d = domain.loc[domain[analysis].eq(group)]
                b = bank.loc[bank[analysis].eq(group)]
                pos = b.loc[b["sample_role"].eq("positive")]
                bg = b.loc[b["sample_role"].eq("background")]
                row = {
                    "city": city,
                    "outer_fold": fold,
                    "analysis": analysis,
                    "group": group,
                    "positive_cells": pos["grid_id"].nunique(),
                    "paired_measurements": len(pos),
                    "evaluation_background_cells": bg["grid_id"].nunique(),
                    "evaluation_background_measurements": len(bg),
                    "domain_cells": len(d),
                    "domain_positive_cells": d["grid_id"].isin(positive_ids).sum(),
                    "domain_background_cells": (~d["grid_id"].isin(positive_ids)).sum(),
                }
                for name in ("K_full", "K_test", "K_added", "q"):
                    row[f"{name}_domain_mean"] = d[name].mean()
                    row[f"{name}_positive_mean"] = pos.drop_duplicates("grid_id")[name].mean()
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(
    cells: pd.DataFrame, totals: pd.DataFrame, counts: pd.DataFrame, protocol_id: str, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["city", "analysis", "group", "comparison"]
    folds = (
        cells.groupby(keys + ["outer_fold"], sort=False)
        .agg(
            rank_gain=("rank_gain", lambda x: x.mean() if x.notna().all() else np.nan),
            ap_contribution=("ap_contribution", "mean"),
            positive_weight_share=("positive_weight_share", "mean"),
            metric_cells=("ap_contribution", "size"),
        )
        .reset_index()
        .merge(counts, on=["city", "analysis", "group", "outer_fold"], validate="m:1")
    )
    overall_folds = (
        totals.groupby(["city", "comparison", "outer_fold"], sort=False)[
            ["ap_difference", "ap_treatment", "ap_reference"]
        ]
        .mean()
        .reset_index()
    )
    rows, loo, overall = [], [], []
    for city in config["cities"]:
        seed_name = config["bootstrap_seed_template"].format(
            effective_protocol_id=protocol_id, city=city
        )
        seed = int.from_bytes(hashlib.sha256(seed_name.encode()).digest()[:16], "big")
        draws = np.random.Generator(np.random.PCG64DXSM(seed)).integers(
            0, 5, (config["bootstrap_replicates"], 5)
        )
        for (analysis, group, comparison), part in folds.loc[folds["city"].eq(city)].groupby(
            keys[1:], sort=False
        ):
            part = part.set_index("outer_fold").reindex(range(5))
            if part["ap_contribution"].isna().any():
                raise ValueError("AP contributions must retain every outer fold")
            meta = dict(zip(keys, [city, analysis, group, comparison], strict=True))
            rank_valid = part["rank_gain"].notna().all()
            rank_dense = (
                part["positive_cells"].ge(config["rank_min_unique_positives_per_fold"]).all()
            )
            row = {
                **meta,
                "outer_fold_count": int(part["rank_gain"].notna().sum()),
                "positive_weight_share": part["positive_weight_share"].mean(),
                "rank_status": "missing_fold"
                if not rank_valid
                else "complete"
                if rank_dense
                else "sparse_no_interval",
            }
            for name in (
                "positive_cells",
                "paired_measurements",
                "evaluation_background_cells",
                "evaluation_background_measurements",
            ):
                row[name] = int(part[name].sum())
            for name, prefix in (("rank_gain", "rank"), ("ap_contribution", "ap")):
                values = part[name].to_numpy()
                valid = name == "ap_contribution" or rank_valid
                interval = name == "ap_contribution" or (rank_valid and rank_dense)
                row[name] = float(values.mean()) if valid else np.nan
                bounds = (
                    np.quantile(values[draws].mean(axis=1), config["interval_quantiles"])
                    if interval
                    else [np.nan, np.nan]
                )
                row[f"{prefix}_ci_lower"], row[f"{prefix}_ci_upper"] = bounds
            rows.append(row)
            for omitted in range(5):
                retained = part.drop(index=omitted)
                loo.append(
                    {
                        **meta,
                        "omitted_fold": omitted,
                        "ap_contribution": retained["ap_contribution"].mean(),
                        "rank_gain": retained["rank_gain"].mean() if rank_valid else np.nan,
                    }
                )
        for comparison, total in overall_folds.loc[overall_folds["city"].eq(city)].groupby(
            "comparison"
        ):
            total = total.set_index("outer_fold").reindex(range(5))
            values = total["ap_difference"].to_numpy()
            sampled = values[draws].mean(axis=1)
            lo, hi = np.quantile(sampled, config["interval_quantiles"])
            overall.append(
                {
                    "city": city,
                    "comparison": comparison,
                    "ap_difference": values.mean(),
                    "ap_ci_lower": lo,
                    "ap_ci_upper": hi,
                    "ap_treatment": total["ap_treatment"].mean(),
                    "ap_reference": total["ap_reference"].mean(),
                }
            )
            for analysis in GROUPS:
                part = folds.loc[
                    folds["city"].eq(city)
                    & folds["comparison"].eq(comparison)
                    & folds["analysis"].eq(analysis)
                ]
                sums = (
                    part.groupby("outer_fold")["ap_contribution"].sum().reindex(range(5)).to_numpy()
                )
                if not (
                    np.allclose(sums, values, rtol=0, atol=config["sum_tolerance"])
                    and np.allclose(
                        sums[draws].mean(axis=1), sampled, rtol=0, atol=config["sum_tolerance"]
                    )
                ):
                    raise ValueError("fold/bootstrap AP additivity failed")
    return pd.DataFrame(rows), folds, pd.DataFrame(loo), pd.DataFrame(overall)


def audit_matrix(frame: pd.DataFrame, config: dict) -> None:
    if len(frame) != config["expected_rows"]:
        raise ValueError("unexpected selected prediction row count")
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
    for method in METHODS:
        actual = frame.loc[frame["method"].eq(method), CELL_KEYS].drop_duplicates()
        if (
            set(actual.itertuples(index=False, name=None)) != expected
            or len(actual) != config["expected_cells_per_method"]
        ):
            raise ValueError("incomplete metric-cell matrix")


def audit_receipts(archive: Path, config: dict) -> dict:
    seen = set()
    with tarfile.open(archive) as stream:
        for member in stream:
            if not member.name.endswith("/status.json") or "task_receipts/" not in member.name:
                continue
            receipt = json.load(stream.extractfile(member))
            task = receipt.get("task", {})
            graph, model = task.get("graph_id"), task.get("model")
            method = (
                "Fixed-filter"
                if model == "Fixed-filter"
                else graph
                if model == "GraphSAGE-G2" and graph in {"G0", "G2"}
                else None
            )
            if (
                task.get("experiment_id") not in {"E1", "E2"}
                or method is None
                or task.get("budget") != config["budget"]
            ):
                continue
            if task.get("city") not in config["cities"]:
                continue
            for name in ("feature_set", "label_layer", "reveal_mechanism"):
                if task[name] != config[name]:
                    raise ValueError(f"source task {name} mismatch")
            if receipt["status"] != "success":
                raise ValueError("unsuccessful source task")
            key = (task["city"], int(task["outer_fold"]), task["trajectory"], task["seed"], method)
            if key in seen:
                raise ValueError("duplicate source task")
            seen.add(key)
    expected = set(
        itertools.product(
            config["cities"],
            config["outer_folds"],
            config["trajectories"],
            config["seeds"],
            METHODS,
        )
    )
    if seen != expected:
        raise ValueError(f"source receipts incomplete: {len(seen)} of {len(expected)}")
    return {"successful_source_tasks": len(seen), "scope_metadata_verified": True}


def run(root: Path, config_path: Path, output: Path) -> dict:
    root, output, config_path = root.resolve(), output.resolve(), config_path.resolve()
    if not output.is_relative_to(root / "artifacts/runtime"):
        raise ValueError("output must remain under ignored artifacts/runtime")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    config = yaml.safe_load(config_path.read_text("utf-8"))
    if (
        config["groups"] != GROUPS
        or config["methods"] != METHODS
        or config["comparisons"] != [list(x) for x in COMPARISONS]
    ):
        raise ValueError("frozen groups and comparisons changed")
    if (
        any(config[name] for name in ("optional_H", "optional_FULL", "intervention_C"))
        or config["point_weight"] != 1
    ):
        raise ValueError("only unit-weight A/B analysis is implemented")
    paths = {
        name: root / config[name]
        for name in (
            "prediction_source",
            "prediction_reference",
            "reference_manifest",
            "source_archive",
        )
    }
    paths.update(config=config_path, protocol_lock=root / "artifacts/protocol_lock.json")
    output.mkdir(parents=True, exist_ok=True)
    # Record the frozen configuration before inspecting new effects.
    (output / "analysis_config.yaml").write_bytes(config_path.read_bytes())
    protocol_id = json.loads(paths["protocol_lock"].read_text())["effective_protocol_id"]
    filters = [("budget", "=", config["budget"]), ("method", "in", METHODS)]
    frame = pd.read_parquet(paths["prediction_source"], filters=filters)
    reference = pd.read_parquet(paths["prediction_reference"], filters=filters)
    order = KEYS + ["method"]
    pd.testing.assert_frame_equal(
        frame.sort_values(order).reset_index(drop=True),
        reference.sort_values(order).reset_index(drop=True),
        check_exact=True,
    )
    del reference
    reference_manifest = json.loads(paths["reference_manifest"].read_text())
    if reference_manifest["output_sha256"] != sha256_file(paths["prediction_reference"]):
        raise ValueError("reference prediction manifest checksum mismatch")
    audit_matrix(frame, config)
    audit = audit_receipts(paths["source_archive"], config)
    paired = pair_predictions(frame)
    audit["prediction_rows"] = len(frame)
    audit["reference_predictions_identical"] = True
    audit["metric_cells_per_method"] = config["expected_cells_per_method"]
    audit["methods"] = METHODS
    structures = []
    for city in config["cities"]:
        city_paths = {
            name: root / "data/interim" / city / rel
            for name, rel in {
                "grid": "final_scorable_grid.parquet",
                "base": "graphs/base_edges.parquet",
                "outer": "graphs/outer_edges.parquet",
                "partition": "evaluation_design/outer_partitions.parquet",
                "banks": "evaluation_design/evaluation_banks.parquet",
            }.items()
        }
        paths.update({f"{city}_{name}": path for name, path in city_paths.items()})
        grid = pd.read_parquet(city_paths["grid"], columns=["grid_id"])
        partitions = pd.read_parquet(city_paths["partition"])
        structure = graph_structure(
            grid["grid_id"].tolist(),
            pd.read_parquet(city_paths["base"], filters=[("graph_id", "=", "G2")]),
            partitions,
            pd.read_parquet(city_paths["outer"], filters=[("graph_id", "=", "G2")]),
        )
        structure["city"] = city
        structures.append(structure)
        banks = pd.read_parquet(
            city_paths["banks"],
            filters=[
                ("scope", "=", "outer"),
                ("label_layer", "=", config["label_layer"]),
            ],
        )
        # Outer banks are shared across budgets, reveals and neural seeds.
        banks = banks.merge(pd.DataFrame({"trajectory": config["trajectories"]}), how="cross")
        check_keys = ["outer_fold", "trajectory", "bank_id", "grid_id", "sample_role"]
        observed = paired.loc[paired["city"].eq(city), check_keys].drop_duplicates()
        pd.testing.assert_frame_equal(
            banks[check_keys].sort_values(check_keys).reset_index(drop=True),
            observed.sort_values(check_keys).reset_index(drop=True),
            check_dtype=False,
        )
        part = partitions.loc[
            partitions["partition"].eq("test"), ["grid_id", "outer_fold", "block_row", "block_col"]
        ].copy()
        part["expected_block"] = part["block_row"].astype(str) + ":" + part["block_col"].astype(str)
        actual = paired.loc[paired["city"].eq(city)].merge(
            part, on=["grid_id", "outer_fold"], how="left", validate="m:1"
        )
        if not actual["outer_block"].eq(actual["expected_block"]).all():
            raise ValueError("prediction outer_block differs from test partition")
    structure = pd.concat(structures, ignore_index=True)
    paired = paired.merge(
        structure, on=["city", "outer_fold", "grid_id"], how="left", validate="m:1"
    )
    if paired[list(GROUPS)].isna().any().any():
        raise ValueError("missing test-graph structure")
    (output / "input_audit.json").write_bytes(canonical_json_bytes(audit))
    print(f"Input audit passed: {len(frame):,} prediction rows", flush=True)
    cells, totals = compute_cells(paired)
    counts = support_counts(paired, structure)
    summary, folds, loo, overall = summarize(cells, totals, counts, protocol_id, config)
    aggregate = output / "aggregates"
    aggregate.mkdir()
    tables = {
        "support_contribution.csv": summary.loc[summary["analysis"].eq("A")],
        "range_increment.csv": summary.loc[summary["analysis"].isin(["L", "K"])],
        "fold_effects.csv": folds,
        "leave_one_fold_out.csv": loo,
        "support_counts.csv": counts,
        "overall_effects.csv": overall,
    }
    for name, table in tables.items():
        table.to_csv(aggregate / name, index=False)
    paired.to_parquet(output / "paired_bank_rows.parquet", index=False)
    structure.to_parquet(output / "test_node_structure.parquet", index=False)
    cells.to_parquet(output / "cell_effects.parquet", index=False)
    totals.to_parquet(output / "cell_totals.parquet", index=False)
    implementation = [
        Path(__file__),
        root / "scripts/analyze_supplementary_mechanisms.py",
        root / "src/floodgnn_ceus/inference.py",
        root / "src/floodgnn_ceus/graphs.py",
        root / "src/floodgnn_ceus/graph_support_audit.py",
    ]
    manifest = {
        "status": "success",
        "analysis_id": config["analysis_id"],
        "interpretation": "post-hoc descriptive decomposition and conditional association; no intervention",
        "effective_protocol_id": protocol_id,
        "environment": _versions(),
        "input_audit": audit,
        "weighting": "unit record weight; equal 45 cells within fold, equal five folds",
        "bootstrap": {
            "replicates": config["bootstrap_replicates"],
            "generator": "PCG64DXSM",
            "seed_template": config["bootstrap_seed_template"],
            "common_draws_within_city": True,
        },
        "validation": {
            "ap_additivity_cell_fold_bootstrap": True,
            "tolerance": config["sum_tolerance"],
            "frozen_graph_reconstruction": True,
            "frozen_bank_membership": True,
        },
        "inputs": {
            name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "implementation": {
            str(path.relative_to(root)): sha256_file(path) for path in implementation
        },
        "aggregate_outputs": {
            name: {"rows": len(table), "sha256": sha256_file(aggregate / name)}
            for name, table in tables.items()
        },
    }
    (output / "provenance.json").write_bytes(canonical_json_bytes(manifest))
    (output / "validation_report.md").write_text(
        "# A/B validation\n\nInput pairing, source receipts, frozen banks and actual test graphs passed.\n"
        "All predeclared groups retained. AP additivity checked per bank, fold and all 10,000 common fold draws (atol=1e-12).\n"
        "Unit evaluation weights; 45 metric cells per fold; five equally weighted folds.\n"
        "Intervals are descriptive spatial-fold percentile intervals. Sparse/missing rank groups retain explicit status.\n"
        "A/B only: no training, H, FULL sensitivity, model selection or causal intervention.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/supplementary_mechanisms_v1.yaml")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.config, args.output_root)
    print(json.dumps({"status": result["status"], "output": str(args.output_root)}))
    return 0
