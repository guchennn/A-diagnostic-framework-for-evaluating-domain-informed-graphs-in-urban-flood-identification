"""Read-only design, training-record, observed-label mixing and DSM-fill audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from analyze_reviewer_training_structure import structure_inputs
from summarize_reviewer_training import logical_run_path, validate_selections, verify_provenance

from floodgnn_ceus.evaluation_design import materialize_partition
from floodgnn_ceus.io import sha256_file

NEURAL = {"MLP", "Fixed-filter", "GraphSAGE-G2"}


def implied_stop_epoch(best_epoch: np.ndarray, patience: int = 30, cap: int = 250) -> np.ndarray:
    """Algorithmic reconstruction for successful inner fits, not a logged runtime counter."""
    best = np.asarray(best_epoch)
    if ((best < 1) | (best > cap)).any():
        raise ValueError("best epochs must lie inside the frozen inner-training range")
    return np.minimum(best + patience, cap)


def design_counts(root: Path, city: str, hashes: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = {
        name: root / f"data/interim/{city}/{relative}"
        for name, relative in {
            "grid": "final_scorable_grid.parquet",
            "labels": "final_label_ledger.parquet",
            "outer": "evaluation_design/outer_partitions.parquet",
            "inner": "evaluation_design/inner_blocks.parquet",
            "reveals": "evaluation_design/reveals.parquet",
            "u": "evaluation_design/u_samples.parquet",
            "banks": "evaluation_design/evaluation_banks.parquet",
        }.items()
    }
    data = {}
    for name, path in paths.items():
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
        data[name] = pd.read_parquet(path)
    labels = set(data["labels"].loc[data["labels"].label_layer.eq("STRICT-LOC"), "grid_id"])
    outer_rows, inner_rows = [], []
    reveals = data["reveals"].loc[
        data["reveals"].label_layer.eq("STRICT-LOC") & data["reveals"].budget.isin(["100", "FULL"])
    ]
    inner = data["inner"].loc[data["inner"].budget.isin(["100", "FULL"])]
    for fold, outer in data["outer"].groupby("outer_fold"):
        train_ids = set(outer.loc[outer.partition.eq("train"), "grid_id"])
        for partition, part in outer.groupby("partition"):
            ids = set(part.grid_id)
            available = len(ids & labels)
            outer_rows.append(
                {
                    "city": city,
                    "outer_fold": fold,
                    "partition": partition,
                    "nodes": len(ids),
                    "recorded_positives": available,
                    "budget100_revealed_positives": 100 if partition == "train" else np.nan,
                    "budget100_fraction_of_available": 100 / available
                    if partition == "train"
                    else np.nan,
                    "full_revealed_positives": available if partition == "train" else np.nan,
                }
            )
        outer_grid = data["grid"].loc[data["grid"].grid_id.isin(train_ids)]
        for (budget, trajectory), blocks in inner.loc[inner.outer_fold.eq(fold)].groupby(
            ["budget", "trajectory_id"]
        ):
            revealed = set(
                reveals.loc[
                    reveals.outer_fold.eq(fold)
                    & reveals.budget.eq(budget)
                    & reveals.trajectory_id.eq(trajectory),
                    "grid_id",
                ]
            )
            if not revealed.issubset(train_ids) or (
                len(revealed) != (100 if budget == "100" else len(train_ids & labels))
            ):
                raise ValueError("outer revealed-positive count/scope differs")
            assignment = blocks.rename(columns={"inner_fold": "fold_id"})
            for inner_fold in range(3):
                partition = materialize_partition(
                    outer_grid, assignment, held_out_fold=inner_fold, buffer_m=1000
                )
                partitions = {role: set(p.grid_id) for role, p in partition.groupby("partition")}
                u = data["u"].loc[
                    data["u"].scope.eq("inner")
                    & data["u"].outer_fold.eq(fold)
                    & data["u"].budget.eq(budget)
                    & data["u"].trajectory_id.eq(trajectory)
                    & data["u"].inner_fold.eq(inner_fold)
                ]
                if len(u) != 2000 or not set(u.grid_id).issubset(partitions["train"]):
                    raise ValueError("inner U sample scope/count differs")
                banks = data["banks"].loc[
                    data["banks"].scope.eq("inner")
                    & data["banks"].outer_fold.eq(fold)
                    & data["banks"].budget.eq(budget)
                    & data["banks"].trajectory_id.eq(trajectory)
                    & data["banks"].inner_fold.eq(inner_fold)
                ]
                validation_positive = partitions["test"] & revealed
                for _, bank in banks.groupby("bank_id"):
                    actual = set(bank.loc[bank.sample_role.eq("positive"), "grid_id"])
                    if actual != validation_positive or len(bank) != 11 * len(actual):
                        raise ValueError("validation positive membership differs")
                row = {
                    "city": city,
                    "outer_fold": fold,
                    "budget": budget,
                    "trajectory": trajectory,
                    "inner_fold": inner_fold,
                    "outer_train_recorded_available": len(train_ids & labels),
                    "outer_revealed_positives": len(revealed),
                    "u_sample_size": len(u),
                    "u_unrevealed_recorded_positives": len(set(u.grid_id) & labels),
                    "validation_positive_cells": len(validation_positive),
                    "validation_background_per_bank": 10 * len(validation_positive),
                }
                for role in ["train", "buffer", "test"]:
                    ids = partitions.get(role, set())
                    row[f"{role}_nodes"] = len(ids)
                    row[f"{role}_recorded_available"] = len(ids & labels)
                    row[f"{role}_revealed_positive"] = len(ids & revealed)
                row["train_revealed_fraction_of_recorded_available"] = (
                    row["train_revealed_positive"] / row["train_recorded_available"]
                )
                inner_rows.append(row)
    data["positive_ids"] = labels
    return pd.DataFrame(outer_rows), pd.DataFrame(inner_rows), data


def mixing(nodes: set[str], positives: set[str], edges: pd.DataFrame) -> dict:
    """Directed observed-P/unlabelled mixing, including endpoint-conditioned baselines."""
    if not set(edges.source_grid_id).issubset(nodes) or not set(edges.target_grid_id).issubset(
        nodes
    ):
        raise ValueError("edge endpoint outside audited partition")
    source = edges.source_grid_id.isin(positives).to_numpy()
    target = edges.target_grid_id.isin(positives).to_numpy()
    counts = np.array(
        [
            [(~source & ~target).sum(), (~source & target).sum()],
            [(source & ~target).sum(), (source & target).sum()],
        ],
        float,
    )
    total = counts.sum()
    e = counts / total if total else np.full((2, 2), np.nan)
    expected = float(np.dot(e.sum(axis=1), e.sum(axis=0)))
    assortativity = (np.trace(e) - expected) / (1 - expected) if expected < 1 else np.nan
    pp_given_source = counts[1, 1] / counts[1].sum() if counts[1].sum() else np.nan
    pp_given_target = counts[1, 1] / counts[:, 1].sum() if counts[:, 1].sum() else np.nan
    p = positives & nodes
    return {
        "nodes": len(nodes),
        "recorded_positive_nodes": len(p),
        "recorded_prevalence": len(p) / len(nodes),
        "directed_edges": int(total),
        "UU_edges": int(counts[0, 0]),
        "UP_edges": int(counts[0, 1]),
        "PU_edges": int(counts[1, 0]),
        "PP_edges": int(counts[1, 1]),
        "same_recorded_label_fraction": float(np.trace(e)),
        "directed_recorded_label_assortativity": assortativity,
        "source_endpoint_positive_fraction": e[1].sum(),
        "target_endpoint_positive_fraction": e[:, 1].sum(),
        "target_P_given_source_P": pp_given_source,
        "source_P_given_target_P": pp_given_target,
        "target_P_enrichment_over_endpoint_baseline": pp_given_source / e[:, 1].sum()
        if e[:, 1].sum() > 0
        else np.nan,
        "positive_nodes_with_positive_incoming_neighbor": len(
            set(edges.loc[source & target, "target_grid_id"])
        ),
        "positive_nodes_with_any_incoming_neighbor": len(set(edges.target_grid_id) & p),
    }


def graph_mixing(root: Path, city: str, data: dict, hashes: dict) -> pd.DataFrame:
    base = root / f"data/interim/{city}/graphs/base_edges.parquet"
    outer = root / f"data/interim/{city}/graphs/outer_edges.parquet"
    for path in [base, outer]:
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    full = pd.read_parquet(base, filters=[("graph_id", "in", ["G1", "G2"])])
    cut = pd.read_parquet(outer, filters=[("graph_id", "in", ["G1", "G2"])])
    rows = []
    for graph, edges in full.groupby("graph_id"):
        rows.append(
            {
                "city": city,
                "graph": graph,
                "outer_fold": -1,
                "partition": "full",
                **mixing(set(data["grid"].grid_id), data["positive_ids"], edges),
            }
        )
    for (fold, partition), nodes in data["outer"].groupby(["outer_fold", "partition"]):
        if partition == "buffer":
            continue
        for graph in ["G1", "G2"]:
            edges = cut.loc[
                cut.outer_fold.eq(fold) & cut.partition.eq(partition) & cut.graph_id.eq(graph)
            ]
            if edges.duplicated(["source_grid_id", "target_grid_id"]).any():
                raise ValueError("duplicate directed edges in selected graph")
            rows.append(
                {
                    "city": city,
                    "graph": graph,
                    "outer_fold": fold,
                    "partition": partition,
                    **mixing(set(nodes.grid_id), data["positive_ids"], edges),
                }
            )
    return pd.DataFrame(rows)


def fill_audit(
    root: Path, city: str, data: dict, structure: pd.DataFrame, hashes: dict
) -> pd.DataFrame:
    paths = [
        root / f"data/interim/{city}/{name}"
        for name in ["dsm_200m_float64.tif", "filled_dsm_200m_float64.tif"]
    ]
    values = []
    grid = data["grid"]
    coordinates = list(zip(grid.center_x, grid.center_y, strict=True))
    metadata = []
    for path in paths:
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
        with rasterio.open(path) as raster:
            sampled = np.ma.concatenate(list(raster.sample(coordinates, masked=True)))
            if np.ma.getmaskarray(sampled).any() or not np.isfinite(sampled).all():
                raise ValueError("invalid DSM at final scorable cell")
            values.append(np.asarray(sampled))
            metadata.append((raster.transform, raster.crs, raster.shape))
    if metadata[0] != metadata[1]:
        raise ValueError("original and filled DSM alignment differs")
    depth = values[1] - values[0]
    if depth.min() < -1e-8:
        raise ValueError("filling lowered DSM elevations")
    nodes = grid[["grid_id"]].assign(
        fill_m=np.maximum(depth, 0), recorded_positive=grid.grid_id.isin(data["positive_ids"])
    )
    nodes = nodes.merge(
        structure.loc[structure.city.eq(city), ["grid_id", "outer_fold", "A"]],
        on="grid_id",
        validate="1:1",
    )
    rows = []
    for fold in [-1, *range(5)]:
        local = nodes if fold == -1 else nodes.loc[nodes.outer_fold.eq(fold)]
        for group in ["all", "Z0", "Zcut", "Zplus"]:
            selected = local if group == "all" else local.loc[local.A.eq(group)]
            for population in ["all", "recorded_positive", "unlabelled"]:
                part = (
                    selected
                    if population == "all"
                    else selected.loc[
                        selected.recorded_positive.eq(population == "recorded_positive")
                    ]
                )
                d = part.fill_m
                rows.append(
                    {
                        "city": city,
                        "outer_fold": fold,
                        "A": group,
                        "population": population,
                        "nodes": len(d),
                        "positive_fill_fraction": (d > 1e-6).mean() if len(d) else np.nan,
                        "mean_fill_m": d.mean(),
                        "median_fill_m": d.median() if len(d) else np.nan,
                        "q90_fill_m": d.quantile(0.9),
                        "q95_fill_m": d.quantile(0.95),
                        "max_fill_m": d.max(),
                    }
                )
    return pd.DataFrame(rows)


def feature_missingness(
    root: Path, city: str, data: dict, structure: pd.DataFrame, hashes: dict
) -> pd.DataFrame:
    path = root / f"data/interim/{city}/features_18.parquet"
    hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    features = pd.read_parquet(path).drop(columns="geometry", errors="ignore")
    names = [column for column in features.columns if column != "grid_id"]
    if len(names) != 18:
        raise ValueError("expected the unchanged 18-feature set")
    features = features.merge(
        structure.loc[structure.city.eq(city), ["grid_id", "A"]], on="grid_id", validate="1:1"
    )
    features["recorded_positive"] = features.grid_id.isin(data["positive_ids"])
    rows = []
    for population in ["all", "recorded_positive"]:
        population_rows = (
            features if population == "all" else features.loc[features.recorded_positive]
        )
        for group in ["all", "Z0", "Zcut", "Zplus"]:
            part = (
                population_rows
                if group == "all"
                else population_rows.loc[population_rows.A.eq(group)]
            )
            for name in names:
                missing = int(part[name].isna().sum())
                total_missing = int(population_rows[name].isna().sum())
                rows.append(
                    {
                        "city": city,
                        "population": population,
                        "A": group,
                        "feature": name,
                        "nodes": len(part),
                        "missing_count": missing,
                        "missing_fraction": missing / len(part) if len(part) else np.nan,
                        "nonfinite_count": int((~np.isfinite(part[name])).sum()),
                        "share_of_population_missing_in_group": missing / total_missing
                        if total_missing
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def parameter_counts(config: dict, grid: dict) -> tuple[pd.DataFrame, dict]:
    """CPU-only module construction and algebraic empty-edge equivalence; no fit/optimizer."""
    import torch
    from torch import nn
    from torch_geometric.nn import SAGEConv

    torch.set_num_threads(1)
    rows, checks = [], []
    for variant in config["variants"]:
        for capacity in ["NA"] if variant["model"] == "KDE" else ["SMALL", "MEDIUM"]:
            row = {
                "variant": variant["id"],
                "capacity": capacity,
                "input_dimensions": 2
                if variant["model"] == "KDE"
                else 91
                if variant["model"] == "Neighbor-XGB"
                else 36
                if variant["model"] == "Fixed-filter"
                else 18,
            }
            if variant["model"] not in NEURAL:
                rows.append(
                    {
                        **row,
                        "trainable_parameters": np.nan,
                        "structurally_inactive_parameters": np.nan,
                        "functional_parameter_upper_bound": np.nan,
                        "note": "learned tree size/checkpoint not saved; neural parameter count inapplicable",
                    }
                )
                continue
            hidden = grid["models"][variant["model"]][capacity]["hidden"][0]
            dim = row["input_dimensions"]
            inactive = redundancy = 0
            if variant["model"] == "GraphSAGE-G2":
                layers = variant.get("layers", 2)
                modules = [SAGEConv(dim, hidden, aggr="mean", root_weight=True)]
                if layers == 2:
                    modules.append(SAGEConv(hidden, hidden, aggr="mean", root_weight=True))
                output = nn.Linear(hidden, 1)
                count = sum(p.numel() for module in [*modules, output] for p in module.parameters())
                if variant["graph"] == "G0":
                    inactive = sum(module.lin_l.weight.numel() for module in modules)
                    x = torch.randn(11, dim, generator=torch.Generator().manual_seed(12))
                    edge = torch.empty((2, 0), dtype=torch.long)
                    y = x
                    z = x
                    for module in modules:
                        y = torch.relu(module(y, edge))
                        linear = nn.Linear(module.lin_r.in_channels, hidden)
                        with torch.no_grad():
                            linear.weight.copy_(module.lin_r.weight)
                            linear.bias.copy_(module.lin_l.bias)
                        z = torch.relu(linear(z))
                    gap = (output(y) - output(z)).abs().max().item()
                    output(y).sum().backward()
                    zero = all(
                        torch.count_nonzero(module.lin_l.weight.grad).item() == 0
                        for module in modules
                    )
                    if gap > 1e-6 or not zero:
                        raise ValueError("empty-edge SAGE equivalence/gradient failed")
                    checks.append(
                        {
                            "capacity": capacity,
                            "max_output_difference": gap,
                            "neighbor_weight_gradients_zero": zero,
                        }
                    )
            else:
                model = nn.Sequential(
                    nn.Linear(dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, 1)
                )
                count = sum(p.numel() for p in model.parameters())
                if variant["id"] == "FF-G0-1":
                    redundancy = 18 * hidden  # [X,X] has duplicated first-layer columns.
            rows.append(
                {
                    **row,
                    "hidden_width": hidden,
                    "trainable_parameters": count,
                    "structurally_inactive_parameters": inactive,
                    "duplicate_input_redundancy": redundancy,
                    "functional_parameter_upper_bound": count - inactive - redundancy,
                    "note": "structural functional count, not empirical rank/identifiability",
                }
            )
    return pd.DataFrame(rows), {"device": "cpu", "optimizer_steps": 0, "empty_edge_checks": checks}


def training_health(
    root: Path, run: Path, selections: pd.DataFrame, inner_counts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records, inventory, basenames = [], Counter(), Counter()
    verified_fit_audits = 0
    expected = inner_counts.loc[inner_counts.budget.eq("100")].set_index(
        ["city", "outer_fold", "trajectory", "inner_fold"]
    )
    for path in (run / "inner/tasks").rglob("inner_scores.parquet"):
        receipt = json.loads(path.with_name("status.json").read_text("utf-8"))
        frame = pd.read_parquet(path)
        task = receipt["task"]
        for audit in receipt["fit_audits"]:
            key = (task["city"], int(task["outer_fold"]), task["trajectory"], audit["inner_fold"])
            count = expected.loc[key, "train_revealed_positive"]
            nfit = int(count) + (0 if task["model"] == "KDE" else 2000)
            if audit["fit_row_count"] != nfit:
                raise ValueError(
                    "inner receipt fit rows disagree with reconstructed training count"
                )
            verified_fit_audits += 1
        if task["model"] in NEURAL:
            records.append(
                frame[
                    [
                        "city",
                        "outer_fold",
                        "trajectory",
                        "realization_id",
                        "config",
                        "inner_fold",
                        "best_epoch",
                    ]
                ]
            )
    for phase in ["inner", "outer"]:
        inventory.update(
            f"{phase}:{p.suffix}" for p in (run / phase / "tasks").rglob("*") if p.is_file()
        )
        basenames.update(p.name for p in (run / phase / "tasks").rglob("*") if p.is_file())
    if set(basenames) != {"inner_scores.parquet", "oof.parquet", "status.json"}:
        raise ValueError("unexpected saved task artifacts; inspect for curves/checkpoints")
    epochs = pd.concat(records).rename(columns={"realization_id": "variant"})
    epochs["outer_fold"] = epochs.outer_fold.astype(int)
    epochs = epochs.merge(
        selections[["city", "outer_fold", "trajectory", "variant", "selected_config"]],
        on=["city", "outer_fold", "trajectory", "variant"],
        validate="m:1",
    )
    epochs["selected_candidate"] = epochs.config.eq(epochs.selected_config)
    epochs["implied_stop_epoch_from_frozen_rule"] = implied_stop_epoch(epochs.best_epoch)
    rows = []
    for (city, variant, config, chosen), part in epochs.groupby(
        ["city", "variant", "config", "selected_candidate"]
    ):
        e = part.best_epoch
        rows.append(
            {
                "city": city,
                "variant": variant,
                "capacity": config,
                "selected_candidate": chosen,
                "inner_fold_measurements": len(e),
                "median_best_epoch": e.median(),
                "q25_best_epoch": e.quantile(0.25),
                "q75_best_epoch": e.quantile(0.75),
                "max_best_epoch": e.max(),
                "best_epoch_at_250": int(e.eq(250).sum()),
                "best_epoch_at_250_fraction": e.eq(250).mean(),
                "best_epoch_over_220": int(e.gt(220).sum()),
                "implied_cap_reached_from_frozen_rule": int(e.ge(220).sum()),
                "implied_cap_fraction_from_frozen_rule": e.ge(220).mean(),
            }
        )
    availability = {
        "task_file_inventory": dict(inventory),
        "task_file_basenames": dict(basenames),
        "inner_fit_audits_verified": verified_fit_audits,
        "saved_learning_curves": False,
        "saved_model_checkpoints": False,
        "actual_stop_epoch_saved": False,
        "rule_implied_stop_epoch": "min(best_epoch + 30, 250)",
        "note": "Reconstructed from frozen strict-improvement/stale=30 rule for successful budget100 fits; not independently logged runtime counters or convergence evidence",
    }
    return pd.DataFrame(rows), epochs, availability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/final_review_design_20261012"))
    args = parser.parse_args()
    root = args.root.resolve()
    run = logical_run_path(root, root / args.run_root)
    hashes = {}
    config, grid = verify_provenance(root, run, hashes)
    selections, outer, _ = validate_selections(root, run, config, grid, hashes)
    print("Complete 1250/1550 execution and selections verified; no models fitted.", flush=True)
    structure, _ = structure_inputs(root, hashes)
    counts, inner, mixing_tables, fills, missingness = [], [], [], [], []
    for city in ["zhengzhou", "beijing"]:
        outer_counts, inner_counts, data = design_counts(root, city, hashes)
        counts.append(outer_counts)
        inner.append(inner_counts)
        mixing_tables.append(graph_mixing(root, city, data, hashes))
        fills.append(fill_audit(root, city, data, structure, hashes))
        missingness.append(feature_missingness(root, city, data, structure, hashes))
        print(f"{city}: FULL/100 partition counts, edge mixing and DSM fill complete.", flush=True)
    inner_counts = pd.concat(inner, ignore_index=True)
    health, epoch_rows, availability = training_health(root, run, selections, inner_counts)
    parameters, parameter_checks = parameter_counts(config, grid)
    tables = {
        "outer_partition_counts.csv": pd.concat(counts, ignore_index=True),
        "inner_partition_counts.csv": inner_counts,
        "neural_epoch_summary.csv": health,
        "neural_epoch_measurements.csv": epoch_rows,
        "selected_capacity_counts.csv": selections.groupby(
            ["city", "variant", "selected_config"], as_index=False
        ).size(),
        "parameter_counts.csv": parameters,
        "observed_label_graph_mixing.csv": pd.concat(mixing_tables, ignore_index=True),
        "dsm_fill_by_support_label.csv": pd.concat(fills, ignore_index=True),
        "feature_missingness_by_support.csv": pd.concat(missingness, ignore_index=True),
    }
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if {"grid_id", "geometry", "center_x", "center_y", "score"} & set(table.columns):
            raise ValueError("individual data in aggregate output")
        table.to_csv(output / name, index=False)
    (output / "training_record_availability.json").write_text(
        json.dumps(availability, indent=2) + "\n"
    )
    (output / "parameter_checks.json").write_text(json.dumps(parameter_checks, indent=2) + "\n")
    evidence = {
        "label_coordinate_assumption": {
            "pointers": [
                "src/floodgnn_ceus/labels.py:85",
                "src/floodgnn_ceus/labels.py:161",
                "configs/sources.yaml:2",
            ],
            "evidence": "parser reads wgs_lng/wgs_lat and transforms from EPSG:4326 to AOI CRS",
            "boundary": "implementation assumption; local CrowdFlood source entry has no CRS field; provider CRS declaration not established by this audit",
        },
        "terrain_context": {
            "pointers": [
                "configs/protocol.yaml:54",
                "src/floodgnn_ceus/terrain.py:763",
                "artifacts/reports/dsm_final_aoi.json",
                "artifacts/reports/terrain.json",
            ],
            "evidence": "administrative AOI buffered by 10000m; routing uses full context before final-grid edge filtering",
        },
        "terrain_boundary_nodata": {
            "pointers": [
                "src/floodgnn_ceus/terrain.py:262",
                "src/floodgnn_ceus/terrain.py:278",
                "src/floodgnn_ceus/terrain.py:318",
                "src/floodgnn_ceus/terrain.py:1067",
                "src/floodgnn_ceus/terrain.py:1089",
                "src/floodgnn_ceus/terrain.py:1113",
            ],
            "evidence": "NoData becomes NaN; raster rim or cells adjacent to invalid neighbors seed Priority-Flood; unresolved downslope boundary nodes flow out; final scorable nodes must be finite; exported D8 edges require both endpoints inside final grid",
        },
        "terrain_feature_windows": {
            "pointers": ["src/floodgnn_ceus/terrain.py:488", "src/floodgnn_ceus/terrain.py:506"],
            "evidence": "slope requires 9 valid cells in 3x3 window; 5x5 relief requires >=13 valid cells; relative elevation averages available valid cells; constant-mode filters at context boundary",
        },
        "epoch_stop_rule": {
            "pointers": ["src/floodgnn_ceus/models.py:261", "src/floodgnn_ceus/e1.py:319"],
            "evidence": "budget100 has no fixed_epochs; successful inner fits imply stopping min(best_epoch+30,250) under strict improvement and patience30",
        },
    }
    for name in [
        "configs/protocol.yaml",
        "configs/sources.yaml",
        "artifacts/reports/dsm_final_aoi.json",
        "artifacts/reports/terrain.json",
    ]:
        hashes[name] = sha256_file(root / name)
    (output / "source_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    readme = """# Final-review design diagnostics (no training or manuscript edits)

All 1,250 inner candidates, 1,550 outer refits, and inner-only selections are hash-validated.
FULL and budget-100 inner partitions are rebuilt with the existing 1-km polygon-distance
buffer rule. Counts distinguish all recorded STRICT-LOC positives available in a partition
from revealed positives actually used for training/validation. Each training U sample has
2,000 cells and may include unrevealed recorded positives by the frozen PU protocol.
FULL and 100 have separately label-stratified inner block assignments: equal inner-fold IDs
do not imply identical geometry across budgets. Outer node/positive availability is shared.
New 13-variant execution is budget 100; FULL counts describe the frozen design, not new fits.
All 3,750 inner fit-audit row counts are checked against reconstructed P (+2,000 U except KDE).

Epoch tables distinguish recorded best validation epochs from implied stopping epochs.
The frozen successful inner-fit rule (strict improvement, patience30, no fixed epochs at
budget100) implies min(best_epoch+30,250); cap-reaching therefore corresponds to best>=220.
This algorithmic reconstruction is explicitly separate from best_epoch==250 and is not an
independently saved runtime stop counter or evidence of convergence.
No saved loss/validation curves or tree/neural checkpoints were found in the execution tree.
Training success and finite metrics do not establish optimization convergence or healthy tree
leaf counts. No training execution fault was detected by the completed matrix, hash, pairing,
selection or fit-scope checks. Convergence cannot be directly assessed from the saved artifacts.
This audit performs no additional fits and makes no recommendation to rerun training.

Parameter counts use the actual installed torch/PyG layer components on CPU, with no
optimizer or training. Empty-edge SAGE has neighbor lin_l weights with zero data gradients,
but its lin_l bias remains active; its functional model is the same-width/layer MLP after
mapping root weights and biases. Actual initialization, selected capacity and training path
can differ. AdamW may decay unused weights without making them affect predictions.
FF-G0 concatenates [X,X]: both halves receive gradients, but their first-layer weights are
functionally redundant. The reported functional count is an architecture upper bound,
not an empirical identifiability or effective-complexity estimate. Tree/KDE neural-style
parameter counts are undefined; fitted tree structure is not saved.

Graph mixing is directed, using recorded positives (P) versus unlabelled cells (U), for full
G1/G2 and each outer train/test partition. U is not a verified negative. PP/PU/UP/UU counts,
P-conditioned edge exposure, positive-node prevalence and degree-weighted endpoint baselines
are included so a large UU majority cannot be mistaken for meaningful positive clustering.
Assortativity concerns observed reporting labels, not true inundation homophily, and cannot
identify a physical or model-performance mechanism. No score-based tests or causal claims.

DSM fill is filled minus original DSM, sampled at aligned final-grid centers; it is a terrain
processing change in metres, not flood-water depth. Positive fill means >1e-6 m. Summaries
cross all/recorded-positive/unlabelled populations with A support groups, both city-wide and
within disjoint outer-test folds. Existing covariate diagnostics remain available in
results/reviewer_structure_20261011/covariates.csv; those unchanged group definitions are reused.
Feature missingness is reported for all 18 features by A support and observed-positive scope.
source_evidence.json locates coordinate handling and terrain boundary/NoData implementation;
the labels parser's EPSG:4326 assumption is not represented as a verified provider declaration.
Only aggregate outputs are published; private inputs and individual coordinates/IDs stay local.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    hashes[Path(__file__).relative_to(root).as_posix()] = sha256_file(Path(__file__))
    manifest = {
        "run": run.name,
        "verified_inner_candidates": 1250,
        "verified_outer_refits": len(outer),
        "training_executed": False,
        "source_hashes": hashes,
        "outputs": {
            p.name: sha256_file(p)
            for p in sorted(output.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Aggregate design diagnostics written to {output}", flush=True)


if __name__ == "__main__":
    main()
