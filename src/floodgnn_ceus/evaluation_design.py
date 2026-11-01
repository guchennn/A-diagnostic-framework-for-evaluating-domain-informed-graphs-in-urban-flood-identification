"""Freeze performance-blind spatial splits, reveals, U samples, and evaluation banks."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file

CITY_IDS = ("zhengzhou", "beijing")
EXPECTED_NODES = {"zhengzhou": 25_155, "beijing": 61_443}
LABEL_LAYERS = ("STRICT-LOC", "ALL-LOC")
OUTPUT_NAMES = (
    "outer_partitions.parquet",
    "inner_blocks.parquet",
    "reveals.parquet",
    "u_master_index.parquet",
    "u_samples.parquet",
    "evaluation_banks.parquet",
)


def _seed(seed_name: str) -> int:
    return int.from_bytes(hashlib.sha256(seed_name.encode("utf-8")).digest()[:16], "big")


def _ordered_hash(values: Iterable[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(values)))


def random_order(values: Iterable[str], seed_name: str) -> list[str]:
    """Return a PCG64DXSM permutation after canonical grid-id sorting."""
    ordered = sorted(values)
    if len(ordered) != len(set(ordered)):
        raise ValueError("random-order candidates must be unique")
    rng = np.random.Generator(np.random.PCG64DXSM(_seed(seed_name)))
    return np.asarray(ordered, dtype=object)[rng.permutation(len(ordered))].tolist()


def assign_blocks(blocks: pd.DataFrame, *, folds: int, tolerance: float) -> pd.DataFrame:
    """Assign 4 km blocks with the frozen positive/node greedy loss."""
    required = {"block_row", "block_col", "positive_count", "node_count"}
    if not required.issubset(blocks.columns) or folds < 2:
        raise ValueError("block assignment inputs are incomplete")
    ordered = blocks.sort_values(
        ["positive_count", "node_count", "block_row", "block_col"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    positive = np.zeros(folds, dtype=np.int64)
    nodes = np.zeros(folds, dtype=np.int64)
    positive_target = float(ordered["positive_count"].sum()) / folds
    node_target = float(ordered["node_count"].sum()) / folds
    assignments: list[dict[str, int]] = []
    for block in ordered.itertuples(index=False):
        losses = []
        for fold_id in range(folds):
            candidate_positive = positive.copy()
            candidate_nodes = nodes.copy()
            candidate_positive[fold_id] += int(block.positive_count)
            candidate_nodes[fold_id] += int(block.node_count)
            positive_loss = np.square(
                (candidate_positive - positive_target) / max(positive_target, 1.0)
            ).sum()
            node_loss = np.square((candidate_nodes - node_target) / max(node_target, 1.0)).sum()
            losses.append(float(positive_loss + 0.2 * node_loss))
        minimum = min(losses)
        tied = [fold_id for fold_id, loss in enumerate(losses) if loss - minimum <= tolerance]
        fold_id = min(tied, key=lambda item: (positive[item], nodes[item], item))
        positive[fold_id] += int(block.positive_count)
        nodes[fold_id] += int(block.node_count)
        assignments.append(
            {
                "block_row": int(block.block_row),
                "block_col": int(block.block_col),
                "fold_id": fold_id,
            }
        )
    return pd.DataFrame(assignments).sort_values(
        ["block_row", "block_col"], kind="stable", ignore_index=True
    )


def materialize_partition(
    grid: gpd.GeoDataFrame,
    block_assignment: pd.DataFrame,
    *,
    held_out_fold: int,
    buffer_m: int,
) -> pd.DataFrame:
    """Resolve one fold into mutually exclusive train, buffer, and test nodes."""
    nodes = grid[["grid_id", "row", "column", "geometry"]].copy()
    nodes["block_row"] = np.floor_divide(nodes["row"], 20).astype("int32")
    nodes["block_col"] = np.floor_divide(nodes["column"], 20).astype("int32")
    assignment = block_assignment.rename(columns={"fold_id": "assigned_fold"})
    nodes = nodes.merge(
        assignment[["block_row", "block_col", "assigned_fold"]],
        on=["block_row", "block_col"],
        how="left",
        validate="many_to_one",
    )
    if nodes["assigned_fold"].isna().any():
        raise ValueError("some grid blocks have no fold assignment")
    is_test = nodes["assigned_fold"].eq(held_out_fold).to_numpy()
    if not is_test.any():
        raise ValueError("held-out fold has no test nodes")
    partitions = np.full(len(nodes), "train", dtype=object)
    partitions[is_test] = "test"
    row = nodes["row"].to_numpy()
    column = nodes["column"].to_numpy()
    row_offset = row - row.min()
    column_offset = column - column.min()
    test_mask = np.zeros((row_offset.max() + 1, column_offset.max() + 1), dtype=bool)
    test_mask[row_offset[is_test], column_offset[is_test]] = True
    radius = int(np.ceil(buffer_m / 200))
    delta = np.arange(-radius, radius + 1)
    delta_row, delta_column = np.meshgrid(delta, delta, indexing="ij")
    row_gap = np.maximum(np.abs(delta_row) - 1, 0) * 200
    column_gap = np.maximum(np.abs(delta_column) - 1, 0) * 200
    structure = np.hypot(row_gap, column_gap) < buffer_m
    near_test = ndimage.binary_dilation(test_mask, structure=structure)
    partitions[(~is_test) & near_test[row_offset, column_offset]] = "buffer"
    result = nodes[["grid_id", "block_row", "block_col"]].copy()
    result["partition"] = partitions
    return result.sort_values("grid_id", kind="stable", ignore_index=True)


def select_u(master_order: Iterable[str], revealed: set[str], *, size: int) -> list[str]:
    """Remove only current revealed positives and take the frozen U prefix."""
    selected = [grid_id for grid_id in master_order if grid_id not in revealed][:size]
    if len(selected) < size:
        raise ValueError(f"U candidate pool has fewer than {size} nodes")
    return selected


def sample_bank(
    positives: Iterable[str],
    background_candidates: Iterable[str],
    *,
    ratio: int,
    seed_name: str,
) -> pd.DataFrame:
    """Create one fixed positive-background bank without replacement."""
    positive = sorted(set(positives))
    if not positive:
        raise ValueError("evaluation bank has no positive nodes")
    candidates = sorted(set(background_candidates).difference(positive))
    required = len(positive) * ratio
    if len(candidates) < required:
        raise ValueError("bank background pool is too small")
    background = random_order(candidates, seed_name)[:required]
    return pd.DataFrame(
        {
            "sample_role": ["positive"] * len(positive) + ["background"] * required,
            "sample_rank": list(range(1, len(positive) + 1)) + list(range(1, required + 1)),
            "grid_id": positive + background,
        }
    )


def label_sets_from_ledger(ledger: pd.DataFrame) -> dict[str, set[str]]:
    """Expose only the safe grid-key and label-layer interface."""
    if not {"grid_id", "label_layer"}.issubset(ledger.columns):
        raise ValueError("label ledger misses the model-facing columns")
    safe = ledger[["grid_id", "label_layer"]]
    if safe.isna().any().any() or safe.duplicated().any():
        raise ValueError("label ledger keys must be non-null and unique")
    unknown = set(safe["label_layer"]).difference(LABEL_LAYERS)
    if unknown:
        raise ValueError(f"unknown label layers: {sorted(unknown)}")
    labels = {
        layer: set(safe.loc[safe["label_layer"].eq(layer), "grid_id"]) for layer in LABEL_LAYERS
    }
    if not labels["STRICT-LOC"].issubset(labels["ALL-LOC"]):
        raise ValueError("STRICT-LOC must be a subset of ALL-LOC")
    return labels


def budget_feasibility(available: Iterable[int], required: int) -> dict[str, object]:
    """Apply the city-wide outer positive-count gate without resampling."""
    counts = list(available)
    minimum = min(counts)
    if minimum < required:
        return {
            "status": "not-estimable",
            "reason": "outer_train_positive_count_below_budget",
            "minimum_available": minimum,
            "required": required,
        }
    return {"status": "success", "reason": None, "minimum_available": minimum, "required": required}


def _read_mapping(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a mapping")
    return value


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)
    return path


def _file_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _block_summary(grid: pd.DataFrame, positives: set[str]) -> pd.DataFrame:
    values = grid[["grid_id", "row", "column"]].copy()
    values["block_row"] = np.floor_divide(values["row"], 20).astype("int32")
    values["block_col"] = np.floor_divide(values["column"], 20).astype("int32")
    values["positive"] = values["grid_id"].isin(positives)
    return values.groupby(["block_row", "block_col"], as_index=False).agg(
        positive_count=("positive", "sum"), node_count=("grid_id", "size")
    )


def _master_record(
    *,
    city_id: str,
    scope: str,
    outer_fold: int,
    budget: str,
    trajectory_id: str,
    inner_fold: int,
    candidates: list[str],
    order: list[str],
    seed_name: str,
) -> dict[str, object]:
    return {
        "city_id": city_id,
        "scope": scope,
        "label_layer": "SHARED" if scope == "outer" else "STRICT-LOC",
        "outer_fold": outer_fold,
        "budget": budget,
        "trajectory_id": trajectory_id,
        "inner_fold": inner_fold,
        "candidate_count": len(candidates),
        "seed_name": seed_name,
        "seed_uint128": str(_seed(seed_name)),
        "candidate_sha256": _ordered_hash(sorted(candidates)),
        "order_sha256": _ordered_hash(order),
    }


def _u_rows(
    selected: list[str],
    *,
    city_id: str,
    scope: str,
    label_layer: str,
    outer_fold: int,
    budget: str,
    trajectory_id: str,
    inner_fold: int,
) -> list[dict[str, object]]:
    return [
        {
            "city_id": city_id,
            "scope": scope,
            "label_layer": label_layer,
            "outer_fold": outer_fold,
            "budget": budget,
            "trajectory_id": trajectory_id,
            "inner_fold": inner_fold,
            "u_rank": rank,
            "grid_id": grid_id,
        }
        for rank, grid_id in enumerate(selected, 1)
    ]


def _bank_rows(
    bank: pd.DataFrame,
    *,
    city_id: str,
    scope: str,
    label_layer: str,
    outer_fold: int,
    budget: str,
    trajectory_id: str,
    inner_fold: int,
    bank_id: int,
) -> list[dict[str, object]]:
    keys = {
        "city_id": city_id,
        "scope": scope,
        "label_layer": label_layer,
        "outer_fold": outer_fold,
        "budget": budget,
        "trajectory_id": trajectory_id,
        "inner_fold": inner_fold,
        "bank_id": bank_id,
    }
    return [{**keys, **row} for row in bank.to_dict("records")]


def _outer_design(
    grid: gpd.GeoDataFrame,
    strict: set[str],
    *,
    folds: int,
    buffer_m: int,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[int, gpd.GeoDataFrame], list[dict[str, int]]]:
    assignment = assign_blocks(_block_summary(grid, strict), folds=folds, tolerance=tolerance)
    frames = []
    train_grids = {}
    counts = []
    for outer_fold in range(folds):
        partition = materialize_partition(
            grid, assignment, held_out_fold=outer_fold, buffer_m=buffer_m
        )
        partition.insert(0, "outer_fold", outer_fold)
        frames.append(partition)
        keyed = grid.merge(partition[["grid_id", "partition"]], on="grid_id", validate="one_to_one")
        train = keyed.loc[keyed["partition"].eq("train")].copy()
        test = keyed.loc[keyed["partition"].eq("test")]
        buffer = keyed.loc[keyed["partition"].eq("buffer")]
        train_grids[outer_fold] = train
        counts.append(
            {
                "outer_fold": outer_fold,
                "train_nodes": len(train),
                "buffer_nodes": len(buffer),
                "test_nodes": len(test),
                "train_STRICT-LOC": int(train["grid_id"].isin(strict).sum()),
                "test_STRICT-LOC": int(test["grid_id"].isin(strict).sum()),
            }
        )
    return pd.concat(frames, ignore_index=True), train_grids, counts


def _add_outer_banks(
    rows: list[dict[str, object]],
    grid: gpd.GeoDataFrame,
    outer: pd.DataFrame,
    labels: Mapping[str, set[str]],
    *,
    city_id: str,
    protocol_id: str,
    bank_count: int,
    ratio: int,
) -> None:
    for label_layer in LABEL_LAYERS:
        full_ledger = labels[label_layer]
        for outer_fold in sorted(outer["outer_fold"].unique()):
            test_ids = set(
                outer.loc[
                    outer["outer_fold"].eq(outer_fold) & outer["partition"].eq("test"),
                    "grid_id",
                ]
            )
            positives = test_ids.intersection(full_ledger)
            candidates = test_ids.difference(full_ledger)
            for bank_id in range(1, bank_count + 1):
                seed_name = (
                    f"{protocol_id}/{city_id}/{label_layer}/{outer_fold}/outer/bank/{bank_id}"
                )
                bank = sample_bank(positives, candidates, ratio=ratio, seed_name=seed_name)
                rows.extend(
                    _bank_rows(
                        bank,
                        city_id=city_id,
                        scope="outer",
                        label_layer=label_layer,
                        outer_fold=int(outer_fold),
                        budget="NA",
                        trajectory_id="NA",
                        inner_fold=-1,
                        bank_id=bank_id,
                    )
                )


def _strict_cell(
    *,
    city_id: str,
    protocol_id: str,
    outer_fold: int,
    budget: str,
    trajectory_id: str,
    train: gpd.GeoDataFrame,
    revealed: set[str],
    outer_master: list[str],
    folds: int,
    buffer_m: int,
    tolerance: float,
    u_size: int,
    bank_count: int,
    ratio: int,
) -> dict[str, list[dict[str, object]]]:
    inner_assignment = assign_blocks(
        _block_summary(train, revealed), folds=folds, tolerance=tolerance
    ).rename(columns={"fold_id": "inner_fold"})
    inner_blocks = _block_summary(train, revealed).merge(
        inner_assignment, on=["block_row", "block_col"], validate="one_to_one"
    )
    inner_blocks.insert(0, "trajectory_id", trajectory_id)
    inner_blocks.insert(0, "budget", budget)
    inner_blocks.insert(0, "label_layer", "STRICT-LOC")
    inner_blocks.insert(0, "outer_fold", outer_fold)
    inner_blocks.insert(0, "city_id", city_id)
    block_rows = inner_blocks.to_dict("records")
    master_rows = []
    u_rows = _u_rows(
        select_u(outer_master, revealed, size=u_size),
        city_id=city_id,
        scope="outer",
        label_layer="STRICT-LOC",
        outer_fold=outer_fold,
        budget=budget,
        trajectory_id=trajectory_id,
        inner_fold=-1,
    )
    bank_rows = []
    partition_hashes = []
    for inner_fold in range(folds):
        partition = materialize_partition(
            train,
            inner_assignment.rename(columns={"inner_fold": "fold_id"}),
            held_out_fold=inner_fold,
            buffer_m=buffer_m,
        )
        partition_hashes.append(
            _ordered_hash(
                f"{grid_id}|{block_row}|{block_col}|{partition_name}"
                for grid_id, block_row, block_col, partition_name in zip(
                    partition["grid_id"],
                    partition["block_row"],
                    partition["block_col"],
                    partition["partition"],
                    strict=True,
                )
            )
        )
        train_ids = set(partition.loc[partition["partition"].eq("train"), "grid_id"])
        validation_ids = set(partition.loc[partition["partition"].eq("test"), "grid_id"])
        validation_positive = validation_ids.intersection(revealed)
        if not validation_positive:
            raise ValueError("inner validation fold has zero revealed positives")
        candidates = sorted(train_ids)
        seed_name = f"{protocol_id}/{city_id}/{outer_fold}/{budget}/{trajectory_id}/{inner_fold}/U"
        master = random_order(candidates, seed_name)
        master_rows.append(
            _master_record(
                city_id=city_id,
                scope="inner",
                outer_fold=outer_fold,
                budget=budget,
                trajectory_id=trajectory_id,
                inner_fold=inner_fold,
                candidates=candidates,
                order=master,
                seed_name=seed_name,
            )
        )
        selected = select_u(master, revealed.intersection(train_ids), size=u_size)
        u_rows.extend(
            _u_rows(
                selected,
                city_id=city_id,
                scope="inner",
                label_layer="STRICT-LOC",
                outer_fold=outer_fold,
                budget=budget,
                trajectory_id=trajectory_id,
                inner_fold=inner_fold,
            )
        )
        for bank_id in range(1, bank_count + 1):
            bank_seed = (
                f"{protocol_id}/{city_id}/{outer_fold}/{budget}/{trajectory_id}/"
                f"{inner_fold}/bank/{bank_id}"
            )
            bank = sample_bank(
                validation_positive,
                validation_ids,
                ratio=ratio,
                seed_name=bank_seed,
            )
            bank_rows.extend(
                _bank_rows(
                    bank,
                    city_id=city_id,
                    scope="inner",
                    label_layer="STRICT-LOC",
                    outer_fold=outer_fold,
                    budget=budget,
                    trajectory_id=trajectory_id,
                    inner_fold=inner_fold,
                    bank_id=bank_id,
                )
            )
    return {
        "inner_blocks": block_rows,
        "master_index": master_rows,
        "u_samples": u_rows,
        "banks": bank_rows,
        "partition_hashes": partition_hashes,
    }


def _city_design(
    root: Path,
    city_id: str,
    split: Mapping[str, object],
) -> tuple[dict[str, object], list[Path]]:
    city_root = root / "data" / "interim" / city_id
    grid_path = city_root / "final_scorable_grid.parquet"
    label_path = city_root / "final_label_ledger.parquet"
    grid = gpd.read_parquet(grid_path)[["grid_id", "row", "column", "geometry"]]
    if len(grid) != EXPECTED_NODES[city_id] or grid["grid_id"].duplicated().any():
        raise ValueError(f"final scorable node baseline changed for {city_id}")
    labels = label_sets_from_ledger(pd.read_parquet(label_path))
    if not labels["ALL-LOC"].issubset(set(grid["grid_id"])):
        raise ValueError(f"label ledger references nodes outside the final grid for {city_id}")

    spatial = split["spatial"]
    protocol_id = str(split["seed_protocol_id"])
    outer, train_grids, outer_counts = _outer_design(
        grid,
        labels["STRICT-LOC"],
        folds=int(spatial["outer_folds"]),
        buffer_m=int(spatial["buffer_m"]),
        tolerance=float(spatial["loss_comparison_tolerance"]),
    )
    outer.insert(0, "city_id", city_id)
    bank_rows: list[dict[str, object]] = []
    _add_outer_banks(
        bank_rows,
        grid,
        outer,
        labels,
        city_id=city_id,
        protocol_id=protocol_id,
        bank_count=int(split["banks"]["count"]),
        ratio=int(split["banks"]["positive_background_ratio"]),
    )

    master_index: list[dict[str, object]] = []
    outer_masters: dict[tuple[int, str], list[str]] = {}
    trajectory_ids = [str(item) for item in split["reveals"]["STRICT-LOC"]["trajectory_ids"]]
    full_trajectory = str(split["reveals"]["STRICT-LOC"]["full_trajectory_id"])
    for outer_fold, train in train_grids.items():
        candidates = sorted(train["grid_id"])
        for trajectory_id in [*trajectory_ids, full_trajectory]:
            seed_name = f"{protocol_id}/{city_id}/{outer_fold}/{trajectory_id}/U"
            order = random_order(candidates, seed_name)
            outer_masters[(outer_fold, trajectory_id)] = order
            master_index.append(
                _master_record(
                    city_id=city_id,
                    scope="outer",
                    outer_fold=outer_fold,
                    budget="NA",
                    trajectory_id=trajectory_id,
                    inner_fold=-1,
                    candidates=candidates,
                    order=order,
                    seed_name=seed_name,
                )
            )

    strict_counts = [int(item["train_STRICT-LOC"]) for item in outer_counts]
    statuses: list[dict[str, object]] = []
    reveal_rows: list[dict[str, object]] = []
    inner_block_rows: list[dict[str, object]] = []
    u_rows: list[dict[str, object]] = []
    inner_partition_hashes: list[str] = []
    budgets = split["reveals"]["STRICT-LOC"]["budgets"]
    for raw_budget in budgets:
        budget = str(raw_budget)
        trajectories = [full_trajectory] if budget == "FULL" else trajectory_ids
        if budget != "FULL":
            status = budget_feasibility(strict_counts, int(budget))
            if status["status"] != "success":
                statuses.append({"label_layer": "STRICT-LOC", "budget": budget, **status})
                continue
        staged = {"reveals": [], "inner_blocks": [], "master_index": [], "u": [], "banks": []}
        staged_hashes = []
        failure = None
        try:
            for outer_fold, train in train_grids.items():
                available = sorted(set(train["grid_id"]).intersection(labels["STRICT-LOC"]))
                for trajectory_id in trajectories:
                    if budget == "FULL":
                        revealed_order = available
                        revealed = set(available)
                    else:
                        seed_name = f"{protocol_id}/{city_id}/{outer_fold}/uniform/{trajectory_id}"
                        revealed_order = random_order(available, seed_name)[: int(budget)]
                        revealed = set(revealed_order)
                    staged["reveals"].extend(
                        {
                            "city_id": city_id,
                            "label_layer": "STRICT-LOC",
                            "reveal_mechanism": "uniform",
                            "outer_fold": outer_fold,
                            "budget": budget,
                            "trajectory_id": trajectory_id,
                            "reveal_rank": rank,
                            "grid_id": grid_id,
                        }
                        for rank, grid_id in enumerate(revealed_order, 1)
                    )
                    cell = _strict_cell(
                        city_id=city_id,
                        protocol_id=protocol_id,
                        outer_fold=outer_fold,
                        budget=budget,
                        trajectory_id=trajectory_id,
                        train=train,
                        revealed=revealed,
                        outer_master=outer_masters[(outer_fold, trajectory_id)],
                        folds=int(spatial["inner_folds"]),
                        buffer_m=int(spatial["buffer_m"]),
                        tolerance=float(spatial["loss_comparison_tolerance"]),
                        u_size=int(split["unlabeled_sample_size"]),
                        bank_count=int(split["banks"]["count"]),
                        ratio=int(split["banks"]["positive_background_ratio"]),
                    )
                    staged["inner_blocks"].extend(cell["inner_blocks"])
                    staged["master_index"].extend(cell["master_index"])
                    staged["u"].extend(cell["u_samples"])
                    staged["banks"].extend(cell["banks"])
                    staged_hashes.extend(cell["partition_hashes"])
        except ValueError as error:
            failure = str(error)
        if failure:
            statuses.append(
                {
                    "label_layer": "STRICT-LOC",
                    "budget": budget,
                    "status": "not-estimable",
                    "reason": failure,
                    "minimum_available": min(strict_counts),
                    "required": None if budget == "FULL" else int(budget),
                }
            )
            continue
        reveal_rows.extend(staged["reveals"])
        inner_block_rows.extend(staged["inner_blocks"])
        master_index.extend(staged["master_index"])
        u_rows.extend(staged["u"])
        bank_rows.extend(staged["banks"])
        inner_partition_hashes.extend(staged_hashes)
        statuses.append(
            {
                "label_layer": "STRICT-LOC",
                "budget": budget,
                "status": "success",
                "reason": None,
                "minimum_available": min(strict_counts),
                "required": None if budget == "FULL" else int(budget),
            }
        )

    all_trajectories = [str(item) for item in split["reveals"]["ALL-LOC"]["trajectory_ids"]]
    all_counts = []
    for train in train_grids.values():
        all_counts.append(int(train["grid_id"].isin(labels["ALL-LOC"]).sum()))
    all_status = budget_feasibility(all_counts, 100)
    if all_status["status"] == "success":
        try:
            staged_reveals = []
            staged_u = []
            for outer_fold, train in train_grids.items():
                available = sorted(set(train["grid_id"]).intersection(labels["ALL-LOC"]))
                for trajectory_id in all_trajectories:
                    seed_name = f"{protocol_id}/{city_id}/{outer_fold}/uniform/{trajectory_id}"
                    revealed_order = random_order(available, seed_name)[:100]
                    revealed = set(revealed_order)
                    staged_reveals.extend(
                        {
                            "city_id": city_id,
                            "label_layer": "ALL-LOC",
                            "reveal_mechanism": "uniform",
                            "outer_fold": outer_fold,
                            "budget": "100",
                            "trajectory_id": trajectory_id,
                            "reveal_rank": rank,
                            "grid_id": grid_id,
                        }
                        for rank, grid_id in enumerate(revealed_order, 1)
                    )
                    selected = select_u(
                        outer_masters[(outer_fold, trajectory_id)],
                        revealed,
                        size=int(split["unlabeled_sample_size"]),
                    )
                    staged_u.extend(
                        _u_rows(
                            selected,
                            city_id=city_id,
                            scope="outer",
                            label_layer="ALL-LOC",
                            outer_fold=outer_fold,
                            budget="100",
                            trajectory_id=trajectory_id,
                            inner_fold=-1,
                        )
                    )
        except ValueError as error:
            all_status = {
                "status": "not-estimable",
                "reason": str(error),
                "minimum_available": min(all_counts),
                "required": 100,
            }
        else:
            reveal_rows.extend(staged_reveals)
            u_rows.extend(staged_u)
    statuses.append({"label_layer": "ALL-LOC", "budget": "100", **all_status})

    design_root = city_root / "evaluation_design"
    frames = {
        "outer_partitions.parquet": outer.sort_values(
            ["outer_fold", "grid_id"], kind="stable", ignore_index=True
        ),
        "inner_blocks.parquet": pd.DataFrame(inner_block_rows).sort_values(
            ["budget", "trajectory_id", "outer_fold", "inner_fold", "block_row", "block_col"],
            kind="stable",
            ignore_index=True,
        ),
        "reveals.parquet": pd.DataFrame(reveal_rows).sort_values(
            ["label_layer", "budget", "trajectory_id", "outer_fold", "reveal_rank"],
            kind="stable",
            ignore_index=True,
        ),
        "u_master_index.parquet": pd.DataFrame(master_index).sort_values(
            ["scope", "budget", "trajectory_id", "outer_fold", "inner_fold"],
            kind="stable",
            ignore_index=True,
        ),
        "u_samples.parquet": pd.DataFrame(u_rows).sort_values(
            [
                "scope",
                "label_layer",
                "budget",
                "trajectory_id",
                "outer_fold",
                "inner_fold",
                "u_rank",
            ],
            kind="stable",
            ignore_index=True,
        ),
        "evaluation_banks.parquet": pd.DataFrame(bank_rows).sort_values(
            [
                "scope",
                "label_layer",
                "budget",
                "trajectory_id",
                "outer_fold",
                "inner_fold",
                "bank_id",
                "sample_role",
                "sample_rank",
            ],
            kind="stable",
            ignore_index=True,
        ),
    }
    paths = [_write_frame(frame, design_root / name) for name, frame in frames.items()]
    report = {
        "city_id": city_id,
        "nodes": len(grid),
        "label_cells": {layer: len(values) for layer, values in labels.items()},
        "outer_folds": outer_counts,
        "statuses": statuses,
        "inner_partition_count": len(inner_partition_hashes),
        "inner_partition_bundle_sha256": _ordered_hash(sorted(inner_partition_hashes)),
        "inputs": [_file_record(root, grid_path), _file_record(root, label_path)],
        "outputs": [_file_record(root, path) for path in paths],
    }
    return report, paths


def _feasibility_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# FloodGNN Data Feasibility Report",
        "",
        "This performance-blind report freezes spatial-design feasibility before model training.",
        "No prediction, metric, model ranking, or outer-test performance was produced.",
        "",
        "| City | Label layer | Budget | Status | Reason |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for city in payload["cities"]:
        for status in city["statuses"]:
            lines.append(
                f"| {city['city_id']} | {status['label_layer']} | {status['budget']} | "
                f"{status['status']} | {status['reason'] or '—'} |"
            )
    lines.extend(["", "## Outer-fold counts", ""])
    for city in payload["cities"]:
        lines.extend(
            [
                f"### {city['city_id']}",
                "",
                "| Fold | Train nodes | Buffer nodes | Test nodes | Train STRICT-LOC | Test STRICT-LOC |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for fold in city["outer_folds"]:
            lines.append(
                f"| {fold['outer_fold']} | {fold['train_nodes']} | {fold['buffer_nodes']} | "
                f"{fold['test_nodes']} | {fold['train_STRICT-LOC']} | "
                f"{fold['test_STRICT-LOC']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Safeguards",
            "",
            "- Failed feasibility cells were not redrawn and produced no partial design artifacts.",
            "- Location-derived rows remain under ignored `data/interim`; this report contains counts and hashes only.",
            "- Outer-test labels were used only to freeze evaluation banks, never for selection or balancing inner CV.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_evaluation_design(root: Path) -> dict[str, object]:
    """Write the frozen Module 3 spatial evaluation design for both cities."""
    split_path = root / "configs" / "split_config.yaml"
    label_schema_path = root / "configs" / "label_schema.yaml"
    split = _read_mapping(split_path)
    label_schema = _read_mapping(label_schema_path)
    if split.get("seed_protocol_id") != "FloodGNN-CEUS-ZZ-BJ-v1.0":
        raise ValueError("split seed protocol ID changed")
    if label_schema.get("model_interface_columns") != ["grid_id", "label_layer"]:
        raise ValueError("label design interface changed")
    forbidden = label_schema.get("forbidden_design_columns")
    if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
        raise TypeError("forbidden design columns must be a string list")
    if len(forbidden) != len(set(forbidden)):
        raise ValueError("forbidden design columns contain duplicates")

    cities = []
    for city_id in CITY_IDS:
        report, _ = _city_design(root, city_id, split)
        cities.append(report)
    payload = {
        "schema_version": 1,
        "phase": "E0",
        "performance_blind": True,
        "labels_read": True,
        "label_columns_read": ["grid_id", "label_layer"],
        "predictions_written": False,
        "metrics_written": False,
        "outer_test_used_for_selection": False,
        "seed_protocol_id": split["seed_protocol_id"],
        "configs": [_file_record(root, split_path), _file_record(root, label_schema_path)],
        "cities": cities,
    }
    report_path = root / "artifacts" / "reports" / "evaluation_design.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial = report_path.with_name(f"{report_path.name}.part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, report_path)
    markdown_path = root / "artifacts" / "reports" / "data_feasibility_report.md"
    partial = markdown_path.with_name(f"{markdown_path.name}.part")
    partial.write_text(_feasibility_markdown(payload), encoding="utf-8", newline="\n")
    os.replace(partial, markdown_path)
    return {
        "cities": cities,
        "reports": [str(report_path), str(markdown_path)],
        "performance_blind": True,
        "labels_read": True,
        "label_columns_read": ["grid_id", "label_layer"],
    }
