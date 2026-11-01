"""Paired, spatially blocked inference for the locked FloodGNN analysis."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from floodgnn_ceus.e1 import _status_path, select_outer_models
from floodgnn_ceus.experiments import GRAPH_BY_MODEL, NEURAL_MODELS, E1OuterUnit, outer_units
from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file
from floodgnn_ceus.training import TaskKey

PAIR_KEYS = (
    "budget",
    "trajectory",
    "seed",
    "bank_id",
    "outer_fold",
    "outer_block",
    "grid_id",
    "sample_role",
)
EVALUATION_COLUMNS = ("city", *PAIR_KEYS, "method", "score")
FULL_OOF_COLUMNS = (
    "city",
    "budget",
    "trajectory",
    "seed",
    "outer_fold",
    "grid_id",
    "method",
    "score",
)


@dataclass(frozen=True)
class Comparison:
    comparison_id: str
    budget: str
    treatment: str
    control: str


class BootstrapResult(NamedTuple):
    effects: pd.DataFrame
    accepted: int
    rejected: int
    banks_resampled: bool


def _outer_task(
    protocol_id: str,
    unit: E1OuterUnit,
    *,
    experiment_id: str,
    model: str,
    graph_id: str,
    config: str,
    seed: str,
) -> TaskKey:
    return TaskKey(
        protocol_id=protocol_id,
        experiment_id=experiment_id,
        city=unit.city,
        label_layer="STRICT-LOC",
        reveal_mechanism="uniform",
        outer_fold=str(unit.outer_fold),
        budget=unit.budget,
        trajectory=unit.trajectory,
        feature_set="FULL-COMMON",
        model=model,
        graph_id=graph_id,
        realization_id="NA",
        config=config,
        seed=seed,
    )


def _task_rows(
    task_root: Path, task: TaskKey
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, str]]:
    status_path = _status_path(task_root, task)
    if not status_path.is_file():
        raise FileNotFoundError(f"outer task is unfinished: {task.relative_path()}")
    status = json.loads(status_path.read_text("utf-8"))
    if status.get("status") != "success":
        raise ValueError(f"outer task is not successful: {task.relative_path()}")
    oof_path = status_path.with_name("oof.parquet")
    if not oof_path.is_file() or status.get("oof_sha256") != sha256_file(oof_path):
        raise ValueError(f"outer OOF hash is missing or invalid: {task.relative_path()}")
    frame = pd.read_parquet(
        oof_path,
        columns=["seed", "bank_id", "grid_id", "sample_role", "score", "status"],
    )
    if frame.empty or not frame["status"].eq("success").all():
        raise ValueError(f"outer bank predictions are incomplete: {task.relative_path()}")
    banks = frame.loc[frame["bank_id"].astype(str).ne("NA")].copy()
    full = frame.loc[frame["bank_id"].astype(str).eq("NA")].copy()
    banks["bank_id"] = banks["bank_id"].astype(int)
    if banks.duplicated(["seed", "bank_id", "grid_id"]).any():
        raise ValueError(f"outer bank predictions contain duplicates: {task.relative_path()}")
    if full.empty or full.duplicated(["seed", "grid_id"]).any():
        raise ValueError(f"full outer predictions are incomplete: {task.relative_path()}")
    receipt = (task.relative_path(), sha256_file(oof_path))
    return banks.drop(columns="status"), full.drop(columns="status"), receipt


def _model_bank_rows(
    task_root: Path,
    protocol_id: str,
    unit: E1OuterUnit,
    *,
    experiment_id: str,
    model: str,
    graph_id: str,
    config: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    task_seeds = ("NEURAL-1", "NEURAL-2", "NEURAL-3") if model in NEURAL_MODELS else ("NEURAL-1",)
    frames = []
    full_frames = []
    receipts = []
    for seed in task_seeds:
        task = _outer_task(
            protocol_id,
            unit,
            experiment_id=experiment_id,
            model=model,
            graph_id=graph_id,
            config=config,
            seed=seed,
        )
        frame, full, receipt = _task_rows(task_root, task)
        frames.append(frame)
        full_frames.append(full)
        receipts.append(receipt)
    result = pd.concat(frames, ignore_index=True)
    full_result = pd.concat(full_frames, ignore_index=True)
    if set(result["seed"]) != {"NEURAL-1", "NEURAL-2", "NEURAL-3"}:
        raise ValueError("outer predictions do not expose the three frozen seed slots")
    return result, full_result, receipts


def _normalize_rows(
    frame: pd.DataFrame,
    unit: E1OuterUnit,
    method: str,
    blocks: pd.Series,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "city", unit.city)
    result.insert(1, "budget", unit.budget)
    result.insert(2, "trajectory", unit.trajectory)
    result.insert(5, "outer_fold", unit.outer_fold)
    result.insert(6, "outer_block", result["grid_id"].map(blocks))
    result.insert(len(result.columns) - 1, "method", method)
    if result["outer_block"].isna().any():
        raise ValueError("OOF grid_id is absent from the frozen outer block map")
    return result.loc[:, EVALUATION_COLUMNS]


def _normalize_full_rows(
    frame: pd.DataFrame,
    unit: E1OuterUnit,
    method: str,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "city", unit.city)
    result.insert(1, "budget", unit.budget)
    result.insert(2, "trajectory", unit.trajectory)
    result.insert(4, "outer_fold", unit.outer_fold)
    result.insert(len(result.columns) - 1, "method", method)
    return result.loc[:, FULL_OOF_COLUMNS]


def build_evaluation_table(
    root: Path,
    inner_task_root: Path,
    e1_task_root: Path,
    e2_task_root: Path,
    output_path: Path,
) -> dict[str, object]:
    """Normalize completed E1/E2 bank predictions into one paired inference table."""
    protocol = json.loads((root / "artifacts" / "protocol_lock.json").read_text("utf-8"))
    protocol_id = str(protocol["base_protocol_id"])
    block_maps = {}
    for city in ("zhengzhou", "beijing"):
        grid = pd.read_parquet(
            root / "data" / "interim" / city / "final_scorable_grid.parquet",
            columns=["grid_id", "row", "column"],
        )
        block_maps[city] = grid.set_index("grid_id").apply(
            lambda row: f"{int(row['row']) // 20}:{int(row['column']) // 20}", axis=1
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".part")
    full_output = output_path.with_name(f"{output_path.stem}_full{output_path.suffix}")
    full_partial = full_output.with_name(full_output.name + ".part")
    partial.unlink(missing_ok=True)
    full_partial.unlink(missing_ok=True)
    writer = None
    full_writer = None
    source_receipts: set[tuple[str, str]] = set()
    row_count = 0
    full_row_count = 0
    try:
        for unit in outer_units(root):
            selection = select_outer_models(root, inner_task_root, unit)
            selected_models = {
                "G2": "GraphSAGE-G2",
                "A*": str(selection["best_alternative"]["model"]),
            }
            if unit.budget == "100":
                selected_models.update(
                    {
                        "MLP": "MLP",
                        "Neighbor-XGB": "Neighbor-XGB",
                        "Fixed-filter": "Fixed-filter",
                    }
                )
            for method, model in selected_models.items():
                choice = selection["models"][model]
                rows, full, receipts = _model_bank_rows(
                    e1_task_root,
                    protocol_id,
                    unit,
                    experiment_id="E1",
                    model=model,
                    graph_id=GRAPH_BY_MODEL[model],
                    config=str(choice["config"]),
                )
                normalized = _normalize_rows(rows, unit, method, block_maps[unit.city])
                table = pa.Table.from_pandas(normalized, preserve_index=False)
                writer = writer or pq.ParquetWriter(partial, table.schema)
                writer.write_table(table)
                full_table = pa.Table.from_pandas(
                    _normalize_full_rows(full, unit, method), preserve_index=False
                )
                full_writer = full_writer or pq.ParquetWriter(full_partial, full_table.schema)
                full_writer.write_table(full_table)
                row_count += len(normalized)
                full_row_count += len(full)
                source_receipts.update(receipts)
            if unit.budget == "100":
                choice = selection["models"]["GraphSAGE-G2"]
                for graph_id in ("G0", "G1", "G4"):
                    rows, full, receipts = _model_bank_rows(
                        e2_task_root,
                        protocol_id,
                        unit,
                        experiment_id="E2",
                        model="GraphSAGE-G2",
                        graph_id=graph_id,
                        config=str(choice["config"]),
                    )
                    normalized = _normalize_rows(rows, unit, graph_id, block_maps[unit.city])
                    table = pa.Table.from_pandas(normalized, preserve_index=False)
                    writer = writer or pq.ParquetWriter(partial, table.schema)
                    writer.write_table(table)
                    full_table = pa.Table.from_pandas(
                        _normalize_full_rows(full, unit, graph_id), preserve_index=False
                    )
                    full_writer = full_writer or pq.ParquetWriter(full_partial, full_table.schema)
                    full_writer.write_table(full_table)
                    row_count += len(normalized)
                    full_row_count += len(full)
                    source_receipts.update(receipts)
    finally:
        if writer is not None:
            writer.close()
        if full_writer is not None:
            full_writer.close()
    if writer is None:
        raise ValueError("no complete E1/E2 predictions were found")
    conflicts = [
        path
        for path, candidate in ((output_path, partial), (full_output, full_partial))
        if path.exists() and sha256_file(path) != sha256_file(candidate)
    ]
    if conflicts:
        partial.unlink(missing_ok=True)
        full_partial.unlink(missing_ok=True)
        raise FileExistsError(f"normalized evaluation output differs: {conflicts[0]}")
    if output_path.exists():
        partial.unlink()
    else:
        os.replace(partial, output_path)
    if full_output.exists():
        full_partial.unlink()
    else:
        os.replace(full_partial, full_output)
    receipt_bundle = sha256_bytes(
        canonical_json_bytes(
            [[str(value) for value in receipt] for receipt in sorted(source_receipts)]
        )
    )
    manifest = {
        "output_bytes": output_path.stat().st_size,
        "output_rows": row_count,
        "output_sha256": sha256_file(output_path),
        "full_output_bytes": full_output.stat().st_size,
        "full_output_name": full_output.name,
        "full_output_rows": full_row_count,
        "full_output_sha256": sha256_file(full_output),
        "source_oof_bundle_sha256": receipt_bundle,
        "source_oof_files": len(source_receipts),
        "status": "success",
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    encoded = canonical_json_bytes(manifest)
    if manifest_path.exists() and manifest_path.read_bytes() != encoded:
        raise FileExistsError(f"evaluation manifest differs: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_bytes(encoded)
    return manifest


def weighted_auprc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Compute average precision with explicit nonnegative observation weights."""
    y = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if not (y.shape == values.shape == sample_weights.shape):
        raise ValueError("labels, scores, and weights must align")
    if not np.isfinite(values).all() or not np.isfinite(sample_weights).all():
        raise ValueError("scores and weights must be finite")
    if (sample_weights < 0).any() or sample_weights.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive total")
    observed = set(y[sample_weights > 0].tolist())
    if observed != {0, 1}:
        raise ValueError("effective sample must contain both classes")
    order = np.argsort(-values, kind="stable")
    ordered_scores = values[order]
    positive_weights = sample_weights[order] * y[order]
    total_weights = sample_weights[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(order) - 1]
    true_positive = np.cumsum(positive_weights)[ends]
    predicted = np.cumsum(total_weights)[ends]
    increments = np.diff(np.r_[0.0, true_positive]) / true_positive[-1]
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    return float(np.sum(increments * precision))


def validate_paired_evaluation(frame: pd.DataFrame, methods: tuple[str, ...]) -> None:
    """Require identical bank rows for every method in a paired comparison."""
    required = set(PAIR_KEYS) | {"method", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"evaluation columns are missing: {missing}")
    selected = frame.loc[frame["method"].isin(methods)]
    if set(selected["method"]) != set(methods):
        raise ValueError("comparison methods are missing")
    if selected.duplicated([*PAIR_KEYS, "method"]).any():
        raise ValueError("evaluation contains duplicate method rows")
    counts = selected.groupby(list(PAIR_KEYS), dropna=False)["method"].nunique()
    if len(counts) == 0 or not counts.eq(len(methods)).all():
        raise ValueError("evaluation methods are not paired")
    roles = set(selected["sample_role"])
    if roles != {"positive", "background"}:
        raise ValueError("evaluation roles are incomplete")


def _stable_rng(protocol_id: str, city: str) -> np.random.Generator:
    name = f"{protocol_id}/{city}/bootstrap".encode()
    seed = int.from_bytes(hashlib.sha256(name).digest()[:16], "big", signed=False)
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _block_counts(
    frame: pd.DataFrame, rng: np.random.Generator
) -> dict[tuple[object, object], int]:
    fold_blocks = tuple(
        (fold, np.sort(fold_rows["outer_block"].unique()))
        for fold, fold_rows in frame.groupby("outer_fold", sort=True)
    )
    return _sample_block_counts(fold_blocks, rng)


def _sample_block_counts(
    fold_blocks: tuple[tuple[object, np.ndarray], ...], rng: np.random.Generator
) -> dict[tuple[object, object], int]:
    counts: dict[tuple[object, object], int] = {}
    for fold, blocks in fold_blocks:
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        for block in sampled:
            key = (fold, block)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _rescaled_weights(
    frame: pd.DataFrame,
    block_counts: dict[tuple[object, object], int],
) -> np.ndarray:
    weights = np.array(
        [
            block_counts.get((fold, block), 0)
            for fold, block in frame[["outer_fold", "outer_block"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=np.float64,
    )
    for (_, role), indices in frame.groupby(
        ["outer_fold", "sample_role"], sort=False
    ).groups.items():
        positions = frame.index.get_indexer(indices)
        effective = weights[positions].sum()
        if effective <= 0:
            raise ValueError(f"bootstrap removed all {role} observations from a fold-bank")
        weights[positions] *= len(positions) / effective
    return weights


def _method_metric(
    frame: pd.DataFrame,
    method: str,
    block_counts: dict[tuple[object, object], int] | None,
) -> float:
    selected = frame.loc[frame["method"].eq(method)].reset_index(drop=True)
    labels = selected["sample_role"].eq("positive").to_numpy(dtype=np.int8)
    weights = (
        np.ones(len(selected), dtype=np.float64)
        if block_counts is None
        else _rescaled_weights(selected, block_counts)
    )
    return weighted_auprc(labels, selected["score"].to_numpy(), weights)


def _comparison_effect(
    frame: pd.DataFrame,
    comparison: Comparison,
    trajectories: np.ndarray,
    seeds: np.ndarray,
    block_counts: dict[tuple[object, object], int] | None,
) -> float:
    selected = frame.loc[frame["budget"].astype(str).eq(comparison.budget)]
    values = []
    banks = np.sort(selected["bank_id"].unique())
    for trajectory in trajectories:
        for seed in seeds:
            cell = selected.loc[
                selected["trajectory"].astype(str).eq(str(trajectory))
                & selected["seed"].astype(str).eq(str(seed))
            ]
            for bank in banks:
                bank_rows = cell.loc[cell["bank_id"].eq(bank)]
                treatment = _method_metric(bank_rows, comparison.treatment, block_counts)
                control = _method_metric(bank_rows, comparison.control, block_counts)
                values.append(treatment - control)
    if not values:
        raise ValueError(f"comparison has no rows: {comparison.comparison_id}")
    return float(np.mean(values))


@dataclass(frozen=True)
class _PreparedMetric:
    labels: np.ndarray
    scores: np.ndarray
    unique_block_keys: tuple[tuple[object, object], ...]
    block_ids: np.ndarray
    rescale_groups: tuple[np.ndarray, ...]
    score_order: np.ndarray
    score_ends: np.ndarray


def _prepare_metric(frame: pd.DataFrame, method: str) -> _PreparedMetric:
    selected = frame.loc[frame["method"].eq(method)].reset_index(drop=True)
    groups = selected.groupby(["outer_fold", "sample_role"], sort=False).indices.values()
    scores = selected["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    block_keys = tuple(selected[["outer_fold", "outer_block"]].itertuples(index=False, name=None))
    unique_block_keys = tuple(dict.fromkeys(block_keys))
    block_lookup = {key: index for index, key in enumerate(unique_block_keys)}
    score_order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[score_order]
    return _PreparedMetric(
        labels=selected["sample_role"].eq("positive").to_numpy(dtype=np.int8),
        scores=scores,
        unique_block_keys=unique_block_keys,
        block_ids=np.fromiter((block_lookup[key] for key in block_keys), dtype=np.int32),
        rescale_groups=tuple(np.asarray(indices) for indices in groups),
        score_order=score_order,
        score_ends=np.r_[np.flatnonzero(np.diff(ordered_scores)), len(score_order) - 1],
    )


def _prepared_metric(
    metric: _PreparedMetric,
    block_counts: dict[tuple[object, object], int],
) -> float:
    block_weights = np.fromiter(
        (block_counts.get(key, 0) for key in metric.unique_block_keys),
        dtype=np.float64,
        count=len(metric.unique_block_keys),
    )
    weights = block_weights[metric.block_ids]
    for positions in metric.rescale_groups:
        effective = weights[positions].sum()
        if effective <= 0:
            raise ValueError("bootstrap removed all observations from a fold-bank stratum")
        weights[positions] *= len(positions) / effective
    ordered_weights = weights[metric.score_order]
    true_positive = np.cumsum(ordered_weights * metric.labels[metric.score_order])[
        metric.score_ends
    ]
    predicted = np.cumsum(ordered_weights)[metric.score_ends]
    if true_positive[-1] <= 0 or predicted[-1] <= true_positive[-1]:
        raise ValueError("effective sample must contain both classes")
    increments = np.diff(np.r_[0.0, true_positive]) / true_positive[-1]
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    return float(np.sum(increments * precision))


def _prepare_comparison_cells(
    frame: pd.DataFrame,
    comparison: Comparison,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[tuple[str, str, object], tuple[_PreparedMetric, _PreparedMetric]],
]:
    selected = frame.loc[frame["budget"].astype(str).eq(comparison.budget)]
    trajectories = np.sort(selected["trajectory"].astype(str).unique())
    seeds = np.sort(selected["seed"].astype(str).unique())
    banks = np.sort(selected["bank_id"].unique())
    cells = {}
    for trajectory in trajectories:
        for seed in seeds:
            cell = selected.loc[
                selected["trajectory"].astype(str).eq(trajectory)
                & selected["seed"].astype(str).eq(seed)
            ]
            for bank in banks:
                bank_rows = cell.loc[cell["bank_id"].eq(bank)]
                cells[(trajectory, seed, bank)] = (
                    _prepare_metric(bank_rows, comparison.treatment),
                    _prepare_metric(bank_rows, comparison.control),
                )
    return trajectories, seeds, banks, cells


def _prepared_comparison_effect(
    comparison: Comparison,
    trajectories: np.ndarray,
    seeds: np.ndarray,
    banks: np.ndarray,
    cells: dict[tuple[str, str, object], tuple[_PreparedMetric, _PreparedMetric]],
    block_counts: dict[tuple[object, object], int],
    metric_cache: dict[tuple[str, str, str, object, str], float] | None = None,
) -> float:
    values = []
    for trajectory in trajectories:
        for seed in seeds:
            for bank in banks:
                treatment, control = cells[(str(trajectory), str(seed), bank)]
                effects = []
                for method, metric in (
                    (comparison.treatment, treatment),
                    (comparison.control, control),
                ):
                    key = (comparison.budget, str(trajectory), str(seed), bank, method)
                    if metric_cache is None or key not in metric_cache:
                        result = _prepared_metric(metric, block_counts)
                        if metric_cache is not None:
                            metric_cache[key] = result
                    else:
                        result = metric_cache[key]
                    effects.append(result)
                values.append(effects[0] - effects[1])
    return float(np.mean(values))


def paired_bootstrap(
    frame: pd.DataFrame,
    comparisons: list[Comparison],
    *,
    replicates: int,
    protocol_id: str,
    city: str,
) -> BootstrapResult:
    """Run the frozen trajectory, seed, and held-out-block paired bootstrap."""
    if replicates <= 0 or not comparisons:
        raise ValueError("bootstrap needs positive replicates and comparisons")
    for comparison in comparisons:
        subset = frame.loc[frame["budget"].astype(str).eq(comparison.budget)]
        validate_paired_evaluation(subset, (comparison.treatment, comparison.control))
    rng = _stable_rng(protocol_id, city)
    sparse_sizes = {
        len(frame.loc[frame["budget"].astype(str).eq(item.budget), "trajectory"].unique())
        for item in comparisons
        if item.budget != "FULL"
    }
    seed_sizes = {
        len(frame.loc[frame["budget"].astype(str).eq(item.budget), "seed"].unique())
        for item in comparisons
    }
    if len(sparse_sizes) > 1 or len(seed_sizes) != 1:
        raise ValueError("paired comparisons must share trajectory and seed dimensions")
    trajectory_count = next(iter(sparse_sizes), 1)
    seed_count = next(iter(seed_sizes))
    prepared = {
        comparison.comparison_id: _prepare_comparison_cells(frame, comparison)
        for comparison in comparisons
    }
    fold_blocks = tuple(
        (fold, np.sort(fold_rows["outer_block"].unique()))
        for fold, fold_rows in frame.groupby("outer_fold", sort=True)
    )
    rows = []
    rejected = 0
    attempts = 0
    maximum_attempts = max(100, replicates * 100)
    while len(rows) < replicates * len(comparisons):
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError("bootstrap rejection limit exceeded")
        block_counts = _sample_block_counts(fold_blocks, rng)
        trajectory_positions = rng.integers(0, trajectory_count, size=trajectory_count)
        seed_positions = rng.integers(0, seed_count, size=seed_count)
        replicate_rows = []
        metric_cache: dict[tuple[str, str, str, object, str], float] = {}
        try:
            for comparison in comparisons:
                trajectories, seeds, banks, cells = prepared[comparison.comparison_id]
                if comparison.budget != "FULL":
                    trajectories = trajectories[trajectory_positions]
                seeds = seeds[seed_positions]
                effect = _prepared_comparison_effect(
                    comparison,
                    trajectories,
                    seeds,
                    banks,
                    cells,
                    block_counts,
                    metric_cache,
                )
                replicate_rows.append(
                    {
                        "replicate": len(rows) // len(comparisons),
                        "comparison_id": comparison.comparison_id,
                        "effect": effect,
                    }
                )
        except ValueError:
            rejected += 1
            continue
        rows.extend(replicate_rows)
    effects = pd.DataFrame(rows).sort_values(["replicate", "comparison_id"]).reset_index(drop=True)
    return BootstrapResult(effects, replicates, rejected, False)


def basic_interval(
    point: float,
    bootstrap_values: np.ndarray,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return the centered basic-bootstrap confidence interval."""
    values = np.asarray(bootstrap_values, dtype=np.float64)
    if len(values) == 0 or not 0 < alpha < 1:
        raise ValueError("interval requires bootstrap values and 0 < alpha < 1")
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(2 * point - upper), float(2 * point - lower)


def basic_upper_bound(
    point: float,
    bootstrap_values: np.ndarray,
    *,
    alpha: float = 0.05,
) -> float:
    """Return the one-sided centered basic-bootstrap upper bound."""
    values = np.asarray(bootstrap_values, dtype=np.float64)
    if len(values) == 0 or not 0 < alpha < 1:
        raise ValueError("upper bound requires bootstrap values and 0 < alpha < 1")
    return float(2 * point - np.quantile(values, alpha))


def classify_increment_evidence(upper_bound: float, *, minimum_effect: float) -> str:
    """Separate exclusion of the target increment from an inconclusive estimate."""
    return "strong-negative" if upper_bound < minimum_effect else "statistically-insufficient"


def one_sided_p(
    point: float,
    bootstrap_values: np.ndarray,
    *,
    minimum_effect: float,
) -> float:
    """Return the finite-sample one-sided centered-bootstrap p-value."""
    values = np.asarray(bootstrap_values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("p-value requires bootstrap values")
    errors = values - point
    exceedances = np.count_nonzero(errors >= point - minimum_effect)
    return float((1 + exceedances) / (len(values) + 1))


def two_sided_p(point: float, bootstrap_values: np.ndarray) -> float:
    """Return the finite-sample two-sided centered-bootstrap p-value."""
    values = np.asarray(bootstrap_values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("p-value requires bootstrap values")
    errors = values - point
    upper = (1 + np.count_nonzero(errors >= point)) / (len(values) + 1)
    lower = (1 + np.count_nonzero(errors <= point)) / (len(values) + 1)
    return float(min(1.0, 2 * min(upper, lower)))


def point_effect(frame: pd.DataFrame, comparison: Comparison) -> float:
    """Compute the paired point effect averaged over trajectories, seeds, and banks."""
    selected = frame.loc[frame["budget"].astype(str).eq(comparison.budget)]
    validate_paired_evaluation(selected, (comparison.treatment, comparison.control))
    trajectories = np.sort(selected["trajectory"].astype(str).unique())
    seeds = np.sort(selected["seed"].astype(str).unique())
    return _comparison_effect(selected, comparison, trajectories, seeds, None)


def stability_checks(frame: pd.DataFrame, comparison: Comparison) -> dict[str, object]:
    """Evaluate the five frozen trajectory, bank, seed, and fold sign checks."""
    selected = frame.loc[frame["budget"].astype(str).eq(comparison.budget)]
    validate_paired_evaluation(selected, (comparison.treatment, comparison.control))
    trajectories = np.sort(selected["trajectory"].astype(str).unique())
    seeds = np.sort(selected["seed"].astype(str).unique())
    banks = np.sort(selected["bank_id"].unique())
    folds = np.sort(selected["outer_fold"].unique())
    if (len(trajectories), len(seeds), len(banks), len(folds)) != (5, 3, 3, 5):
        raise ValueError("stability checks require the frozen 5x3x3x5 design")

    def effect(rows: pd.DataFrame) -> float:
        methods = {}
        for method in (comparison.treatment, comparison.control):
            methods[method] = _method_metric(rows, method, None)
        return methods[comparison.treatment] - methods[comparison.control]

    trajectory_values = [
        np.mean(
            [
                effect(
                    selected.loc[
                        selected["trajectory"].astype(str).eq(trajectory)
                        & selected["seed"].astype(str).eq(seed)
                        & selected["bank_id"].eq(bank)
                    ]
                )
                for seed in seeds
                for bank in banks
            ]
        )
        for trajectory in trajectories
    ]
    bank_values = [
        np.mean(
            [
                effect(
                    selected.loc[
                        selected["trajectory"].astype(str).eq(trajectory)
                        & selected["seed"].astype(str).eq(seed)
                        & selected["bank_id"].eq(bank)
                    ]
                )
                for trajectory in trajectories
                for seed in seeds
            ]
        )
        for bank in banks
    ]
    seed_values = [
        np.mean(
            [
                effect(
                    selected.loc[
                        selected["trajectory"].astype(str).eq(trajectory)
                        & selected["seed"].astype(str).eq(seed)
                        & selected["bank_id"].eq(bank)
                    ]
                )
                for trajectory in trajectories
                for bank in banks
            ]
        )
        for seed in seeds
    ]
    fold_values = [
        point_effect(selected.loc[~selected["outer_fold"].eq(fold)], comparison) for fold in folds
    ]
    checks = {
        "trajectory_positive": int(np.count_nonzero(np.asarray(trajectory_values) > 0)) >= 4,
        "banks_positive": bool(np.all(np.asarray(bank_values) > 0)),
        "seeds_positive": int(np.count_nonzero(np.asarray(seed_values) > 0)) >= 2,
        "leave_one_fold_out_positive": bool(np.all(np.asarray(fold_values) > 0)),
    }
    return {**checks, "passed": all(checks.values())}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Adjust a fixed family of p-values with Holm's step-down method."""
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (family_size - rank) * float(p_values[key])))
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def _available_comparisons(frame: pd.DataFrame) -> list[Comparison]:
    methods = set(frame["method"])
    budgets = set(frame["budget"].astype(str))
    comparisons = [
        Comparison(f"Gate-1/{budget}", budget, "G2", "A*")
        for budget in ("20", "50", "100", "200", "FULL")
        if budget in budgets and {"G2", "A*"}.issubset(methods)
    ]
    comparisons.extend(
        Comparison(f"Gate-2/{control}", "100", "G2", control)
        for control in ("MLP", "G0", "Neighbor-XGB", "Fixed-filter")
        if {"G2", control}.issubset(methods)
    )
    comparisons.extend(
        Comparison(f"Gate-3/{control}", "100", "G2", control)
        for control in ("G1", "G4", "mean-G5")
        if {"G2", control}.issubset(methods)
    )
    return comparisons


def _full_direction_anomaly(
    inference: pd.DataFrame,
    boundary: dict[str, object],
) -> dict[str, object]:
    full = inference.loc[inference["comparison_id"].eq("Gate-1/FULL"), "point_effect"]
    if full.empty:
        return {"status": "not-estimable"}
    full_effect = float(full.iloc[0])
    if boundary["status"] != "supported":
        return {"full_effect": full_effect, "status": "not-applicable"}
    stable = inference.loc[
        inference["comparison_id"].eq(f"Gate-1/{boundary['budget']}"), "point_effect"
    ]
    if stable.empty:
        raise ValueError("supported boundary is missing its inference row")
    stable_effect = float(stable.iloc[0])
    return {
        "full_effect": full_effect,
        "stable_effect": stable_effect,
        "status": "direction-reversal" if full_effect * stable_effect < 0 else "none",
    }


def _stable_boundary(gate1: dict[str, dict[str, object]]) -> dict[str, object]:
    budgets = ("50", "100", "200")
    if any(gate1[budget]["status"] == "not-estimable" for budget in budgets):
        return {"budget": "NA", "status": "not-estimable"}
    passed = {
        budget: bool(gate1[budget]["holm_p"] < 0.05 and gate1[budget]["stability_passed"])
        for budget in budgets
    }
    boundary = "NA"
    if passed["50"] and passed["100"] and passed["200"]:
        boundary = "50"
    elif passed["100"] and passed["200"]:
        boundary = "100"
    return {
        "budget": boundary,
        "status": "supported" if boundary != "NA" else "not-supported",
    }


def replication_level(cities: dict[str, dict[str, object]]) -> dict[str, object]:
    """Assign the first matching cross-city utility level from protocol section 14.2."""
    if set(cities) != {"beijing", "zhengzhou"}:
        raise ValueError("replication classification requires Beijing and Zhengzhou")
    budgets = ("50", "100", "200")
    gates = {city: value["Gate-1"] for city, value in cities.items()}
    if any(
        gates[city][budget]["status"] == "not-estimable" for city in cities for budget in budgets
    ):
        return {"level": 1, "status": "not-estimable", "label": "indeterminate"}

    for budget in budgets:
        directions = {gates[city][budget]["exclusion_direction"] for city in cities}
        if directions == {"positive", "negative"}:
            return {
                "level": 2,
                "status": "statistical-contradiction",
                "label": "statistical contradiction",
            }

    if all(float(gates[city][budget]["upper_95"]) < 0.01 for city in cities for budget in budgets):
        return {
            "level": 3,
            "status": "supported",
            "label": "jointly excludes minimum-effect benefit",
        }

    boundaries = {city: cities[city]["stable_budget_boundary"] for city in cities}
    supported = [city for city, boundary in boundaries.items() if boundary["status"] == "supported"]
    if len(supported) == 2:
        values = {city: str(boundaries[city]["budget"]) for city in cities}
        if len(set(values.values())) == 1:
            return {
                "level": 4,
                "status": "supported",
                "label": "boundary-consistent replication",
            }
        common = [budget for budget in budgets if int(budget) >= max(map(int, values.values()))]
        if all(
            float(gates["beijing"][budget]["point_effect"])
            * float(gates["zhengzhou"][budget]["point_effect"])
            > 0
            for budget in common
        ):
            return {
                "level": 5,
                "status": "supported",
                "label": "boundary-shifted utility replication",
            }

    if len(supported) == 1:
        source = supported[0]
        other = next(city for city in cities if city != source)
        stable = [budget for budget in budgets if int(budget) >= int(boundaries[source]["budget"])]
        if all(float(gates[other][budget]["upper_95"]) < 0.01 for budget in stable):
            return {
                "level": 6,
                "status": "supported",
                "label": "city-event-specific",
            }
        if all(float(gates[other][budget]["point_effect"]) > 0 for budget in stable):
            return {"level": 7, "status": "supported", "label": "partial replication"}

    if any(
        float(gates["beijing"][budget]["point_effect"])
        * float(gates["zhengzhou"][budget]["point_effect"])
        < 0
        for budget in budgets
    ):
        return {
            "level": 8,
            "status": "not-supported",
            "label": "directionally inconsistent",
        }
    return {"level": 9, "status": "not-supported", "label": "insufficient evidence"}


def directional_replication(cities: dict[str, dict[str, object]]) -> dict[str, str]:
    """Classify cross-city directional evidence separately from utility."""
    statuses = [str(cities[city]["Gate-3"]["status"]) for city in sorted(cities)]
    if "not-estimable" in statuses:
        return {
            "status": "not-estimable",
            "label": "directional replication indeterminate",
        }
    supported = statuses.count("supported")
    if supported == 2:
        return {"status": "supported", "label": "directional evidence replicated"}
    if supported == 1:
        return {
            "status": "city-event-specific",
            "label": "directional evidence city-event-specific",
        }
    return {
        "status": "not-supported",
        "label": "directional specificity unsupported",
    }


def _graph_g5_statuses(root: Path, cities: set[str]) -> dict[str, str]:
    report = json.loads(
        (root / "artifacts" / "reports" / "graph_counterfactuals.json").read_text("utf-8")
    )
    statuses = {str(item["city_id"]): str(item["G5_status"]) for item in report["cities"]}
    if set(statuses) != cities:
        raise ValueError("graph counterfactual report does not cover every inference city")
    return statuses


def analyze_city(
    frame: pd.DataFrame,
    *,
    protocol_id: str,
    city: str,
    replicates: int,
    g5_status: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compute all estimable paired comparisons and frozen claim gates for one city."""
    if set(frame["city"].astype(str)) != {city}:
        raise ValueError("evaluation input must contain exactly the requested city")
    comparisons = _available_comparisons(frame)
    bootstrap = paired_bootstrap(
        frame,
        comparisons,
        replicates=replicates,
        protocol_id=protocol_id,
        city=city,
    )
    rows = []
    by_id = {comparison.comparison_id: comparison for comparison in comparisons}
    for comparison_id, values in bootstrap.effects.groupby("comparison_id", sort=True):
        comparison = by_id[str(comparison_id)]
        point = point_effect(frame, comparison)
        samples = values["effect"].to_numpy(dtype=np.float64)
        lower, upper = basic_interval(point, samples)
        upper_95 = basic_upper_bound(point, samples)
        threshold = 0.01 if comparison_id.startswith("Gate-1/") else 0.0
        rows.append(
            {
                "comparison_id": comparison_id,
                "budget": comparison.budget,
                "treatment": comparison.treatment,
                "control": comparison.control,
                "point_effect": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "upper_95": upper_95,
                "minimum_effect": threshold,
                "one_sided_p": one_sided_p(point, samples, minimum_effect=threshold),
                "one_sided_p_zero": one_sided_p(point, samples, minimum_effect=0.0),
                "two_sided_p": two_sided_p(point, samples),
                "status": "success",
            }
        )
    inference = pd.DataFrame(rows).sort_values("comparison_id").reset_index(drop=True)

    raw_gate1 = {
        budget: float(
            inference.loc[inference["comparison_id"].eq(f"Gate-1/{budget}"), "one_sided_p"].iloc[0]
        )
        if f"Gate-1/{budget}" in set(inference["comparison_id"])
        else 1.0
        for budget in ("50", "100", "200")
    }
    holm = holm_adjust(raw_gate1)
    raw_two_sided = {
        budget: float(
            inference.loc[inference["comparison_id"].eq(f"Gate-1/{budget}"), "two_sided_p"].iloc[0]
        )
        if f"Gate-1/{budget}" in set(inference["comparison_id"])
        else 1.0
        for budget in ("50", "100", "200")
    }
    holm_two_sided = holm_adjust(raw_two_sided)
    raw_p_zero = {
        budget: float(
            inference.loc[
                inference["comparison_id"].eq(f"Gate-1/{budget}"), "one_sided_p_zero"
            ].iloc[0]
        )
        if f"Gate-1/{budget}" in set(inference["comparison_id"])
        else 1.0
        for budget in ("50", "100", "200")
    }
    holm_p_zero = holm_adjust(raw_p_zero)
    gate1 = {}
    for budget in ("50", "100", "200"):
        comparison_id = f"Gate-1/{budget}"
        if comparison_id not in set(inference["comparison_id"]):
            gate1[budget] = {
                "exclusion_direction": "none",
                "holm_p": 1.0,
                "holm_two_sided_p": 1.0,
                "holm_p_zero": 1.0,
                "increment_evidence": "not-estimable",
                "point_effect": "NA",
                "stability_passed": False,
                "status": "not-estimable",
                "upper_95": "NA",
            }
            continue
        checks = stability_checks(frame, by_id[comparison_id])
        point = float(
            inference.loc[inference["comparison_id"].eq(comparison_id), "point_effect"].iloc[0]
        )
        direction = "none"
        if holm_two_sided[budget] < 0.05 and point != 0:
            direction = "positive" if point > 0 else "negative"
        gate1[budget] = {
            "exclusion_direction": direction,
            "holm_p": holm[budget],
            "holm_two_sided_p": holm_two_sided[budget],
            "holm_p_zero": holm_p_zero[budget],
            "increment_evidence": classify_increment_evidence(
                float(
                    inference.loc[inference["comparison_id"].eq(comparison_id), "upper_95"].iloc[0]
                ),
                minimum_effect=0.01,
            ),
            "point_effect": point,
            "stability_passed": checks["passed"],
            "status": (
                "supported" if holm[budget] < 0.05 and bool(checks["passed"]) else "not-supported"
            ),
            "upper_95": float(
                inference.loc[inference["comparison_id"].eq(comparison_id), "upper_95"].iloc[0]
            ),
        }
        inference.loc[inference["comparison_id"].eq(comparison_id), "holm_p"] = holm[budget]
        inference.loc[inference["comparison_id"].eq(comparison_id), "holm_two_sided_p"] = (
            holm_two_sided[budget]
        )
        inference.loc[inference["comparison_id"].eq(comparison_id), "holm_p_zero"] = holm_p_zero[
            budget
        ]

    gate2_rows = inference.loc[inference["comparison_id"].str.startswith("Gate-2/")]
    gate2_complete = len(gate2_rows) == 4
    gate2_p = float(gate2_rows["one_sided_p"].max()) if gate2_complete else 1.0
    gate2_supported = (
        gate1["100"]["status"] == "supported"
        and gate2_complete
        and gate2_p < 0.05
        and gate2_rows["point_effect"].gt(0).all()
    )
    gate2 = {
        "iut_p": gate2_p,
        "status": "supported" if gate2_supported else "not-supported",
    }

    gate3_rows = inference.loc[inference["comparison_id"].str.startswith("Gate-3/")]
    present_gate3 = set(gate3_rows["comparison_id"])
    missing_gate3 = {"Gate-3/G1", "Gate-3/G4", "Gate-3/mean-G5"} - present_gate3
    if "Gate-3/mean-G5" in present_gate3 and g5_status != "success":
        raise ValueError("mean-G5 inference exists although G5 is not estimable")
    if missing_gate3:
        if "Gate-3/mean-G5" in missing_gate3 and g5_status != "not-estimable":
            raise ValueError("mean-G5 inference is missing although G5 is estimable")
        gate3 = {
            "graph_g5_status": g5_status,
            "missing_comparisons": sorted(missing_gate3),
            "reason": "required directional comparison is not estimable",
            "status": "not-estimable",
        }
    else:
        gate3_p = float(gate3_rows["one_sided_p"].max())
        gate3_supported = (
            gate2_supported and gate3_p < 0.05 and gate3_rows["point_effect"].gt(0).all()
        )
        gate3 = {
            "iut_p": gate3_p,
            "status": "supported" if gate3_supported else "not-supported",
        }
    boundary = _stable_boundary(gate1)
    descriptive = {}
    comparison_ids = set(inference["comparison_id"])
    for budget in ("20", "FULL"):
        comparison_id = f"Gate-1/{budget}"
        if comparison_id not in comparison_ids:
            descriptive[budget] = {"status": "not-estimable"}
            continue
        row = inference.loc[inference["comparison_id"].eq(comparison_id)].iloc[0]
        descriptive[budget] = {
            "ci_lower": float(row["ci_lower"]),
            "ci_upper": float(row["ci_upper"]),
            "point_effect": float(row["point_effect"]),
            "status": "success",
        }
    conclusions: dict[str, object] = {
        "city": city,
        "Gate-1": gate1,
        "Gate-2": gate2,
        "Gate-3": gate3,
        "bootstrap_rejections": bootstrap.rejected,
        "banks_resampled": False,
        "descriptive_budgets": descriptive,
        "full_direction_anomaly": _full_direction_anomaly(inference, boundary),
        "replicates": replicates,
        "stable_budget_boundary": boundary,
    }
    return inference, bootstrap.effects, conclusions


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    frame.to_parquet(partial, index=False)
    if path.exists() and sha256_file(path) != sha256_file(partial):
        partial.unlink()
        raise FileExistsError(f"inference output differs: {path}")
    if path.exists():
        partial.unlink()
    else:
        os.replace(partial, path)


def run_inference(root: Path, evaluation_path: Path, output_root: Path) -> dict[str, object]:
    """Run the locked 10,000-replicate analysis from a normalized local evaluation table."""
    contract = yaml.safe_load((root / "configs" / "analysis_contract.yaml").read_text("utf-8"))
    replicates = int(contract["bootstrap"]["replicates"])
    if replicates != 10_000:
        raise ValueError("the locked analysis requires exactly 10,000 bootstrap replicates")
    frame = pd.read_parquet(evaluation_path)
    required = {"city", *PAIR_KEYS, "method", "score"}
    if not required.issubset(frame.columns):
        raise ValueError("normalized evaluation table is incomplete")
    protocol_id = str(contract["protocol_id"])
    g5_statuses = _graph_g5_statuses(root, set(frame["city"].astype(str)))
    conclusions = {}
    inference_tables = []
    bootstrap_tables = []
    for city, city_rows in frame.groupby("city", sort=True):
        inference, bootstrap, city_conclusions = analyze_city(
            city_rows,
            protocol_id=protocol_id,
            city=str(city),
            replicates=replicates,
            g5_status=g5_statuses[str(city)],
        )
        inference.insert(0, "city", city)
        bootstrap.insert(0, "city", city)
        inference_tables.append(inference)
        bootstrap_tables.append(bootstrap)
        conclusions[str(city)] = city_conclusions
    cross_city = {
        "directional_topology": directional_replication(conclusions),
        "utility_replication": replication_level(conclusions),
    }
    inference_path = output_root / "inference_long.parquet"
    bootstrap_path = output_root / "paired_bootstrap.parquet"
    conclusions_path = output_root / "conclusions.json"
    _write_parquet(pd.concat(inference_tables, ignore_index=True), inference_path)
    _write_parquet(pd.concat(bootstrap_tables, ignore_index=True), bootstrap_path)
    encoded = canonical_json_bytes(
        {"cities": conclusions, "cross_city": cross_city, "schema_version": 1}
    )
    conclusions_path.parent.mkdir(parents=True, exist_ok=True)
    if conclusions_path.exists() and conclusions_path.read_bytes() != encoded:
        raise FileExistsError(f"inference conclusions differ: {conclusions_path}")
    if not conclusions_path.exists():
        partial = conclusions_path.with_name(conclusions_path.name + ".part")
        partial.write_bytes(encoded)
        os.replace(partial, conclusions_path)
    manifest = {
        "bootstrap_sha256": sha256_file(bootstrap_path),
        "conclusions_sha256": sha256_file(conclusions_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "inference_sha256": sha256_file(inference_path),
        "replicates": replicates,
        "status": "success",
    }
    manifest_path = output_root / "inference_manifest.json"
    manifest_bytes = canonical_json_bytes(manifest)
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise FileExistsError(f"inference manifest differs: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_bytes)
    return manifest
