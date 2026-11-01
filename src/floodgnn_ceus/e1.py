"""Run explicit E1 inner-CV tasks with atomic status and result shards."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from floodgnn_ceus.evaluation_design import materialize_partition
from floodgnn_ceus.experiments import (
    MODELS,
    NEURAL_MODELS,
    E1OuterUnit,
    inner_task_keys,
)
from floodgnn_ceus.graphs import cut_edges
from floodgnn_ceus.io import canonical_json_bytes, sha256_file
from floodgnn_ceus.models import (
    fixed_filter_features,
    mean_bank_auprc,
    neighbor_features,
    predict_kde,
    predict_tree,
    train_neural,
)
from floodgnn_ceus.protocol import EXPECTED_FEATURE_NAMES
from floodgnn_ceus.training import (
    ALTERNATIVE_ORDER,
    TASK_FIELDS,
    TaskKey,
    TaskResourceMonitor,
    balanced_sample_weights,
    fit_audit,
    fit_preprocessor,
    seed_from_name,
    task_action,
    write_oof,
)

INNER_RESULT_COLUMNS = (
    *TASK_FIELDS,
    "inner_fold",
    "score",
    "best_epoch",
    "status",
)


def _read_city(root: Path, city: str) -> dict[str, pd.DataFrame]:
    city_root = root / "data" / "interim" / city
    design = city_root / "evaluation_design"
    graphs = city_root / "graphs"
    return {
        "grid": gpd.read_parquet(city_root / "final_scorable_grid.parquet"),
        "features": pd.read_parquet(city_root / "features_18.parquet"),
        "outer": pd.read_parquet(design / "outer_partitions.parquet"),
        "inner": pd.read_parquet(design / "inner_blocks.parquet"),
        "reveals": pd.read_parquet(design / "reveals.parquet"),
        "u": pd.read_parquet(design / "u_samples.parquet"),
        "banks": pd.read_parquet(design / "evaluation_banks.parquet"),
        "edges": pd.read_parquet(graphs / "base_edges.parquet"),
        "outer_edges": pd.read_parquet(graphs / "outer_edges.parquet"),
    }


def _select(frame: pd.DataFrame, **values: object) -> pd.DataFrame:
    selected = np.ones(len(frame), dtype=bool)
    for column, value in values.items():
        selected &= frame[column].eq(value).to_numpy()
    return frame.loc[selected].copy()


def _inner_fold_data(
    data: dict[str, pd.DataFrame],
    task: TaskKey,
    inner_fold: int,
) -> dict[str, object]:
    outer_fold = int(task.outer_fold)
    outer = _select(data["outer"], outer_fold=outer_fold)
    outer_train = set(outer.loc[outer["partition"].eq("train"), "grid_id"])
    grid = data["grid"].loc[data["grid"]["grid_id"].isin(outer_train)].copy()
    blocks = _select(
        data["inner"],
        outer_fold=outer_fold,
        label_layer=task.label_layer,
        budget=task.budget,
        trajectory_id=task.trajectory,
    ).rename(columns={"inner_fold": "fold_id"})
    partitions = materialize_partition(grid, blocks, held_out_fold=inner_fold, buffer_m=1000)
    usable = partitions.loc[partitions["partition"].isin(("train", "test"))]
    node_ids = sorted(usable["grid_id"].astype(str))
    positions = {grid_id: index for index, grid_id in enumerate(node_ids)}
    train_ids = set(usable.loc[usable["partition"].eq("train"), "grid_id"])
    validation_ids = set(usable.loc[usable["partition"].eq("test"), "grid_id"])

    reveals = _select(
        data["reveals"],
        label_layer=task.label_layer,
        reveal_mechanism=task.reveal_mechanism,
        outer_fold=outer_fold,
        budget=task.budget,
        trajectory_id=task.trajectory,
    )
    positive_ids = sorted(set(reveals["grid_id"]).intersection(train_ids))
    u = _select(
        data["u"],
        scope="inner",
        label_layer=task.label_layer,
        outer_fold=outer_fold,
        budget=task.budget,
        trajectory_id=task.trajectory,
        inner_fold=inner_fold,
    ).sort_values("u_rank", kind="stable")
    u_ids = u["grid_id"].astype(str).tolist()
    if not positive_ids or len(u_ids) != 2000:
        raise ValueError("inner training P or U is incomplete")
    if not set(u_ids).issubset(train_ids) or not set(u_ids).isdisjoint(positive_ids):
        raise ValueError("inner training P/U scope leaked")

    banks = _select(
        data["banks"],
        scope="inner",
        label_layer=task.label_layer,
        outer_fold=outer_fold,
        budget=task.budget,
        trajectory_id=task.trajectory,
        inner_fold=inner_fold,
    )
    if set(banks["bank_id"]) != {1, 2, 3} or not set(banks["grid_id"]).issubset(validation_ids):
        raise ValueError("inner validation bank is incomplete or outside validation")
    banks["node_index"] = banks["grid_id"].map(positions)

    city_edges = data["edges"]
    inner_nodes = set(grid["grid_id"].astype(str))
    city_edges = city_edges.loc[
        city_edges["source_grid_id"].isin(inner_nodes)
        & city_edges["target_grid_id"].isin(inner_nodes)
    ]
    cut = cut_edges(city_edges, partitions)
    return {
        "banks": banks,
        "cut_edges": cut,
        "grid": grid.set_index("grid_id").loc[node_ids],
        "node_ids": node_ids,
        "positions": positions,
        "positive_ids": positive_ids,
        "raw": data["features"].set_index("grid_id").loc[node_ids, list(EXPECTED_FEATURE_NAMES)],
        "train_node_ids": sorted(train_ids),
        "u_ids": u_ids,
        "validation_node_ids": sorted(validation_ids),
    }


def _edge_index(fold: dict[str, object], graph_id: str) -> np.ndarray:
    if graph_id in {"NA", "G0"}:
        return np.empty((2, 0), dtype=np.int64)
    edges = fold["cut_edges"]
    assert isinstance(edges, pd.DataFrame)
    edges = edges.loc[edges["graph_id"].eq(graph_id)]
    positions = fold["positions"]
    assert isinstance(positions, dict)
    return np.asarray(
        [
            [positions[value] for value in edges["source_grid_id"]],
            [positions[value] for value in edges["target_grid_id"]],
        ],
        dtype=np.int64,
    )


def _partition_edge_index(
    fold: dict[str, object],
    graph_id: str,
    partition: str,
    node_ids: list[str],
) -> np.ndarray:
    edges = fold["cut_edges"]
    assert isinstance(edges, pd.DataFrame)
    if graph_id in {"NA", "G0"}:
        return np.empty((2, 0), dtype=np.int64)
    edges = edges.loc[edges["graph_id"].eq(graph_id) & edges["partition"].eq(partition)]
    positions = {grid_id: index for index, grid_id in enumerate(node_ids)}
    return np.asarray(
        [
            [positions[value] for value in edges["source_grid_id"]],
            [positions[value] for value in edges["target_grid_id"]],
        ],
        dtype=np.int64,
    )


def _score_inner_fold(
    fold: dict[str, object],
    task: TaskKey,
    model_grid: dict[str, object],
) -> tuple[float, int, dict[str, object]]:
    raw = fold["raw"]
    banks = fold["banks"]
    grid = fold["grid"]
    positions = fold["positions"]
    positive_ids = fold["positive_ids"]
    u_ids = fold["u_ids"]
    train_node_ids = fold["train_node_ids"]
    validation_node_ids = fold["validation_node_ids"]
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(banks, pd.DataFrame)
    assert isinstance(grid, pd.DataFrame)
    assert isinstance(positions, dict)
    assert isinstance(positive_ids, list)
    assert isinstance(u_ids, list)
    assert isinstance(train_node_ids, list)
    assert isinstance(validation_node_ids, list)
    sample_ids = [*positive_ids, *u_ids]
    sample_indices = np.asarray([positions[value] for value in sample_ids], dtype=np.int64)
    labels = np.concatenate(
        [np.ones(len(positive_ids), dtype=np.float64), np.zeros(len(u_ids), dtype=np.float64)]
    )
    weights = balanced_sample_weights(len(positive_ids), len(u_ids))
    seed_name = model_grid["seeds"]["namespace"].format(
        protocol_id=task.protocol_id,
        experiment_id="E1",
        city=task.city,
        outer_fold=task.outer_fold,
        budget=task.budget,
        trajectory_id=task.trajectory,
        model=task.model,
        config=task.config,
        seed_id=task.seed,
    )
    seed = seed_from_name(seed_name)
    train_cut_edges = np.empty((2, 0), dtype=np.int64)
    evaluation_cut_edges = np.empty((2, 0), dtype=np.int64)
    if task.graph_id != "NA":
        train_cut_edges = _partition_edge_index(fold, task.graph_id, "train", train_node_ids)
        evaluation_cut_edges = _partition_edge_index(
            fold, task.graph_id, "test", validation_node_ids
        )

    if task.model == "KDE":
        coordinates = grid[["center_x", "center_y"]].to_numpy(dtype=np.float64)
        positive_coordinates = coordinates[sample_indices[: len(positive_ids)]]
        scores = predict_kde(positive_coordinates, coordinates)
        audit = fit_audit(
            positive_ids,
            None,
            fit_matrix=positive_coordinates,
            prediction_matrix=coordinates,
            evaluation_cut_edge_index=evaluation_cut_edges,
            train_cut_edge_index=train_cut_edges,
        )
        return mean_bank_auprc(scores, banks), 0, audit

    scale = task.model in {"MLP", "Fixed-filter", "GraphSAGE-G2"}
    preprocessor = fit_preprocessor(raw, sample_ids, scale=scale)
    x = preprocessor.transform(raw).to_numpy(dtype=np.float64)
    if task.model == "Neighbor-XGB":
        g1 = _edge_index(fold, "G1")
        x = neighbor_features(raw.to_numpy(dtype=np.float64), x, g1)
    elif task.model == "Fixed-filter":
        x = fixed_filter_features(
            x, _edge_index(fold, task.graph_id), steps=model_grid.get("fixed_filter_steps", 1)
        )
    params = model_grid["models"][task.model][task.config]
    if task.model in {"XGBoost", "LightGBM", "Neighbor-XGB"}:
        training_x = x[sample_indices]
        scores = predict_tree(
            task.model,
            params,
            training_x,
            labels,
            weights,
            x,
            seed=seed,
        )
        audit = fit_audit(
            sample_ids,
            preprocessor,
            fit_matrix=training_x,
            prediction_matrix=x,
            evaluation_cut_edge_index=evaluation_cut_edges,
            train_cut_edge_index=train_cut_edges,
        )
        return mean_bank_auprc(scores, banks), 0, audit
    train_positions = np.asarray([positions[value] for value in train_node_ids], dtype=np.int64)
    validation_positions = np.asarray(
        [positions[value] for value in validation_node_ids], dtype=np.int64
    )
    train_sample_positions = {value: index for index, value in enumerate(train_node_ids)}
    train_indices = np.asarray(
        [train_sample_positions[value] for value in sample_ids], dtype=np.int64
    )
    validation_positions_by_id = {value: index for index, value in enumerate(validation_node_ids)}
    validation_banks = banks.copy()
    validation_banks["node_index"] = validation_banks["grid_id"].map(validation_positions_by_id)
    if task.model == "GraphSAGE-G2":
        training_x = x[train_positions]
        training_indices = train_indices
        training_edges = train_cut_edges
    else:
        training_x = x[sample_indices]
        training_indices = np.arange(len(sample_indices), dtype=np.int64)
        training_edges = np.empty((2, 0), dtype=np.int64)
    low_data = task.budget == str(model_grid["low_data"]["budget"])
    result = train_neural(
        task.model,
        params,
        training_x,
        training_edges,
        training_indices,
        labels,
        weights,
        None if low_data else validation_banks,
        seed=seed,
        max_epochs=int(model_grid["neural_training"]["max_epochs"]),
        patience=int(model_grid["neural_training"]["patience"]),
        fixed_epochs=int(model_grid["low_data"]["epochs"]) if low_data else None,
        prediction_x=x[validation_positions],
        prediction_edge_index=evaluation_cut_edges,
        **({"layers": model_grid["graphsage_layers"]} if "graphsage_layers" in model_grid else {}),
    )
    audit = fit_audit(
        sample_ids,
        preprocessor,
        fit_matrix=training_x,
        prediction_matrix=x[validation_positions],
        evaluation_cut_edge_index=evaluation_cut_edges,
        train_cut_edge_index=train_cut_edges,
    )
    return mean_bank_auprc(result.scores, validation_banks), result.best_epoch, audit


def _status_path(run_root: Path, task: TaskKey) -> Path:
    return run_root / "tasks" / task.relative_path() / "status.json"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, path)


def run_inner_task(
    root: Path,
    run_root: Path,
    task: TaskKey,
    data: dict[str, pd.DataFrame],
    model_grid: dict[str, object],
    *,
    prepared_folds: list[dict[str, object]] | None = None,
) -> str:
    """Run or resume one complete three-fold E1 inner candidate."""
    status_path = _status_path(run_root, task)
    previous = None
    if status_path.is_file():
        previous = json.loads(status_path.read_text("utf-8")).get("status")
    action = task_action(previous)
    if action == "skip":
        return str(previous)
    started = time.perf_counter()
    monitor = TaskResourceMonitor()
    try:
        with monitor:
            rows = []
            fold_audits = []
            for inner_fold in range(3):
                fold = (
                    _inner_fold_data(data, task, inner_fold)
                    if prepared_folds is None
                    else prepared_folds[inner_fold]
                )
                score, best_epoch, audit = _score_inner_fold(fold, task, model_grid)
                fold_audits.append({"inner_fold": inner_fold, **audit})
                rows.append(
                    {
                        **task.as_dict(),
                        "inner_fold": inner_fold,
                        "score": score,
                        "best_epoch": best_epoch,
                        "status": "success",
                    }
                )
            result = pd.DataFrame(rows, columns=INNER_RESULT_COLUMNS)
            result_path = status_path.with_name("inner_scores.parquet")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            partial = result_path.with_name(result_path.name + ".part")
            result.to_parquet(partial, index=False)
            if result_path.exists() and sha256_file(result_path) != sha256_file(partial):
                raise FileExistsError("successful inner result changed on resume")
            if result_path.exists():
                partial.unlink()
            else:
                os.replace(partial, result_path)
        payload = {
            "fit_audits": fold_audits,
            "result_sha256": sha256_file(result_path),
            "status": "success",
            "task": task.as_dict(),
            "wall_seconds": time.perf_counter() - started,
        }
        status = "success"
    except Exception as error:  # noqa: BLE001 - every task failure must become explicit state.
        payload = {
            "error": f"{type(error).__name__}: {error}",
            "status": "failed",
            "task": task.as_dict(),
            "wall_seconds": time.perf_counter() - started,
        }
        status = "failed"
    payload["resources"] = monitor.record()
    _write_json(status_path, payload)
    return status


def _unit_tasks(root: Path, unit: E1OuterUnit) -> list[TaskKey]:
    return [
        task
        for task in inner_task_keys(root)
        if task.city == unit.city
        and int(task.outer_fold) == unit.outer_fold
        and task.budget == unit.budget
        and task.trajectory == unit.trajectory
    ]


def select_outer_models(
    root: Path,
    inner_task_root: Path,
    unit: E1OuterUnit,
    *,
    task_keys: list[TaskKey] | None = None,
    models: tuple[str, ...] = MODELS,
    alternative_order: tuple[str, ...] = ALTERNATIVE_ORDER,
) -> dict[str, object]:
    """Select complete model configurations without shrinking the family set."""
    rows = []
    invalid = []
    for task in _unit_tasks(root, unit) if task_keys is None else task_keys:
        status_path = _status_path(inner_task_root, task)
        if not status_path.is_file():
            raise FileNotFoundError(f"inner task is unfinished: {task.relative_path()}")
        status = json.loads(status_path.read_text("utf-8")).get("status")
        if status == "failed":
            invalid.append({"config": task.config, "model": task.model})
            continue
        if status != "success":
            raise ValueError(f"inner task has invalid status: {status}")
        result = pd.read_parquet(status_path.with_name("inner_scores.parquet"))
        if set(result["inner_fold"]) != {0, 1, 2} or not result["status"].eq("success").all():
            raise ValueError("inner result is incomplete")
        rows.append(result)
    scores = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    selected = {}
    config_order = {"NA": 0, "SMALL": 1, "MEDIUM": 2}
    for model in models:
        choices = []
        model_scores = scores.loc[scores["model"].eq(model)]
        for config, frame in model_scores.groupby("config", sort=False):
            if set(frame["inner_fold"]) == {0, 1, 2}:
                choices.append(
                    {
                        "best_epochs": frame.sort_values("inner_fold")["best_epoch"]
                        .astype(int)
                        .tolist(),
                        "config": str(config),
                        "score": float(frame["score"].mean()),
                    }
                )
        if not choices:
            raise ValueError(f"required model family has no complete config: {model}")
        choices.sort(key=lambda item: (-float(item["score"]), config_order[str(item["config"])]))
        selected[model] = choices[0]
    alternatives = list(alternative_order)
    best_score = max(float(selected[model]["score"]) for model in alternatives)
    best_model = next(
        model for model in alternatives if float(selected[model]["score"]) == best_score
    )
    return {
        "best_alternative": {
            "config": selected[best_model]["config"],
            "model": best_model,
            "score": best_score,
        },
        "invalid_configs": invalid,
        "models": selected,
        "unit": {
            "budget": unit.budget,
            "city": unit.city,
            "outer_fold": unit.outer_fold,
            "trajectory": unit.trajectory,
        },
    }


def _outer_fold_data(
    data: dict[str, pd.DataFrame],
    unit: E1OuterUnit,
    *,
    label_layer: str = "STRICT-LOC",
    reveal_mechanism: str = "uniform",
) -> dict[str, object]:
    partitions = _select(data["outer"], outer_fold=unit.outer_fold)
    usable = partitions.loc[partitions["partition"].isin(("train", "test"))]
    train_node_ids = sorted(usable.loc[usable["partition"].eq("train"), "grid_id"].astype(str))
    test_node_ids = sorted(usable.loc[usable["partition"].eq("test"), "grid_id"].astype(str))
    node_ids = [*train_node_ids, *test_node_ids]
    positions = {grid_id: index for index, grid_id in enumerate(node_ids)}
    reveals = _select(
        data["reveals"],
        label_layer=label_layer,
        reveal_mechanism=reveal_mechanism,
        outer_fold=unit.outer_fold,
        budget=unit.budget,
        trajectory_id=unit.trajectory,
    )
    positive_ids = sorted(reveals["grid_id"].astype(str))
    u = _select(
        data["u"],
        scope="outer",
        label_layer=label_layer,
        outer_fold=unit.outer_fold,
        budget=unit.budget,
        trajectory_id=unit.trajectory,
        inner_fold=-1,
    ).sort_values("u_rank", kind="stable")
    u_ids = u["grid_id"].astype(str).tolist()
    train_ids = set(train_node_ids)
    if not positive_ids or len(u_ids) != 2000:
        raise ValueError("outer training P or U is incomplete")
    if not set(positive_ids).union(u_ids).issubset(train_ids):
        raise ValueError("outer training sample leaked outside train")
    if not set(positive_ids).isdisjoint(u_ids):
        raise ValueError("outer P and U overlap")
    banks = _select(
        data["banks"],
        scope="outer",
        label_layer=label_layer,
        outer_fold=unit.outer_fold,
        budget="NA",
        trajectory_id="NA",
        inner_fold=-1,
    )
    if set(banks["bank_id"]) != {1, 2, 3} or not set(banks["grid_id"]).issubset(test_node_ids):
        raise ValueError("outer evaluation bank is incomplete or outside test")
    test_positions = {grid_id: index for index, grid_id in enumerate(test_node_ids)}
    banks["node_index"] = banks["grid_id"].map(test_positions)
    edges = _select(data["outer_edges"], outer_fold=unit.outer_fold)
    return {
        "banks": banks,
        "edges": edges,
        "grid": data["grid"].set_index("grid_id").loc[node_ids],
        "node_ids": node_ids,
        "positive_ids": positive_ids,
        "positions": positions,
        "raw": data["features"].set_index("grid_id").loc[node_ids, list(EXPECTED_FEATURE_NAMES)],
        "test_node_ids": test_node_ids,
        "train_node_ids": train_node_ids,
        "u_ids": u_ids,
    }


def _outer_edge_index(
    fold: dict[str, object], graph_id: str, partition: str, node_ids: list[str]
) -> np.ndarray:
    if graph_id in {"NA", "G0"}:
        return np.empty((2, 0), dtype=np.int64)
    edges = fold["edges"]
    assert isinstance(edges, pd.DataFrame)
    edges = edges.loc[
        edges["graph_id"].eq(graph_id)
        & edges["partition"].eq(partition)
        & edges["realization_id"].eq("NA")
    ]
    positions = {grid_id: index for index, grid_id in enumerate(node_ids)}
    return np.asarray(
        [
            [positions[value] for value in edges["source_grid_id"]],
            [positions[value] for value in edges["target_grid_id"]],
        ],
        dtype=np.int64,
    )


def _partition_local_g2_indegree(fold: dict[str, object]) -> np.ndarray:
    """Return log1p in-degree within each outer partition's G2 graph, aligned with fold order."""
    train_ids = fold["train_node_ids"]
    test_ids = fold["test_node_ids"]
    assert isinstance(train_ids, list)
    assert isinstance(test_ids, list)
    train_edges = _outer_edge_index(fold, "G2", "train", train_ids)
    test_edges = _outer_edge_index(fold, "G2", "test", test_ids)
    train_degrees = np.bincount(train_edges[1], minlength=len(train_ids)).astype(np.float64)
    test_degrees = np.bincount(test_edges[1], minlength=len(test_ids)).astype(np.float64)
    return np.log1p(np.concatenate([train_degrees, test_degrees]))


def _predict_outer(
    fold: dict[str, object],
    task: TaskKey,
    model_grid: dict[str, object],
    epochs: int,
    *,
    aggregator: str = "mean",
    extra_node_features: tuple[str, ...] = (),
) -> tuple[np.ndarray, dict[str, object]]:
    raw = fold["raw"]
    grid = fold["grid"]
    positive_ids = fold["positive_ids"]
    u_ids = fold["u_ids"]
    node_ids = fold["node_ids"]
    train_node_ids = fold["train_node_ids"]
    test_node_ids = fold["test_node_ids"]
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(grid, pd.DataFrame)
    assert isinstance(positive_ids, list)
    assert isinstance(u_ids, list)
    assert isinstance(node_ids, list)
    assert isinstance(train_node_ids, list)
    assert isinstance(test_node_ids, list)
    positions = {grid_id: index for index, grid_id in enumerate(node_ids)}
    sample_ids = [*positive_ids, *u_ids]
    sample_positions = np.asarray([positions[value] for value in sample_ids], dtype=np.int64)
    test_positions = np.asarray([positions[value] for value in test_node_ids], dtype=np.int64)
    labels = np.concatenate(
        [np.ones(len(positive_ids), dtype=np.float64), np.zeros(len(u_ids), dtype=np.float64)]
    )
    weights = balanced_sample_weights(len(positive_ids), len(u_ids))
    seed_name = model_grid["seeds"]["namespace"].format(
        protocol_id=task.protocol_id,
        experiment_id="E1",
        city=task.city,
        outer_fold=task.outer_fold,
        budget=task.budget,
        trajectory_id=task.trajectory,
        model=task.model,
        config=task.config,
        seed_id=task.seed,
    )
    seed = seed_from_name(seed_name)
    train_cut_edges = np.empty((2, 0), dtype=np.int64)
    evaluation_cut_edges = np.empty((2, 0), dtype=np.int64)
    if task.graph_id != "NA":
        train_cut_edges = _outer_edge_index(fold, task.graph_id, "train", train_node_ids)
        evaluation_cut_edges = _outer_edge_index(fold, task.graph_id, "test", test_node_ids)
    if task.model == "KDE":
        coordinates = grid[["center_x", "center_y"]].to_numpy(dtype=np.float64)
        positive_coordinates = coordinates[sample_positions[: len(positive_ids)]]
        prediction_coordinates = coordinates[test_positions]
        scores = predict_kde(positive_coordinates, prediction_coordinates)
        return scores, fit_audit(
            positive_ids,
            None,
            fit_matrix=positive_coordinates,
            prediction_matrix=prediction_coordinates,
            evaluation_cut_edge_index=evaluation_cut_edges,
            train_cut_edge_index=train_cut_edges,
        )

    scale = task.model in NEURAL_MODELS
    if task.model == "GraphSAGE-G2" and "log1p_partition_local_g2_in_degree" in extra_node_features:
        raw = raw.copy()
        raw["log1p_partition_local_g2_in_degree"] = _partition_local_g2_indegree(fold)
    preprocessor = fit_preprocessor(raw, sample_ids, scale=scale)
    x = preprocessor.transform(raw).to_numpy(dtype=np.float64)
    if task.model == "Neighbor-XGB":
        combined_g1 = np.concatenate(
            [
                _outer_edge_index(fold, "G1", "train", train_node_ids),
                _outer_edge_index(fold, "G1", "test", test_node_ids)
                + np.asarray([[len(train_node_ids)], [len(train_node_ids)]]),
            ],
            axis=1,
        )
        x = neighbor_features(raw.to_numpy(dtype=np.float64), x, combined_g1)
    elif task.model == "Fixed-filter":
        combined_filter_edges = np.concatenate(
            [
                _outer_edge_index(fold, task.graph_id, "train", train_node_ids),
                _outer_edge_index(fold, task.graph_id, "test", test_node_ids)
                + np.asarray([[len(train_node_ids)], [len(train_node_ids)]]),
            ],
            axis=1,
        )
        x = fixed_filter_features(
            x, combined_filter_edges, steps=model_grid.get("fixed_filter_steps", 1)
        )
    params = model_grid["models"][task.model][task.config]
    if task.model in {"XGBoost", "LightGBM", "Neighbor-XGB"}:
        training_x = x[sample_positions]
        prediction_x = x[test_positions]
        scores = predict_tree(
            task.model,
            params,
            training_x,
            labels,
            weights,
            prediction_x,
            seed=seed,
        )
        return scores, fit_audit(
            sample_ids,
            preprocessor,
            fit_matrix=training_x,
            prediction_matrix=prediction_x,
            evaluation_cut_edge_index=evaluation_cut_edges,
            train_cut_edge_index=train_cut_edges,
        )
    if epochs <= 0:
        raise ValueError("outer neural refit requires a positive frozen epoch")
    if task.model == "GraphSAGE-G2":
        train_positions = np.asarray([positions[value] for value in train_node_ids], dtype=np.int64)
        sample_index = {value: index for index, value in enumerate(train_node_ids)}
        train_indices = np.asarray([sample_index[value] for value in sample_ids], dtype=np.int64)
        training_x = x[train_positions]
        training_edges = train_cut_edges
    else:
        training_x = x[sample_positions]
        training_edges = np.empty((2, 0), dtype=np.int64)
        train_indices = np.arange(len(sample_positions), dtype=np.int64)
    result = train_neural(
        task.model,
        params,
        training_x,
        training_edges,
        train_indices,
        labels,
        weights,
        None,
        seed=seed,
        max_epochs=epochs,
        patience=1,
        fixed_epochs=epochs,
        prediction_x=x[test_positions],
        prediction_edge_index=evaluation_cut_edges,
        aggregator=aggregator,
        **({"layers": model_grid["graphsage_layers"]} if "graphsage_layers" in model_grid else {}),
    )
    return result.scores, fit_audit(
        sample_ids,
        preprocessor,
        fit_matrix=training_x,
        prediction_matrix=x[test_positions],
        evaluation_cut_edge_index=evaluation_cut_edges,
        train_cut_edge_index=train_cut_edges,
    )


def _oof_rows(
    task: TaskKey,
    fold: dict[str, object],
    scores: np.ndarray,
    seed_slots: list[str],
) -> pd.DataFrame:
    test_node_ids = fold["test_node_ids"]
    banks = fold["banks"]
    assert isinstance(test_node_ids, list)
    assert isinstance(banks, pd.DataFrame)
    if len(scores) != len(test_node_ids) or not np.isfinite(scores).all():
        raise ValueError("outer prediction is incomplete or non-finite")
    by_id = dict(zip(test_node_ids, scores, strict=True))
    rows = []
    for seed in seed_slots:
        identity = {**task.as_dict(), "seed": seed}
        rows.extend(
            {
                **identity,
                "bank_id": "NA",
                "grid_id": grid_id,
                "sample_role": "NA",
                "score": float(score),
                "status": "success",
            }
            for grid_id, score in zip(test_node_ids, scores, strict=True)
        )
        rows.extend(
            {
                **identity,
                "bank_id": str(int(row.bank_id)),
                "grid_id": str(row.grid_id),
                "sample_role": str(row.sample_role),
                "score": float(by_id[str(row.grid_id)]),
                "status": "success",
            }
            for row in banks.itertuples(index=False)
        )
    from floodgnn_ceus.training import OOF_COLUMNS

    return pd.DataFrame(rows, columns=OOF_COLUMNS)


def run_outer_fit(
    task_root: Path,
    task: TaskKey,
    fold: dict[str, object],
    model_grid: dict[str, object],
    *,
    epochs: int,
) -> str:
    """Run or resume one selected E1 outer fit."""
    status_path = _status_path(task_root, task)
    previous = None
    if status_path.is_file():
        previous = json.loads(status_path.read_text("utf-8")).get("status")
    if task_action(previous) == "skip":
        return str(previous)
    started = time.perf_counter()
    monitor = TaskResourceMonitor()
    try:
        with monitor:
            scores, audit = _predict_outer(fold, task, model_grid, epochs)
            slots = (
                ["NEURAL-1", "NEURAL-2", "NEURAL-3"]
                if task.model not in NEURAL_MODELS
                else [task.seed]
            )
            oof = _oof_rows(task, fold, scores, slots)
            output = status_path.with_name("oof.parquet")
            write_oof(oof, output)
        payload = {
            "fit_audit": audit,
            "oof_sha256": sha256_file(output),
            "status": "success",
            "task": task.as_dict(),
            "wall_seconds": time.perf_counter() - started,
        }
        status = "success"
    except Exception as error:  # noqa: BLE001 - every task failure must become explicit state.
        payload = {
            "error": f"{type(error).__name__}: {error}",
            "status": "failed",
            "task": task.as_dict(),
            "wall_seconds": time.perf_counter() - started,
        }
        status = "failed"
    payload["resources"] = monitor.record()
    _write_json(status_path, payload)
    return status
