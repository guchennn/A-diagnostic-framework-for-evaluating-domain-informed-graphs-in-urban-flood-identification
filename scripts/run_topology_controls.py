"""Train independently selected topology controls on the existing frozen design."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from floodgnn_ceus import e1
from floodgnn_ceus import train as revision
from floodgnn_ceus.experiments import E1OuterUnit
from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.training import TaskKey, outer_refit_epochs, seed_from_name

CONFIG = "configs/topology_controls_v1.yaml"


def configuration(root: Path = ROOT) -> dict:
    return yaml.safe_load((root / CONFIG).read_text("utf-8"))


def run_path(root: Path, config: dict) -> Path:
    return root / "artifacts/runs" / config["run_id"]


def immutable_json(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text("utf-8")) != value:
            raise ValueError(f"refusing to replace frozen metadata: {path}")
    else:
        e1._write_json(path, value)


def task_key(
    config: dict,
    grid: dict,
    city: str,
    fold: int,
    draw: str,
    variant: dict,
    realization: int,
    candidate: str,
    seed: str,
) -> TaskKey:
    return TaskKey(
        protocol_id=grid["protocol_id"],
        experiment_id=config["run_id"] + ("-inner" if seed == "HPO-1" else ""),
        city=city,
        label_layer="STRICT-LOC",
        reveal_mechanism="uniform",
        outer_fold=str(fold),
        budget="100",
        trajectory=str(draw),
        feature_set="FULL-COMMON",
        model=variant["model"],
        graph_id="G7",
        realization_id=f"{variant['id']}-r{realization}",
        config=candidate,
        seed=seed,
    )


def cache_key(
    city: str, fold: int, draw: str, inner: int | None, partition: str, realization: int
) -> str:
    scope = "outer" if inner is None else f"inner-t{draw}-i{inner}"
    return f"{city}/f{fold}/{scope}/{partition}-r{realization}"


def graph_invariants(
    original: pd.DataFrame, generated: pd.DataFrame, geometry: pd.DataFrame, config: dict
) -> dict:
    columns = ["source_grid_id", "target_grid_id"]
    a = original[columns].astype(str)
    b = generated[columns].astype(str)
    if len(a) != len(b) or b.duplicated().any():
        raise ValueError("edge count or uniqueness differs")
    for column in columns:
        if Counter(a[column]) != Counter(b[column]):
            raise ValueError(f"degree differs: {column}")
    places = geometry.set_index("grid_id")[["row", "column"]]
    pairs = list(b.itertuples(index=False, name=None))
    if not set(b[columns].to_numpy().ravel()).issubset(places.index):
        raise ValueError("partition leakage")
    delta = np.abs(
        places.loc[b.source_grid_id].to_numpy() - places.loc[b.target_grid_id].to_numpy()
    )
    if len(delta) and not np.all(delta.max(axis=1) == 1):
        raise ValueError("self-loop or edge outside eight neighbours")
    before_delta = np.abs(
        places.loc[a.source_grid_id].to_numpy() - places.loc[a.target_grid_id].to_numpy()
    )
    distance_counts_match = int(np.all(delta == 1, axis=1).sum()) == int(
        np.all(before_delta == 1, axis=1).sum()
    )
    if config.get("preserve_global_distance_counts", True) and not distance_counts_match:
        raise ValueError("rook/diagonal counts differ")
    original_set = set(a.itertuples(index=False, name=None))
    changed = 1 - len(original_set.intersection(pairs)) / len(a) if len(a) else 0.0
    if (
        len(a) >= config.get("minimum_edges_for_change_check", 1)
        and changed < config["min_changed_fraction"]
    ):
        raise ValueError(f"insufficient changed edges: {changed}")
    return {
        "edge_count": len(a),
        "changed_fraction": changed,
        "exact_in_out_degree": True,
        "same_partition_eight_neighbours": True,
        "distance_counts_match": distance_counts_match,
    }


def partition_graph(
    root: Path,
    config: dict,
    original: pd.DataFrame,
    geometry: pd.DataFrame,
    key: str,
    *,
    create: bool,
) -> pd.DataFrame:
    base = run_path(root, config) / "graphs" / key
    data_path, audit_path = base.with_suffix(".parquet"), base.with_suffix(".json")
    fingerprints = {
        "graph_config_sha256": hashlib.sha256(
            json.dumps(config["graph"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "input_edges_sha256": hashlib.sha256(
            original[["source_grid_id", "target_grid_id"]]
            .sort_values(["source_grid_id", "target_grid_id"])
            .to_csv(index=False)
            .encode("utf-8")
        ).hexdigest(),
        "input_geometry_sha256": hashlib.sha256(
            geometry[["grid_id", "row", "column"]]
            .sort_values("grid_id")
            .to_csv(index=False)
            .encode("utf-8")
        ).hexdigest(),
    }
    if audit_path.exists():
        audit = json.loads(audit_path.read_text("utf-8"))
        if audit.get("accepted") is not True or any(
            audit.get(k) != v for k, v in fingerprints.items()
        ):
            raise ValueError(f"cached graph inputs or acceptance differ: {key}")
        if sha256_file(data_path) != audit["parquet_sha256"]:
            raise ValueError(f"cached graph hash differs: {key}")
        return pd.read_parquet(data_path)
    if not create:
        raise FileNotFoundError(f"generate and freeze graph first: {key}")
    from floodgnn_ceus.topology_controls import build_control

    graph, audit = build_control(
        original[["source_grid_id", "target_grid_id"]].copy(),
        geometry[["grid_id", "row", "column"]].copy(),
        seed_name=f"{config['run_id']}/{key}",
        config=config["graph"],
    )
    checks = graph_invariants(original, graph, geometry, config["graph"])
    if audit.get("accepted") is not True:
        raise ValueError(f"graph builder rejected {key}: {audit}")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        raise ValueError(f"unreceipted graph exists: {key}")
    graph.to_parquet(data_path, index=False)
    immutable_json(
        audit_path,
        {**audit, **checks, **fingerprints, "key": key, "parquet_sha256": sha256_file(data_path)},
    )
    return graph


def inject_graphs(
    root: Path,
    config: dict,
    fold: dict,
    city: str,
    outer: int,
    draw: str,
    inner: int | None,
    realization: int,
    *,
    create: bool,
) -> dict:
    field = "edges" if inner is None else "cut_edges"
    edges = fold[field]
    original = edges.loc[edges.graph_id.eq("G2")]
    if inner is None:
        original = original.loc[original.realization_id.eq("NA")]
    additions = []
    for partition, members in (
        ("train", fold["train_node_ids"]),
        ("test", fold["test_node_ids"] if inner is None else fold["validation_node_ids"]),
    ):
        part = original.loc[original.partition.eq(partition)]
        geometry = fold["grid"].loc[members, ["row", "column"]].reset_index()
        key = cache_key(city, outer, draw, inner, partition, realization)
        graph = partition_graph(root, config, part, geometry, key, create=create).copy()
        graph["graph_id"], graph["partition"] = "G7", partition
        if inner is None:
            graph["realization_id"], graph["outer_fold"] = "NA", outer
        additions.append(graph)
    return {**fold, field: pd.concat([edges, *additions], ignore_index=True)}


def units(config: dict, city: str):
    for fold in config["outer_folds"]:
        for draw in config["trajectories"]:
            for realization in config["graph_realizations"]:
                yield fold, str(draw), realization


def cache_graphs(root: Path, config: dict, city: str, start: int = 0, limit: int = 100000) -> None:
    data = e1._read_city(root, city)
    grid = yaml.safe_load((root / "configs/model_grid.yaml").read_text("utf-8"))
    for fold, draw, realization in list(units(config, city))[start : start + limit]:
        task = task_key(
            config, grid, city, fold, draw, config["variants"][0], realization, "SMALL", "HPO-1"
        )
        for inner in config["inner_folds"]:
            inject_graphs(
                root,
                config,
                e1._inner_fold_data(data, task, inner),
                city,
                fold,
                draw,
                inner,
                realization,
                create=True,
            )
        inject_graphs(
            root,
            config,
            e1._outer_fold_data(data, E1OuterUnit(city, fold, "100", draw)),
            city,
            fold,
            draw,
            None,
            realization,
            create=True,
        )
        print(
            json.dumps(
                {
                    "stage": "cache",
                    "city": city,
                    "fold": fold,
                    "draw": draw,
                    "realization": realization,
                }
            ),
            flush=True,
        )


def identity(root: Path, config: dict) -> dict:
    sources = [
        CONFIG,
        "scripts/run_topology_controls.py",
        "src/floodgnn_ceus/topology_controls.py",
        "configs/model_grid.yaml",
    ]
    sources += [f"src/floodgnn_ceus/{name}.py" for name in ("e1", "models", "training", "train")]
    graphs = run_path(root, config) / "graphs"
    paths = sorted(graphs.rglob("*.json")) + sorted(graphs.rglob("*.parquet"))
    expected = (
        len(config["cities"])
        * len(config["outer_folds"])
        * len(config["graph_realizations"])
        * (2 + 2 * len(config["trajectories"]) * len(config["inner_folds"]))
    )
    if len(paths) != expected * 2:
        raise ValueError(f"incomplete graph cache: {len(paths)} files; expected {expected * 2}")
    return {
        "source_files": {p: sha256_file(root / p) for p in sources},
        "graph_files": {p.relative_to(root).as_posix(): sha256_file(p) for p in paths},
        "protocol_lock_sha256": sha256_file(root / "artifacts/protocol_lock.json"),
        "construction_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "pyarrow")
        },
    }


def freeze(root: Path, config: dict) -> None:
    binding = identity(root, config)
    local = revision.freeze_local_environment(root)
    binding["local_revision_lock"] = Path(local["lock_path"]).as_posix()
    binding["local_revision_lock_sha256"] = sha256_file(root / binding["local_revision_lock"])
    immutable_json(run_path(root, config) / "local_environment_lock.json", binding)


def require_lock(root: Path, config: dict, *, probe: bool = True) -> dict:
    path = run_path(root, config) / "local_environment_lock.json"
    lock = json.loads(path.read_text("utf-8"))
    for key, value in identity(root, config).items():
        if lock[key] != value:
            raise ValueError(f"frozen identity changed: {key}")
    if sha256_file(root / lock["local_revision_lock"]) != lock["local_revision_lock_sha256"]:
        raise ValueError("local revision lock changed")
    if probe:
        revision.require_revision_environment(root, revision.prepare_plan(root))
        receipt = json.loads((run_path(root, config) / "resource_probe.json").read_text("utf-8"))
        if receipt["topology_lock_sha256"] != sha256_file(path):
            raise ValueError("topology probe binding changed")
    return lock


def succeeded(path: Path, payload_name: str, hash_key: str) -> bool:
    if not path.exists():
        return False
    receipt = json.loads(path.read_text("utf-8"))
    if receipt["status"] != "success":
        return False
    if sha256_file(path.with_name(payload_name)) != receipt[hash_key]:
        raise ValueError(f"successful receipt payload changed: {path}")
    return True


def train(root: Path, config: dict, city: str, start: int, limit: int) -> None:
    require_lock(root, config)
    import torch

    torch.set_num_threads(config["cpu_threads"])
    data = e1._read_city(root, city)
    grid = yaml.safe_load((root / "configs/model_grid.yaml").read_text("utf-8"))
    destination = run_path(root, config)
    for index, (fold, draw, realization) in enumerate(
        list(units(config, city))[start : start + limit], start=start
    ):
        unit = E1OuterUnit(city, fold, "100", draw)
        first = task_key(
            config, grid, city, fold, draw, config["variants"][0], realization, "SMALL", "HPO-1"
        )
        prepared = [
            inject_graphs(
                root,
                config,
                e1._inner_fold_data(data, first, inner),
                city,
                fold,
                draw,
                inner,
                realization,
                create=False,
            )
            for inner in config["inner_folds"]
        ]
        outer = inject_graphs(
            root,
            config,
            e1._outer_fold_data(data, unit),
            city,
            fold,
            draw,
            None,
            realization,
            create=False,
        )
        choices = {}
        for variant in config["variants"]:
            model_grid = copy.deepcopy(grid)
            model_grid["fixed_filter_steps"] = variant.get("steps", 1)
            model_grid["graphsage_layers"] = variant.get("layers", 2)
            tasks = [
                task_key(config, grid, city, fold, draw, variant, realization, c, "HPO-1")
                for c in config["candidates"]
            ]
            for task in tasks:
                path = e1._status_path(destination / "inner", task)
                if not succeeded(path, "inner_scores.parquet", "result_sha256"):
                    status = e1.run_inner_task(
                        root, destination / "inner", task, data, model_grid, prepared_folds=prepared
                    )
                    if status != "success":
                        raise RuntimeError(f"inner fit failed: {path}")
            selected = e1.select_outer_models(
                root,
                destination / "inner",
                unit,
                task_keys=tasks,
                models=(variant["model"],),
                alternative_order=(variant["model"],),
            )["models"][variant["model"]]
            epochs = outer_refit_epochs(selected["best_epochs"])
            choices[variant["id"]] = {**selected, "refit_epochs": epochs}
            selection_path = (
                destination
                / "selections"
                / (f"{city}-f{fold}-t{draw}-r{realization}-{variant['id']}.json")
            )
            immutable_json(selection_path, choices[variant["id"]])
            for seed in config["outer_seeds"]:
                task = task_key(
                    config, grid, city, fold, draw, variant, realization, selected["config"], seed
                )
                path = e1._status_path(destination / "outer", task)
                if not succeeded(path, "oof.parquet", "oof_sha256"):
                    status = e1.run_outer_fit(
                        destination / "outer", task, outer, model_grid, epochs=epochs
                    )
                    if status != "success":
                        raise RuntimeError(f"outer fit failed: {path}")
                name = grid["seeds"]["namespace"].format(
                    protocol_id=task.protocol_id,
                    experiment_id="E1",
                    city=city,
                    outer_fold=task.outer_fold,
                    budget="100",
                    trajectory_id=draw,
                    model=task.model,
                    config=task.config,
                    seed_id=seed,
                )
                immutable_json(
                    path.with_name("training_metadata.json"),
                    {
                        "selected_epochs": epochs,
                        "seed_namespace": name,
                        "seed_uint128": str(seed_from_name(name)),
                        "torch_seed": seed_from_name(name) % (2**63),
                        "source": "locked e1 optimizer seed rule",
                    },
                )
        print(
            json.dumps(
                {
                    "stage": "train",
                    "city": city,
                    "unit": index,
                    "fold": fold,
                    "draw": draw,
                    "realization": realization,
                }
            ),
            flush=True,
        )


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cache", "freeze", "probe", "train", "status"))
    parser.add_argument("--city", choices=("zhengzhou", "beijing"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()
    config = configuration()
    if args.action == "cache":
        for city in [args.city] if args.city else config["cities"]:
            cache_graphs(ROOT, config, city, args.start, args.limit)
    elif args.action == "freeze":
        freeze(ROOT, config)
    elif args.action == "probe":
        require_lock(ROOT, config, probe=False)
        receipt = revision.run_local_probe(ROOT)
        receipt["topology_lock_sha256"] = sha256_file(
            run_path(ROOT, config) / "local_environment_lock.json"
        )
        immutable_json(run_path(ROOT, config) / "resource_probe.json", receipt)
        print(json.dumps(receipt))
    elif args.action == "train":
        for city in [args.city] if args.city else config["cities"]:
            train(ROOT, config, city, args.start, args.limit)
    else:
        print(
            json.dumps(
                {
                    "run_id": config["run_id"],
                    "units_per_city": len(list(units(config, config["cities"][0]))),
                    "cached_graphs": len(
                        list((run_path(ROOT, config) / "graphs").rglob("*.parquet"))
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
