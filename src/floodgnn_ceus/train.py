"""Post-review matched controls; local CUDA execution requires a fresh bound probe."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from floodgnn_ceus import e1
from floodgnn_ceus.experiments import NEURAL_MODELS, E1OuterUnit
from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file
from floodgnn_ceus.models import train_neural
from floodgnn_ceus.training import (
    PROBE_CELL,
    TaskKey,
    balanced_sample_weights,
    fit_preprocessor,
    outer_refit_epochs,
    seed_from_name,
)

CONFIG = "configs/matched_models.yaml"


def _config(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text("utf-8"))
    if config["preregistered"] is not False or config["budget"] != "100":
        raise ValueError("revision must retain its post-review budget-100 scope")
    return config


def _source_identity(root: Path) -> dict[str, str]:
    paths = [root / "pyproject.toml", root / "uv.lock"]
    paths.extend(sorted((root / "configs").rglob("*.yaml")))
    paths.extend(sorted((root / "src/floodgnn_ceus").glob("*.py")))
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def prepare_plan(root: Path) -> dict:
    """Describe the complete matrix without importing torch or reading labels."""
    config = _config(root)
    units = len(config["cities"]) * len(config["outer_folds"]) * len(config["trajectories"])
    variants = config["variants"]
    candidates = sum(1 if v["model"] == "KDE" else len(config["configs"]) for v in variants)
    refits = sum(len(config["outer_seeds"]) if v["model"] in NEURAL_MODELS else 1 for v in variants)
    source = _source_identity(root)
    digest = sha256_bytes(canonical_json_bytes(source))
    return {
        "revision_id": config["revision_id"],
        "run_namespace": f"{config['revision_id']}-{digest[:12]}",
        "source_sha256": digest,
        "source_files": source,
        "preregistered": False,
        "labels_read": False,
        "performance_blind": True,
        "outer_units": units,
        "inner_candidate_tasks": units * candidates,
        "inner_fold_fits": units * candidates * len(config["inner_folds"]),
        "outer_refits": units * refits,
        "variants": variants,
        "alternative_order": config["alternative_order"],
        "selection": "SMALL/MEDIUM independently within each variant; inner folds only",
        "tree_seed_policy": "one deterministic fit, three identical paired output slots",
    }


def _run_root(root: Path, plan: dict) -> Path:
    return root / "artifacts/runs" / plan["run_namespace"]


def _input_identity(root: Path) -> list[dict]:
    """Bind local primary inputs, including every file consumed by ``e1._read_city``.

    Source provenance is bound by the immutable protocol lock. The primary runner
    does not read source manifests, remote launch scripts, or sensitivity inputs.
    """
    paths = [root / "artifacts/protocol_lock.json"]
    paths.extend(root / name for name in _source_identity(root))
    city_files = (
        "final_scorable_grid.parquet",
        "features_18.parquet",
        "evaluation_design/outer_partitions.parquet",
        "evaluation_design/inner_blocks.parquet",
        "evaluation_design/reveals.parquet",
        "evaluation_design/u_samples.parquet",
        "evaluation_design/evaluation_banks.parquet",
        "graphs/base_edges.parquet",
        "graphs/outer_edges.parquet",
    )
    for city in _config(root)["cities"]:
        paths.extend(root / "data/interim" / city / name for name in city_files)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(paths))
    ]


def _environment() -> dict:
    import torch

    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torch-geometric", "xgboost", "lightgbm", "scikit-learn")
    }
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError("revision requires Python 3.12")
    if versions["torch"].split("+")[0] != "2.8.0" or versions["torch-geometric"] != "2.7.0":
        raise ValueError("revision requires locked torch 2.8.0 / PyG 2.7.0")
    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        raise ValueError("revision requires CUDA 12.8 torch with available local GPU")
    return {
        "python": platform.python_version(),
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }


def freeze_local_environment(root: Path) -> dict:
    """Bind an independent local lock to allowlisted inputs and current revision code."""
    plan = prepare_plan(root)
    environment = _environment()
    files = _input_identity(root)
    lock = {
        "schema_version": 1,
        "execution_authority": "user-requested-local-CUDA",
        "source_sha256": plan["source_sha256"],
        "files": files,
        "input_sha256": sha256_bytes(canonical_json_bytes(files)),
        "protocol_lock_sha256": sha256_file(root / "artifacts/protocol_lock.json"),
        "environment": environment,
    }
    destination = _run_root(root, plan) / "local_environment_lock.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(canonical_json_bytes(lock))
    except FileExistsError:
        if json.loads(destination.read_text("utf-8")) != lock:
            raise ValueError("existing local revision lock differs; do not replace it") from None
    return {"status": "success", "lock_path": destination.relative_to(root).as_posix(), **lock}


def _require_local_lock(root: Path, plan: dict) -> dict:
    path = _run_root(root, plan) / "local_environment_lock.json"
    lock = json.loads(path.read_text("utf-8"))
    if lock["source_sha256"] != plan["source_sha256"]:
        raise ValueError("local lock source differs")
    files = _input_identity(root)
    if lock["input_sha256"] != sha256_bytes(canonical_json_bytes(files)):
        raise ValueError("local lock inputs differ")
    if lock["environment"] != _environment():
        raise ValueError("local CUDA environment differs from revision lock")
    return lock


def _task(config: dict, unit: E1OuterUnit, variant: dict, candidate: str, seed: str) -> TaskKey:
    return TaskKey(
        protocol_id=config["protocol_id"],
        experiment_id="reviewer-revision-v1-inner" if seed == "HPO-1" else "reviewer-revision-v1",
        city=unit.city,
        label_layer="STRICT-LOC",
        reveal_mechanism="uniform",
        outer_fold=str(unit.outer_fold),
        budget=unit.budget,
        trajectory=unit.trajectory,
        feature_set="FULL-COMMON",
        model=variant["model"],
        graph_id=variant["graph"],
        realization_id=variant["id"],
        config=candidate,
        seed=seed,
    )


def run_local_probe(root: Path) -> dict:
    """Run the fixed resource probe first; discard predictions without computing metrics."""
    plan = prepare_plan(root)
    lock = _require_local_lock(root, plan)
    import torch

    torch.set_num_threads(int(_config(root)["cpu_threads"]))
    grid = yaml.safe_load((root / "configs/model_grid.yaml").read_text("utf-8"))
    unit = E1OuterUnit("zhengzhou", 0, "20", "1")
    variant = {"id": "resource-probe", "model": "GraphSAGE-G2", "graph": "G2"}
    task = _task(grid, unit, variant, "SMALL", "HPO-1")
    fold = e1._inner_fold_data(e1._read_city(root, unit.city), task, 0)
    train_ids = fold["train_node_ids"]
    sample_ids = fold["positive_ids"] + fold["u_ids"]
    raw = fold["raw"].loc[train_ids]
    x = fit_preprocessor(raw, sample_ids, scale=True).transform(raw).to_numpy(dtype=float)
    positions = {value: i for i, value in enumerate(train_ids)}
    positive_count = len(fold["positive_ids"])
    u_count = len(fold["u_ids"])
    edges = e1._partition_edge_index(fold, "G2", "train", train_ids)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = train_neural(
        "GraphSAGE-G2",
        grid["models"]["GraphSAGE-G2"]["SMALL"],
        x,
        edges,
        np.array([positions[value] for value in sample_ids], dtype=np.int64),
        np.r_[np.ones(positive_count), np.zeros(u_count)],
        balanced_sample_weights(positive_count, u_count),
        None,
        seed=seed_from_name("FloodGNN-CEUS/resource-probe"),
        max_epochs=1,
        patience=1,
        fixed_epochs=1,
    )
    torch.cuda.synchronize()
    receipt = {
        "status": "success",
        "cell": PROBE_CELL,
        "epochs": result.best_epoch,
        "source_sha256": plan["source_sha256"],
        "input_sha256": lock["input_sha256"],
        "lock_sha256": sha256_file(_run_root(root, plan) / "local_environment_lock.json"),
        "performance_blind": True,
        "predictions_written": False,
        "metrics_computed": False,
        "nodes": len(train_ids),
        "edges": int(edges.shape[1]),
        "feature_dim": x.shape[1],
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_bytes": result.checkpoint_bytes,
        "wall_seconds": time.perf_counter() - started,
    }
    del result
    e1._write_json(_run_root(root, plan) / "resource_probe.json", receipt)
    return receipt


def require_revision_environment(root: Path, plan: dict) -> None:
    lock = _require_local_lock(root, plan)
    run_root = _run_root(root, plan)
    probe = json.loads((run_root / "resource_probe.json").read_text("utf-8"))
    if (
        probe.get("status") != "success"
        or probe.get("epochs") != 1
        or probe.get("cell") != PROBE_CELL
        or probe.get("performance_blind") is not True
        or probe.get("predictions_written") is not False
        or probe.get("metrics_computed") is not False
        or probe.get("source_sha256") != plan["source_sha256"]
        or probe.get("input_sha256") != lock["input_sha256"]
        or probe.get("lock_sha256") != sha256_file(run_root / "local_environment_lock.json")
    ):
        raise ValueError("missing or stale performance-blind revision probe")


def preflight(root: Path) -> dict:
    plan = prepare_plan(root)
    missing = []
    try:
        require_revision_environment(root, plan)
    except (FileNotFoundError, ValueError, KeyError, ImportError) as error:
        missing.append(f"{type(error).__name__}: {error}")
    return {**plan, "status": "blocked" if missing else "ready", "missing_gates": missing}


def _variant_grid(model_grid: dict, variant: dict, revision: dict) -> dict:
    grid = copy.deepcopy(model_grid)
    grid["fixed_filter_steps"] = variant.get("steps", 1)
    grid["graphsage_layers"] = variant.get("layers", 2)
    # Keep the frozen model seed pairing; each variant is distinguished in its task identity.
    if variant["model"] in ("XGBoost", "Neighbor-XGB"):
        for candidate in revision["configs"]:
            grid["models"][variant["model"]][candidate].update(
                {
                    key: value
                    for key, value in revision["xgboost_fixed"].items()
                    if key != "min_child_weight"
                }
            )
    return grid


def _units(revision: dict) -> list[E1OuterUnit]:
    return [
        E1OuterUnit(city, fold, revision["budget"], trajectory)
        for city in revision["cities"]
        for fold in revision["outer_folds"]
        for trajectory in revision["trajectories"]
    ]


def _candidates(revision: dict, variant: dict) -> list[str]:
    return ["NA"] if variant["model"] == "KDE" else revision["configs"]


def _check_success(task_root: Path, task: TaskKey) -> None:
    status_path = e1._status_path(task_root, task)
    status = json.loads(status_path.read_text("utf-8"))
    if status["status"] != "success":
        raise ValueError(f"required revision candidate is not successful: {task.relative_path()}")
    result = status_path.with_name("inner_scores.parquet")
    if sha256_file(result) != status["result_sha256"]:
        raise ValueError("revision inner result hash differs")


def select_revision_unit(root: Path, run_root: Path, unit: E1OuterUnit) -> dict:
    """Select each control separately; update A* within the original six families only."""
    revision = _config(root)
    model_grid = yaml.safe_load((root / "configs/model_grid.yaml").read_text("utf-8"))
    choices = {}
    for variant in revision["variants"]:
        tasks = [
            _task(model_grid, unit, variant, c, "HPO-1") for c in _candidates(revision, variant)
        ]
        for task in tasks:
            _check_success(run_root / "inner", task)
        selected = e1.select_outer_models(
            root,
            run_root / "inner",
            unit,
            task_keys=tasks,
            models=(variant["model"],),
            alternative_order=(variant["model"],),
        )
        choice = selected["models"][variant["model"]]
        choices[variant["id"]] = {
            **choice,
            **variant,
            "refit_epochs": (
                outer_refit_epochs(choice["best_epochs"])
                if variant["model"] in NEURAL_MODELS
                else 0
            ),
        }
    alternative_order = revision["alternative_order"]
    best = max(alternative_order, key=lambda value: choices[value]["score"])
    return {
        "unit": unit.__dict__,
        "variants": choices,
        "best_alternative": choices[best],
        "alternative_order": alternative_order,
        "selection_source": "three inner-fold mean PB-AUPRC, no outer scores",
    }


def progress(root: Path) -> dict:
    plan = prepare_plan(root)
    run_root = _run_root(root, plan)
    counts = {}
    for phase in ("inner", "outer"):
        statuses = {"success": 0, "failed": 0}
        for path in (run_root / phase / "tasks").rglob("status.json"):
            status = json.loads(path.read_text("utf-8"))["status"]
            statuses[status] = statuses.get(status, 0) + 1
        counts[phase] = statuses
    return {
        "run_root": str(run_root),
        "counts": counts,
        "expected": {"inner": plan["inner_candidate_tasks"], "outer": plan["outer_refits"]},
    }


def run_batch(root: Path, *, phase: str = "all", start: int = 0, limit: int = 50) -> dict:
    """Run serial GPU fits with per-task atomic resume and fold reuse within each unit."""
    if phase not in {"all", "inner", "outer"} or start < 0 or limit <= 0:
        raise ValueError("invalid revision batch phase/start/limit")
    plan = prepare_plan(root)
    require_revision_environment(root, plan)
    import torch

    revision = _config(root)
    torch.set_num_threads(int(revision["cpu_threads"]))
    model_grid = yaml.safe_load((root / "configs/model_grid.yaml").read_text("utf-8"))
    run_root = _run_root(root, plan)
    e1._write_json(run_root / "plan.json", plan)
    cache = {}
    completed = 0
    for unit in _units(revision)[start : start + limit]:
        if unit.city not in cache:
            cache.clear()
            cache[unit.city] = e1._read_city(root, unit.city)
        data = cache[unit.city]
        prepared_folds = None
        if phase in {"all", "inner"}:
            for variant in revision["variants"]:
                grid = _variant_grid(model_grid, variant, revision)
                for candidate in _candidates(revision, variant):
                    task = _task(grid, unit, variant, candidate, "HPO-1")
                    status_path = e1._status_path(run_root / "inner", task)
                    if status_path.exists():
                        status = json.loads(status_path.read_text("utf-8"))
                        if status["status"] == "success":
                            _check_success(run_root / "inner", task)
                            continue
                    if prepared_folds is None:
                        prepared_folds = [e1._inner_fold_data(data, task, i) for i in range(3)]
                    status = e1.run_inner_task(
                        root,
                        run_root / "inner",
                        task,
                        data,
                        grid,
                        prepared_folds=prepared_folds,
                    )
                    _record_progress(run_root, unit, variant, phase="inner", status=status)
                    if status != "success":
                        raise RuntimeError(f"inner fit failed; see {status_path}")
        if phase in {"all", "outer"}:
            selection = select_revision_unit(root, run_root, unit)
            name = f"{unit.city}-f{unit.outer_fold}-b{unit.budget}-t{unit.trajectory}"
            e1._write_json(run_root / "selections" / f"{name}.json", selection)
            fold = e1._outer_fold_data(data, unit)
            for variant in revision["variants"]:
                choice = selection["variants"][variant["id"]]
                neural = variant["model"] in NEURAL_MODELS
                seeds = revision["outer_seeds"] if neural else [revision["outer_seeds"][0]]
                epochs = outer_refit_epochs(choice["best_epochs"]) if neural else 0
                grid = _variant_grid(model_grid, variant, revision)
                for seed in seeds:
                    task = _task(grid, unit, variant, choice["config"], seed)
                    status = e1.run_outer_fit(run_root / "outer", task, fold, grid, epochs=epochs)
                    status_path = e1._status_path(run_root / "outer", task)
                    receipt = json.loads(status_path.read_text("utf-8"))
                    if status == "success" and receipt["oof_sha256"] != sha256_file(
                        status_path.with_name("oof.parquet")
                    ):
                        raise ValueError("revision outer result hash differs")
                    _record_progress(run_root, unit, variant, phase="outer", status=status)
                    if status != "success":
                        raise RuntimeError(f"outer fit failed; see {status_path}")
        completed += 1
    result = {
        "status": "success",
        "completed_units": completed,
        "phase": phase,
        "start": start,
        "end": start + completed,
        **progress(root),
    }
    e1._write_json(run_root / "batch_summary.json", result)
    return result


def _record_progress(run_root: Path, unit: E1OuterUnit, variant: dict, *, phase: str, status: str):
    record = {
        "unit": unit.__dict__,
        "variant": variant["id"],
        "phase": phase,
        "status": status,
        "timestamp": time.time(),
    }
    e1._write_json(run_root / "latest_progress.json", record)
    print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("plan", "preflight", "freeze-local", "probe-local", "run", "progress")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("all", "inner", "outer"), default="all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    root = args.root.resolve()
    # Required by deterministic CUDA matmul, before torch initializes CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    functions = {
        "plan": prepare_plan,
        "preflight": preflight,
        "progress": progress,
        "freeze-local": freeze_local_environment,
        "probe-local": run_local_probe,
    }
    result = (
        run_batch(root, phase=args.phase, start=args.start, limit=args.limit)
        if args.action == "run"
        else functions[args.action](root)
    )
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        e1._write_json(output, result)
    print(json.dumps(result, indent=2))
    if args.action == "preflight" and result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
