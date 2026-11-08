"""Independently select each own/D8 predictor within each attribute condition."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_topology_controls import immutable_json, succeeded

from floodgnn_ceus import e1
from floodgnn_ceus import train as revision
from floodgnn_ceus.experiments import E1OuterUnit
from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.training import TaskKey, outer_refit_epochs, seed_from_name

CONFIG = "configs/independent_attribute_controls_v1.yaml"
FEATURE_LABELS = {"F18": "FULL-COMMON", "F16-noD8": "F16-noD8", "F14-noNeigh": "F14-noNeigh"}


def configuration(root: Path = ROOT) -> dict:
    return yaml.safe_load((root / CONFIG).read_text("utf-8"))


def blocks(config: dict, city: str) -> list[dict]:
    return [
        {
            "city": city,
            "outer_fold": fold,
            "budget": condition["budget"],
            "trajectory": draw,
            "feature_set": features,
            "series": "attributes",
        }
        for condition in config["conditions"]
        for features in condition["features"]
        for fold in config["outer_folds"]
        for draw in (config["trajectories"] if condition["budget"] == "100" else ["NA"])
    ]


def run_path(root: Path, config: dict) -> Path:
    return root / "artifacts/runs" / config["run_id"]


def task_key(
    config: dict, grid: dict, block: dict, variant: dict, candidate: str, seed: str
) -> TaskKey:
    return TaskKey(
        protocol_id=grid["protocol_id"],
        experiment_id=config["run_id"] + ("-inner" if seed == "HPO-1" else ""),
        city=block["city"],
        label_layer="STRICT-LOC",
        reveal_mechanism="uniform",
        outer_fold=str(block["outer_fold"]),
        budget=block["budget"],
        trajectory=block["trajectory"],
        feature_set=FEATURE_LABELS[block["feature_set"]],
        model=variant["model"],
        graph_id=variant["graph"],
        realization_id=variant["id"],
        config=candidate,
        seed=seed,
    )


def subset(fold: dict, members: list[str]) -> dict:
    if len(set(members)) != len(members) or not set(members).issubset(fold["raw"].columns):
        raise ValueError("feature names differ from frozen table")
    return {**fold, "raw": fold["raw"][[x for x in fold["raw"].columns if x in members]]}


def variant_grid(grid: dict, variant: dict) -> dict:
    result = copy.deepcopy(grid)
    result["fixed_filter_steps"] = variant.get("steps", 1)
    result["graphsage_layers"] = variant.get("layers", 2)
    return result


def seed_record(task: TaskKey, grid: dict) -> dict:
    name = grid["seeds"]["namespace"].format(
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
    return {
        "seed_namespace": name,
        "seed_uint128": str(seed_from_name(name)),
        "torch_seed": seed_from_name(name) % (2**63),
    }


def identity(root: Path) -> dict:
    sources = [
        CONFIG,
        "scripts/run_independent_attribute_controls.py",
        "scripts/run_topology_controls.py",
        "configs/feature_sets_v1_2.yaml",
        "configs/model_grid.yaml",
    ]
    sources += [
        f"src/floodgnn_ceus/{x}.py"
        for x in ("e1", "models", "training", "train", "graphs", "evaluation_design")
    ]
    return {
        "source_files": {x: sha256_file(root / x) for x in sources},
        "inputs": revision._input_identity(root),
    }


def check_inputs(root: Path, config: dict) -> None:
    """Fail on missing FULL-specific partitions/reveals/U/banks; never synthesize replacements."""
    grid = yaml.safe_load((root / "configs/model_grid.yaml").read_text("utf-8"))
    for city in config["cities"]:
        data = e1._read_city(root, city)
        checked = set()
        for block in blocks(config, city):
            key = (block["outer_fold"], block["budget"], block["trajectory"])
            if key in checked:
                continue
            task = task_key(config, grid, block, config["variants"][0], "SMALL", "HPO-1")
            for inner in config["inner_folds"]:
                e1._inner_fold_data(data, task, inner)
            e1._outer_fold_data(data, E1OuterUnit(city, *key))
            checked.add(key)


def freeze(root: Path, config: dict) -> None:
    check_inputs(root, config)
    binding = identity(root)
    local = revision.freeze_local_environment(root)
    binding["local_lock"] = local["lock_path"]
    binding["local_lock_sha256"] = sha256_file(root / local["lock_path"])
    immutable_json(run_path(root, config) / "local_environment_lock.json", binding)


def require_lock(root: Path, config: dict, *, probe: bool = True) -> None:
    path = run_path(root, config) / "local_environment_lock.json"
    lock = json.loads(path.read_text("utf-8"))
    for key, value in identity(root).items():
        if lock[key] != value:
            raise ValueError(f"frozen identity changed: {key}")
    if sha256_file(root / lock["local_lock"]) != lock["local_lock_sha256"]:
        raise ValueError("local lock changed")
    if probe:
        revision.require_revision_environment(root, revision.prepare_plan(root))
        receipt = json.loads((run_path(root, config) / "resource_probe.json").read_text("utf-8"))
        if receipt["attribute_lock_sha256"] != sha256_file(path):
            raise ValueError("attribute probe binding changed")


def run_block(
    root: Path, config: dict, data: dict, grid: dict, block: dict, members: list[str]
) -> None:
    destination = run_path(root, config)
    unit = E1OuterUnit(block["city"], block["outer_fold"], block["budget"], block["trajectory"])
    first = task_key(config, grid, block, config["variants"][0], "SMALL", "HPO-1")
    prepared = [subset(e1._inner_fold_data(data, first, i), members) for i in config["inner_folds"]]
    choices = {}
    for variant in config["variants"]:
        model_grid = variant_grid(grid, variant)
        tasks = [task_key(config, grid, block, variant, c, "HPO-1") for c in config["candidates"]]
        for task in tasks:
            path = e1._status_path(destination / "inner", task)
            if not succeeded(path, "inner_scores.parquet", "result_sha256"):
                status = e1.run_inner_task(
                    root, destination / "inner", task, data, model_grid, prepared_folds=prepared
                )
                if status != "success":
                    raise RuntimeError(f"inner fit failed: {path}")
            immutable_json(path.with_name("training_seed.json"), seed_record(task, grid))
        selected = e1.select_outer_models(
            root,
            destination / "inner",
            unit,
            task_keys=tasks,
            models=(variant["model"],),
            alternative_order=(variant["model"],),
        )["models"][variant["model"]]
        choices[variant["id"]] = {
            **selected,
            **variant,
            "refit_epochs": outer_refit_epochs(selected["best_epochs"]),
        }
    name = (
        f"attributes-{block['city']}-f{block['outer_fold']}-b{block['budget']}-"
        f"t{block['trajectory']}-{block['feature_set']}"
    )
    immutable_json(
        destination / "selections" / f"{name}.json",
        {"block": block, "variants": choices, "selection_source": "inner sampled AP only"},
    )
    outer = subset(e1._outer_fold_data(data, unit), members)
    for variant in config["variants"]:
        choice = choices[variant["id"]]
        for seed in config["outer_seeds"]:
            task = task_key(config, grid, block, variant, choice["config"], seed)
            path = e1._status_path(destination / "outer", task)
            if not succeeded(path, "oof.parquet", "oof_sha256"):
                status = e1.run_outer_fit(
                    destination / "outer",
                    task,
                    outer,
                    variant_grid(grid, variant),
                    epochs=choice["refit_epochs"],
                )
                if status != "success":
                    raise RuntimeError(f"outer fit failed: {path}")
            immutable_json(
                path.with_name("training_metadata.json"),
                {**seed_record(task, grid), "selected_epochs": choice["refit_epochs"]},
            )


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check-inputs", "freeze", "probe", "train"))
    parser.add_argument("--city", choices=("zhengzhou", "beijing"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()
    config = configuration()
    if args.action == "plan":
        count = sum(len(blocks(config, c)) for c in config["cities"])
        print(
            json.dumps(
                {
                    "blocks": count,
                    "independent_selections": count * 4,
                    "inner_fold_fits": count * 4 * 2 * 3,
                    "outer_fits": count * 4 * 3,
                    "numerical_replication_verified": False,
                }
            )
        )
    elif args.action == "check-inputs":
        check_inputs(ROOT, config)
    elif args.action == "freeze":
        freeze(ROOT, config)
    elif args.action == "probe":
        require_lock(ROOT, config, probe=False)
        receipt = revision.run_local_probe(ROOT)
        receipt["attribute_lock_sha256"] = sha256_file(
            run_path(ROOT, config) / "local_environment_lock.json"
        )
        immutable_json(run_path(ROOT, config) / "resource_probe.json", receipt)
    else:
        require_lock(ROOT, config)
        import torch

        torch.set_num_threads(config["cpu_threads"])
        grid = yaml.safe_load((ROOT / "configs/model_grid.yaml").read_text("utf-8"))
        sets = yaml.safe_load((ROOT / "configs/feature_sets_v1_2.yaml").read_text("utf-8"))["sets"]
        for city in [args.city] if args.city else config["cities"]:
            data = e1._read_city(ROOT, city)
            for block in blocks(config, city)[args.start : args.start + args.limit]:
                run_block(ROOT, config, data, grid, block, sets[block["feature_set"]]["members"])
                print(json.dumps({"stage": "complete", **block}), flush=True)


if __name__ == "__main__":
    main()
