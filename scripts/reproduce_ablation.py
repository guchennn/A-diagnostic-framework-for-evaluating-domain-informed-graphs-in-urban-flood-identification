"""Replay FINAL-v8 recorded ablation selections using the released E1 fit helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from floodgnn_ceus import e1, reviewer_revision
from floodgnn_ceus.experiments import E1OuterUnit
from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.protocol import EXPECTED_FEATURE_NAMES
from floodgnn_ceus.training import TaskKey, outer_refit_epochs, seed_from_name

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "FloodGNN-CEUS-ZZ-BJ-v1.0"
SETS = {"full": "F18", "noD8": "F16-noD8", "noNeigh": "F14-noNeigh"}
SEEDS = ("NEURAL-1", "NEURAL-2", "NEURAL-3")
KEYS = ["budget", "outer_fold", "trajectory", "feature_set"]


def load_selections(path: Path) -> pd.DataFrame:
    """Validate the complete 180-row v8 selection inventory, preserving literal NA."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {*KEYS, "city", "model", "config", "best_inner_epochs", "refit_epochs"}
    if not required.issubset(frame.columns):
        raise ValueError("selection inventory lacks required columns")
    frame["feature_set"] = frame["feature_set"].replace(SETS)
    expected = {
        (budget, str(fold), trajectory, feature, model)
        for budget in ("100", "FULL")
        for fold in range(5)
        for trajectory in (tuple(str(t) for t in range(1, 6)) if budget == "100" else ("NA",))
        for feature in SETS.values()
        for model in ("GraphSAGE-G2", "XGBoost")
    }
    actual = list(frame[[*KEYS, "model"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("selection inventory must contain each original unit/family exactly once")
    if set(frame.city) != {"zhengzhou"} or not set(frame.config) <= {"SMALL", "MEDIUM"}:
        raise ValueError("unsupported city or config")
    for row in frame.itertuples(index=False):
        if row.budget == "100":
            epochs = json.loads(row.best_inner_epochs)
            expected_epoch = outer_refit_epochs(epochs) if row.model == "GraphSAGE-G2" else 0
            if int(row.refit_epochs) != expected_epoch:
                raise ValueError("recorded refit epochs disagree with selected inner epochs")
    return frame


def feature_names(root: Path) -> dict[str, list[str]]:
    contract = yaml.safe_load((root / "configs/feature_sets_v1_2.yaml").read_text("utf-8"))
    result = {}
    for feature, count in zip(SETS.values(), (18, 16, 14), strict=True):
        entry = contract["sets"][feature]
        names = [n for n in EXPECTED_FEATURE_NAMES if n not in entry["dropped_from_F18"]]
        if names != entry["members"] or len(names) != count:
            raise ValueError("feature membership differs from frozen 18/16/14 contract")
        result[feature] = names
    return result


def plan(frame: pd.DataFrame, budget: str) -> dict:
    if budget not in {"100", "FULL"}:
        raise ValueError("budget must be 100 or FULL")
    selected = frame.loc[frame.budget.eq(budget)]
    missing = selected.loc[selected.model.eq("GraphSAGE-G2") & selected.refit_epochs.eq("")]
    return {
        "scope": "recorded-selection replay with current E1 helpers",
        "budget": budget,
        "units": len(selected) // 2,
        "fits": len(selected) // 2 * 10,
        "status": "blocked" if budget == "FULL" or len(missing) else "selection-ready",
        "missing_neural_epoch_records": len(missing),
        "numerical_replication_verified": False,
        "full_seed_provenance": "original generator unavailable; seed construction unverified",
    }


def refit_unit(data: dict, rows: pd.DataFrame, names: list[str], model_grid: dict) -> tuple:
    """One fixed selection, shared D8 schedule for G0/G1/G2, ten actual fits."""
    first = rows.iloc[0]
    unit = E1OuterUnit("zhengzhou", int(first.outer_fold), first.budget, first.trajectory)
    fold = e1._outer_fold_data(data, unit)
    fold = {**fold, "raw": fold["raw"].loc[:, names].copy()}
    selections = rows.set_index("model")
    graph = selections.loc["GraphSAGE-G2"]
    epochs = int(graph.refit_epochs)
    if epochs <= 0:
        raise ValueError("positive recorded D8 refit epochs required")
    blocks = {
        str(grid_id): f"{int(row.row) // 20}:{int(row.column) // 20}"
        for grid_id, row in fold["grid"].iterrows()
    }
    output, audit = [], []
    for label in ("XGB", "G0", "G1", "G2"):
        model = "XGBoost" if label == "XGB" else "GraphSAGE-G2"
        config = str(selections.loc[model, "config"])
        graph_id = "NA" if label == "XGB" else label
        for seed in ("NEURAL-1",) if label == "XGB" else SEEDS:
            task = TaskKey(
                protocol_id=PROTOCOL,
                experiment_id="E1",
                city="zhengzhou",
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
            scores, _ = e1._predict_outer(fold, task, model_grid, 0 if label == "XGB" else epochs)
            scores = np.asarray(scores)
            if scores.shape != (len(fold["test_node_ids"]),) or not np.isfinite(scores).all():
                raise ValueError("incomplete/nonfinite outer predictions")
            seed_name = model_grid["seeds"]["namespace"].format(
                protocol_id=PROTOCOL,
                experiment_id="E1",
                city="zhengzhou",
                outer_fold=task.outer_fold,
                budget=task.budget,
                trajectory_id=task.trajectory,
                model=model,
                config=config,
                seed_id=seed,
            )
            audit.append(
                {
                    "method": label,
                    "seed_slot": seed,
                    "seed_name": seed_name,
                    "seed_128": str(seed_from_name(seed_name)),
                    "config": config,
                    "epochs": 0 if label == "XGB" else epochs,
                }
            )
            by_id = dict(zip(fold["test_node_ids"], scores, strict=True))
            for slot in SEEDS if label == "XGB" else (seed,):
                bank = fold["banks"][["grid_id", "bank_id", "sample_role"]].copy()
                bank["score"] = bank.grid_id.map(by_id)
                bank["outer_block"] = bank.grid_id.map(blocks)
                if bank.score.isna().any():
                    raise ValueError("bank cell missing from predictions")
                bank = bank.assign(
                    city="zhengzhou",
                    budget=unit.budget,
                    trajectory=unit.trajectory,
                    outer_fold=unit.outer_fold,
                    feature_set=first.feature_set,
                    method=f"{label}|{first.feature_set}",
                    seed=slot,
                )
                output.append(bank)
    return pd.concat(output, ignore_index=True), audit


def run(root: Path, selection_path: Path, output: Path, budget: str) -> dict:
    frame = load_selections(selection_path)
    report = plan(frame, budget)
    if budget == "FULL":
        raise ValueError(
            "FULL replay blocked: original generator, refit epochs and model seed provenance "
            "are unrecovered. See docs/ABLATION.md."
        )
    if report["status"] != "selection-ready":
        raise ValueError("recorded selection is incomplete")
    names = feature_names(root)
    # Reuse the full live environment/input/source/probe validation before reading city data.
    gate_plan = reviewer_revision.prepare_plan(root)
    reviewer_revision.require_revision_environment(root, gate_plan)
    output = output.resolve()
    if not output.is_relative_to(root.resolve() / "reproduced"):
        raise ValueError("output must be a new directory below root/reproduced")
    output.mkdir(parents=True, exist_ok=False)
    model_grid = yaml.safe_load((root / "configs/model_grid.yaml").read_text("utf-8"))
    data = e1._read_city(root, "zhengzhou")
    report.update(
        {
            "status": "running",
            "selection_sha256": sha256_file(selection_path),
            "script_sha256": sha256_file(Path(__file__)),
            "gate_plan": gate_plan,
            "feature_contract_sha256": sha256_file(root / "configs/feature_sets_v1_2.yaml"),
            "model_grid_sha256": sha256_file(root / "configs/model_grid.yaml"),
            "completed_units": 0,
        }
    )
    e1._write_json(output / "manifest.json", report)
    for key, rows in frame.loc[frame.budget.eq(budget)].groupby(KEYS, sort=True):
        _, outer_fold, trajectory, feature = key
        stem = f"{feature}__f{outer_fold}__t{trajectory}"
        scores, audit = refit_unit(data, rows, names[feature], model_grid)
        scores.to_parquet(output / f"{stem}.parquet", index=False)
        e1._write_json(output / f"{stem}.json", {"fits": audit})
        report["completed_units"] += 1
        e1._write_json(output / "manifest.json", report)
    report["status"] = "selection-replay-complete"
    e1._write_json(output / "manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--selections",
        type=Path,
        default=ROOT / "provenance/selected_ablation.csv",
        help="published v8 selection inventory (must exist)",
    )
    parser.add_argument("--budget", choices=("100", "FULL"), default="100")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "plan":
        result = plan(load_selections(args.selections), args.budget)
    else:
        output = args.output or Path("reproduced/ablation-selection-replay")
        result = run(root, args.selections, root / output, args.budget)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
