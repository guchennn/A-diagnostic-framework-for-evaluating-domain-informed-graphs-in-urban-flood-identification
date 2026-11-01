"""Explicit E1 task materialization from the frozen E0 design."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file
from floodgnn_ceus.training import TaskKey

CITIES = ("zhengzhou", "beijing")
BUDGETS = ("20", "50", "100", "200", "FULL")
MODELS = (
    "KDE",
    "XGBoost",
    "LightGBM",
    "MLP",
    "Neighbor-XGB",
    "Fixed-filter",
    "GraphSAGE-G2",
)
GRAPH_BY_MODEL = {
    "KDE": "NA",
    "XGBoost": "NA",
    "LightGBM": "NA",
    "MLP": "NA",
    "Neighbor-XGB": "G1",
    "Fixed-filter": "G2",
    "GraphSAGE-G2": "G2",
}
NEURAL_MODELS = {"MLP", "Fixed-filter", "GraphSAGE-G2"}


@dataclass(frozen=True)
class E1Cell:
    city: str
    budget: str
    trajectory: str
    status: str
    reason: str


@dataclass(frozen=True)
class E1OuterUnit:
    city: str
    outer_fold: int
    budget: str
    trajectory: str


def _evaluation_statuses(root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    report = json.loads(
        (root / "artifacts" / "reports" / "evaluation_design.json").read_text("utf-8")
    )
    statuses = {}
    for city in report["cities"]:
        city_id = str(city["city_id"])
        for item in city["statuses"]:
            if item["label_layer"] == "STRICT-LOC":
                statuses[(city_id, str(item["budget"]))] = (
                    str(item["status"]),
                    "" if item["reason"] is None else str(item["reason"]),
                )
    return statuses


def e1_cells(root: Path) -> list[E1Cell]:
    """Return every frozen E1 cell, preserving legal not-estimable budgets."""
    statuses = _evaluation_statuses(root)
    cells = []
    for city in CITIES:
        for budget in BUDGETS:
            status, reason = statuses[(city, budget)]
            trajectories: Iterable[str]
            if status != "success" or budget == "FULL":
                trajectories = ("NA",)
            else:
                trajectories = ("1", "2", "3", "4", "5")
            cells.extend(
                E1Cell(city, budget, trajectory, status, reason) for trajectory in trajectories
            )
    return cells


def configuration_candidates(budget: str, model: str) -> tuple[str, ...]:
    """Return the only configurations allowed before outer selection."""
    if model not in MODELS or budget not in BUDGETS:
        raise ValueError("unknown E1 budget or model")
    if model == "KDE":
        return ("NA",)
    if budget == "20":
        return ("SMALL",)
    return ("SMALL", "MEDIUM")


def inner_task_keys(root: Path) -> list[TaskKey]:
    """Materialize one task key per complete three-fold inner-CV candidate."""
    protocol = json.loads((root / "artifacts" / "protocol_lock.json").read_text("utf-8"))
    protocol_id = str(protocol["base_protocol_id"])
    tasks = []
    for cell in e1_cells(root):
        if cell.status != "success":
            continue
        for outer_fold in range(5):
            for model in MODELS:
                for config in configuration_candidates(cell.budget, model):
                    tasks.append(
                        TaskKey(
                            protocol_id=protocol_id,
                            experiment_id="E1-inner",
                            city=cell.city,
                            label_layer="STRICT-LOC",
                            reveal_mechanism="uniform",
                            outer_fold=str(outer_fold),
                            budget=cell.budget,
                            trajectory=cell.trajectory,
                            feature_set="FULL-COMMON",
                            model=model,
                            graph_id=GRAPH_BY_MODEL[model],
                            realization_id="NA",
                            config=config,
                            seed="HPO-1",
                        )
                    )
    return tasks


def outer_units(root: Path) -> list[E1OuterUnit]:
    """Return each feasible E1 outer fold after inner selection."""
    return [
        E1OuterUnit(cell.city, outer_fold, cell.budget, cell.trajectory)
        for cell in e1_cells(root)
        if cell.status == "success"
        for outer_fold in range(5)
    ]


def prepare_e1_task_plan(root: Path) -> dict[str, object]:
    """Write a performance-blind E1 task-count and status report."""
    root = root.resolve()
    cells = e1_cells(root)
    tasks = inner_task_keys(root)
    by_city = {}
    for city in CITIES:
        city_cells = [cell for cell in cells if cell.city == city]
        successful = [cell for cell in city_cells if cell.status == "success"]
        city_tasks = [task for task in tasks if task.city == city]
        by_city[city] = {
            "inner_cv_tasks": len(city_tasks),
            "inner_fold_runs": len(city_tasks) * 3,
            "not_estimable": [
                {"budget": cell.budget, "reason": cell.reason}
                for cell in city_cells
                if cell.status == "not-estimable"
            ],
            "outer_refit_tasks_after_selection": len(successful) * 5 * 13,
            "successful_budget_trajectories": len(successful),
        }
    task_hash = sha256_bytes(canonical_json_bytes([task.as_dict() for task in tasks]))
    payload: dict[str, object] = {
        "cities": by_city,
        "deterministic_seed_slots": ["NEURAL-1", "NEURAL-2", "NEURAL-3"],
        "inner_task_key_sha256": task_hash,
        "labels_read": False,
        "performance_blind": True,
        "protocol_lock_sha256": sha256_file(root / "artifacts" / "protocol_lock.json"),
        "schema_version": 1,
        "status": "success",
        "totals": {
            "inner_cv_tasks": len(tasks),
            "inner_fold_runs": len(tasks) * 3,
            "outer_refit_tasks_after_selection": sum(
                int(value["outer_refit_tasks_after_selection"]) for value in by_city.values()
            ),
        },
    }
    path = root / "artifacts" / "reports" / "e1_task_plan.json"
    encoded = canonical_json_bytes(payload)
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError(f"E1 task plan differs: {path}")
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        partial.write_bytes(encoded)
        os.replace(partial, path)
    return payload
