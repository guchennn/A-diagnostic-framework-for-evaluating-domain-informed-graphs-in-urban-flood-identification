"""Run the v1.2 RQ2 prospective scarcity x feature-set substitution (Zhengzhou, n+=100).

RQ2 measures whether the increment of GraphSAGE on G2 over its controls changes when
supervision is reduced from FULL to n+ = 100 and specified D8-flow / neighbourhood
features are removed. It is a fixed-FULL conditional interaction: only the n+=100 side
is resampled. Every decision it consumes is frozen before inspection
(``configs/rq2_scarcity_freeze_v1_2.yaml``; see ``docs/ABLATION.md``).

Estimand chain (all in PB-AUPRC units, mean over the three outer evaluation banks):

    D(F, b)   = PB-AUPRC(G2, F, b) - PB-AUPRC(baseline, F, b)   baseline in {XGB, G0, G1}
    S(F)      = D(F, 100) - D(F, FULL)                          scarcity effect
    I_noD8    = S(F16-noD8) - S(F18)
    I_noNeigh = S(F14-noNeigh) - S(F18)

``D(F, FULL)`` and the FULL feature-set interactions are read from the frozen ablation
artifact and treated as fixed constants. ``D(F, 100)`` is fitted here (inner selection on
the G2 graph, then outer refit of G2/G0/G1/XGBoost per outer fold and trajectory) and
bootstrapped by resampling the outer folds (the top-level unit), carrying each resampled
fold's trajectory x seed x bank cells wholesale.

Confirmatory family (Holm, alpha = 0.05): {I_noD8|G0, I_noNeigh|G0}. XGBoost and G1
versions are secondary and reported with effect sizes and intervals but do not enter the
Holm family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from floodgnn_ceus.e1 import (
    _inner_fold_data,
    _outer_fold_data,
    _predict_outer,
    _read_city,
    _score_inner_fold,
)
from floodgnn_ceus.experiments import E1OuterUnit
from floodgnn_ceus.inference import (
    Comparison,
    basic_interval,
    holm_adjust,
    point_effect,
    two_sided_p,
    validate_paired_evaluation,
)
from floodgnn_ceus.io import canonical_json_bytes, sha256_file
from floodgnn_ceus.protocol import EXPECTED_FEATURE_NAMES
from floodgnn_ceus.reporting import markdown_table
from floodgnn_ceus.training import TaskKey, outer_refit_epochs

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CITY = "zhengzhou"
BUDGET = "100"
LABEL_LAYER = "STRICT-LOC"
REVEAL_MECHANISM = "uniform"
TRAJECTORIES = ("1", "2", "3", "4", "5")
OUTER_FOLDS = (0, 1, 2, 3, 4)
SEED_SLOTS = ("NEURAL-1", "NEURAL-2", "NEURAL-3")

# Freeze-config keys to the ablation regime names.
FEATURE_SET_TO_REGIME = {"F18": "full", "F16-noD8": "noD8", "F14-noNeigh": "noNeigh"}
DROP_TO_FEATURE_SET = {"noD8": "F16-noD8", "noNeigh": "F14-noNeigh"}
# (baseline label) -> (model family, graph_id).
BASELINES = {
    "XGB": ("XGBoost", "NA"),
    "G0": ("GraphSAGE-G2", "G0"),
    "G1": ("GraphSAGE-G2", "G1"),
}
TREATMENT = ("GraphSAGE-G2", "G2")  # G2 label
INNER_FAMILIES = ("XGBoost", "GraphSAGE-G2")
INNER_CONFIGS = ("SMALL", "MEDIUM")
CONFIG_ORDER = {"NA": 0, "SMALL": 1, "MEDIUM": 2}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _versions() -> dict[str, str]:
    names = ("numpy", "pandas", "pyarrow", "scikit-learn", "torch", "torch-geometric", "xgboost")
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def _custom_rng(seed_name: str) -> np.random.Generator:
    digest = hashlib.sha256(seed_name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:16], "big")
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _task(
    protocol_id: str,
    outer_fold: int,
    trajectory: str,
    model: str,
    graph_id: str,
    config: str,
    seed: str,
) -> TaskKey:
    return TaskKey(
        protocol_id=protocol_id,
        experiment_id="E1",
        city=CITY,
        label_layer=LABEL_LAYER,
        reveal_mechanism=REVEAL_MECHANISM,
        outer_fold=str(outer_fold),
        budget=BUDGET,
        trajectory=trajectory,
        feature_set="FULL-COMMON",
        model=model,
        graph_id=graph_id,
        realization_id="NA",
        config=config,
        seed=seed,
    )


def _feature_names(dropped: list[str]) -> list[str]:
    removed = set(dropped)
    names = [name for name in EXPECTED_FEATURE_NAMES if name not in removed]
    if len(names) != len(EXPECTED_FEATURE_NAMES) - len(removed):
        raise ValueError(f"dropped feature names are not in the frozen 18: {sorted(removed)}")
    return names


def _subset_features(fold: dict[str, object], names: list[str]) -> dict[str, object]:
    """Return the fold with its raw feature matrix restricted to ``names``."""
    result = dict(fold)
    raw = result["raw"]
    assert isinstance(raw, pd.DataFrame)
    result["raw"] = raw[list(names)]
    return result


def _block_map(grid: pd.DataFrame) -> dict[str, str]:
    return {
        str(row.grid_id): f"{int(row.row) // 20}:{int(row.column) // 20}"
        for row in grid.itertuples(index=False)
    }


def _select_unit(
    data: dict[str, pd.DataFrame],
    model_grid: dict[str, object],
    protocol_id: str,
    outer_fold: int,
    trajectory: str,
    names: list[str],
) -> dict[str, object]:
    """Select the best config per inner family for one (outer fold, trajectory) unit."""
    selected: dict[str, object] = {}
    for family in INNER_FAMILIES:
        model = "GraphSAGE-G2" if family == "GraphSAGE-G2" else family
        graph_id = "G2" if family == "GraphSAGE-G2" else "NA"
        choices: list[dict[str, object]] = []
        for config in INNER_CONFIGS:
            scores: list[float] = []
            best_epochs: list[int] = []
            for inner_fold in range(3):
                task = _task(protocol_id, outer_fold, trajectory, model, graph_id, config, "HPO-1")
                fold = _inner_fold_data(data, task, inner_fold)
                fold = _subset_features(fold, names)
                score, best_epoch, _ = _score_inner_fold(fold, task, model_grid)
                scores.append(float(score))
                best_epochs.append(int(best_epoch))
            choices.append(
                {
                    "config": config,
                    "score": float(np.mean(scores)),
                    "best_epochs": best_epochs,
                    "inner_scores": scores,
                }
            )
        choices.sort(key=lambda item: (-float(item["score"]), CONFIG_ORDER[str(item["config"])]))
        selected[family] = choices[0]
    return selected


def _refit_unit(
    data: dict[str, pd.DataFrame],
    grid: pd.DataFrame,
    model_grid: dict[str, object],
    protocol_id: str,
    outer_fold: int,
    trajectory: str,
    feature_set: str,
    names: list[str],
    selection: dict[str, object],
) -> pd.DataFrame:
    """Refit the selected configs for one unit and return per-cell bank rows."""
    unit = E1OuterUnit(CITY, outer_fold, BUDGET, trajectory)
    fold = _outer_fold_data(data, unit)
    fold = _subset_features(fold, names)
    test_ids = [str(value) for value in fold["test_node_ids"]]
    banks = fold["banks"]
    assert isinstance(banks, pd.DataFrame)
    blocks = _block_map(grid)

    graph_config = str(selection["GraphSAGE-G2"]["config"])
    graph_epochs = outer_refit_epochs(selection["GraphSAGE-G2"]["best_epochs"])
    schedules: dict[str, tuple[str, str, str, int]] = {}
    schedules["XGB"] = ("XGBoost", "NA", str(selection["XGBoost"]["config"]), 0)
    for label in ("G0", "G1", "G2"):
        schedules[label] = ("GraphSAGE-G2", label, graph_config, graph_epochs)

    rows: list[dict[str, object]] = []
    for label, (model, graph_id, config, epochs) in schedules.items():
        method = f"{label}|{feature_set}"
        n_seeds = 1 if model == "XGBoost" else 3
        seed_slots = SEED_SLOTS if n_seeds == 3 else ("NEURAL-1",)
        score_by_seed: dict[str, np.ndarray] = {}
        for seed in seed_slots:
            task = _task(protocol_id, outer_fold, trajectory, model, graph_id, config, seed)
            scores, _ = _predict_outer(fold, task, model_grid, epochs)
            score_by_seed[seed] = scores
        if n_seeds == 1:  # deterministic tree: replicate the single fit across seed slots
            single = score_by_seed["NEURAL-1"]
            score_by_seed = {seed: single for seed in SEED_SLOTS}
        for seed in SEED_SLOTS:
            scores = score_by_seed[seed]
            if len(scores) != len(test_ids) or not np.isfinite(scores).all():
                raise ValueError(f"incomplete scores for {method}, fold {outer_fold}")
            score_by_id = dict(zip(test_ids, scores, strict=True))
            for bank_id, bank in banks.groupby("bank_id", sort=True):
                for cell in bank.itertuples(index=False):
                    grid_id = str(cell.grid_id)
                    rows.append(
                        {
                            "city": CITY,
                            "budget": BUDGET,
                            "trajectory": trajectory,
                            "seed": seed,
                            "bank_id": int(bank_id),
                            "outer_fold": outer_fold,
                            "outer_block": blocks[grid_id],
                            "grid_id": grid_id,
                            "sample_role": str(cell.sample_role),
                            "method": method,
                            "score": float(score_by_id[grid_id]),
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "city",
            "budget",
            "trajectory",
            "seed",
            "bank_id",
            "outer_fold",
            "outer_block",
            "grid_id",
            "sample_role",
            "method",
            "score",
        ],
    )


def _load_full_side(
    root: Path, freeze: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = root / str(freeze["full_side"]["source"])
    expected = str(freeze["full_side"]["source_sha256"])
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(f"full-side ablation inference SHA changed: {actual}")
    table = pd.read_csv(source)
    return table, freeze["full_side"]


def _full_constants(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return D(F, FULL) and FULL feature-set interaction constants by baseline."""
    points: dict[str, dict[str, float]] = {}
    interactions: dict[str, dict[str, float]] = {}
    for baseline in ("XGB", "G0", "G1"):
        points[baseline] = {}
        for feature_set, regime in FEATURE_SET_TO_REGIME.items():
            row = table.loc[
                table["contrast"].eq(f"G2-{baseline}") & table["regime"].eq(regime), "point"
            ]
            if len(row) != 1:
                raise ValueError(f"missing D({baseline}, {regime}) in full-side ablation")
            points[baseline][feature_set] = float(row.iloc[0])
        interactions[baseline] = {}
        for drop, feature_set in DROP_TO_FEATURE_SET.items():
            row = table.loc[
                table["contrast"].eq(f"G2-{baseline}") & table["regime"].eq(f"{drop}-full"),
                "point",
            ]
            if len(row) != 1:
                raise ValueError(f"missing full interaction ({baseline}, {drop})")
            interactions[baseline][feature_set] = float(row.iloc[0])
    return {"D_full": points, "full_interaction": interactions}


def _bootstrap_effects(
    frame: pd.DataFrame,
    comparisons: list[Comparison],
    *,
    replicates: int,
    seed_name: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Paired outer-fold bootstrap (RQ2 primary inference).

    The outer fold is the top-level resampling unit: all held-out blocks within a
    fold share one fitted model, so folds — not blocks — are resampled with
    replacement. A fold's value is the paired treatment-minus-control effect
    averaged over trajectory x seed x bank; one replicate resamples the 5 folds
    and averages, drawing a single fold-index vector shared by every comparison so
    the interaction contrasts remain paired.
    """
    rng = _custom_rng(seed_name)
    for comparison in comparisons:
        subset = frame.loc[frame["budget"].astype(str).eq(comparison.budget)]
        validate_paired_evaluation(subset, (comparison.treatment, comparison.control))
    folds = np.sort(frame["outer_fold"].unique())
    if len(folds) < 2:
        raise ValueError("outer-fold bootstrap needs at least two outer folds")
    fold_effects = {
        comparison.comparison_id: np.asarray(
            [point_effect(frame.loc[frame["outer_fold"].eq(fold)], comparison) for fold in folds],
            dtype=np.float64,
        )
        for comparison in comparisons
    }
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        fold_positions = rng.integers(0, len(folds), size=len(folds))
        for comparison in comparisons:
            effect = float(fold_effects[comparison.comparison_id][fold_positions].mean())
            rows.append(
                {
                    "replicate": replicate,
                    "comparison_id": comparison.comparison_id,
                    "effect": effect,
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["replicate", "comparison_id"]).reset_index(drop=True),
        fold_effects,
    )


def run_rq2_scarcity(
    root: Path,
    output_root: Path,
    *,
    replicates: int = 10_000,
    outer_folds: tuple[int, ...] = OUTER_FOLDS,
    trajectories: tuple[str, ...] = TRAJECTORIES,
    resume: bool = False,
) -> dict[str, object]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not outer_folds or any(fold not in range(5) for fold in outer_folds):
        raise ValueError("outer folds must be a non-empty subset of 0..4")
    if not trajectories or any(str(t) not in TRAJECTORIES for t in trajectories):
        raise ValueError("trajectories must be a non-empty subset of 1..5")
    from floodgnn_ceus import reviewer_revision

    plan = reviewer_revision.prepare_plan(root)
    reviewer_revision.require_revision_environment(root, plan)
    output_root = output_root.resolve()
    units_dir = output_root / "units"
    selection_dir = output_root / "selections"
    if not resume and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    units_dir.mkdir(parents=True, exist_ok=True)
    selection_dir.mkdir(parents=True, exist_ok=True)

    freeze_path = root / "configs" / "rq2_scarcity_freeze_v1_2.yaml"
    freeze = yaml.safe_load(freeze_path.read_text("utf-8"))
    protocol_id = str(freeze["protocol_id"])
    if protocol_id != "FloodGNN-CEUS-ZZ-BJ-v1.0":
        raise ValueError("freeze protocol ID changed")
    feature_sets = list(freeze["feature_membership"]["feature_sets"])
    if feature_sets != ["F18", "F16-noD8", "F14-noNeigh"]:
        raise ValueError("freeze feature sets changed")
    dropped_by_set = {
        feature_set: list(freeze["feature_membership"][feature_set]["dropped"])
        for feature_set in feature_sets
    }
    confirmatory_baseline = str(freeze["inference"]["confirmatory_baseline"])
    confirmatory_family = [str(item) for item in freeze["inference"]["confirmatory_family"]]
    secondary_contrasts = [str(item) for item in freeze["inference"]["secondary_contrasts"]]
    seed_name = str(freeze["inference"]["bootstrap"]["seed_name"]).format(protocol_id=protocol_id)
    if confirmatory_baseline != "G0":
        raise ValueError("confirmatory baseline changed")
    if replicates != int(freeze["inference"]["bootstrap"]["replicates"]):
        raise ValueError("bootstrap replicates changed")

    full_table, _ = _load_full_side(root, freeze)
    full_constants = _full_constants(full_table)

    model_grid = yaml.safe_load((root / "configs" / "model_grid.yaml").read_text("utf-8"))
    data = _read_city(root, CITY)
    grid = data["grid"][["grid_id", "row", "column"]].copy()
    grid["grid_id"] = grid["grid_id"].astype(str)

    # Phase 1+2: per (feature set, outer fold, trajectory) selection and refit.
    names_by_set = {
        feature_set: _feature_names(dropped_by_set[feature_set]) for feature_set in feature_sets
    }
    completed_units: set[tuple[str, int, str]] = set()
    if resume:
        for path in units_dir.glob("*.parquet"):
            stem = path.stem.split("__")
            if len(stem) == 3:
                completed_units.add((stem[0], int(stem[1][1:]), stem[2][1:]))

    for feature_set in feature_sets:
        names = names_by_set[feature_set]
        for outer_fold in outer_folds:
            for trajectory in trajectories:
                key = (feature_set, outer_fold, trajectory)
                if key in completed_units:
                    continue
                selection = _select_unit(
                    data, model_grid, protocol_id, outer_fold, trajectory, names
                )
                _write_json(
                    selection_dir / f"{feature_set}__f{outer_fold}__t{trajectory}.json",
                    {
                        "feature_set": feature_set,
                        "outer_fold": outer_fold,
                        "trajectory": trajectory,
                        "selection": {
                            family: {
                                "config": str(selection[family]["config"]),
                                "mean_inner_score": float(selection[family]["score"]),
                                "best_epochs": [int(e) for e in selection[family]["best_epochs"]],
                            }
                            for family in INNER_FAMILIES
                        },
                    },
                )
                rows = _refit_unit(
                    data,
                    grid,
                    model_grid,
                    protocol_id,
                    outer_fold,
                    trajectory,
                    feature_set,
                    names,
                    selection,
                )
                shard = units_dir / f"{feature_set}__f{outer_fold}__t{trajectory}.parquet"
                partial = shard.with_name(shard.name + ".part")
                rows.to_parquet(partial, index=False)
                os.replace(partial, shard)
                completed_units.add(key)
                print(
                    f"completed {feature_set} fold {outer_fold} trajectory {trajectory}", flush=True
                )

    shards = sorted(units_dir.glob("*.parquet"))
    if not shards:
        raise ValueError("no unit shards were produced")
    frame = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    if frame.duplicated(
        [
            "budget",
            "trajectory",
            "seed",
            "bank_id",
            "outer_fold",
            "outer_block",
            "grid_id",
            "sample_role",
            "method",
        ]
    ).any():
        raise ValueError("combined cell frame contains duplicate method rows")

    # Phase 3: D(F, 100) bootstrap and the interactions.
    comparisons_by_id = {
        f"D-{baseline}-{feature_set}": Comparison(
            comparison_id=f"D-{baseline}-{feature_set}",
            budget=BUDGET,
            treatment=f"G2|{feature_set}",
            control=f"{baseline}|{feature_set}",
        )
        for baseline in ("XGB", "G0", "G1")
        for feature_set in feature_sets
    }
    comparisons = list(comparisons_by_id.values())
    bootstrap, fold_effects = _bootstrap_effects(
        frame, comparisons, replicates=replicates, seed_name=seed_name
    )

    draws_by_comparison = {
        comparison_id: group["effect"].to_numpy(dtype=np.float64)
        for comparison_id, group in bootstrap.groupby("comparison_id", sort=True)
    }

    d_rows: list[dict[str, object]] = []
    for comparison in comparisons:
        point = float(fold_effects[comparison.comparison_id].mean())
        samples = draws_by_comparison[comparison.comparison_id]
        lower, upper = basic_interval(point, samples)
        d_rows.append(
            {
                "baseline": comparison.control.split("|")[0],
                "feature_set": comparison.treatment.split("|", 1)[1],
                "point_effect": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "two_sided_p": two_sided_p(point, samples),
            }
        )
    d_inference = pd.DataFrame(d_rows)

    interaction_rows: list[dict[str, object]] = []
    for baseline in ("XGB", "G0", "G1"):
        base_id = f"D-{baseline}-F18"
        for drop, dropped_set in DROP_TO_FEATURE_SET.items():
            contrasted_id = f"D-{baseline}-{dropped_set}"
            frozen_full = full_constants["full_interaction"][baseline][dropped_set]
            point = float(
                (fold_effects[contrasted_id] - fold_effects[base_id]).mean() - frozen_full
            )
            draws = draws_by_comparison[contrasted_id] - draws_by_comparison[base_id] - frozen_full
            lower, upper = basic_interval(point, draws)
            interaction_id = f"I_{drop}|{baseline}"
            interaction_rows.append(
                {
                    "interaction": interaction_id,
                    "baseline": baseline,
                    "drop": drop,
                    "role": "confirmatory" if baseline == confirmatory_baseline else "secondary",
                    "point_effect": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "two_sided_p": two_sided_p(point, draws),
                }
            )

    interaction = pd.DataFrame(interaction_rows)
    raw_p = {
        row["interaction"]: float(row["two_sided_p"])
        for row in interaction_rows
        if row["interaction"] in set(confirmatory_family)
    }
    holm = holm_adjust(raw_p)
    interaction["holm_two_sided_p"] = interaction["interaction"].map(
        lambda key: holm.get(key, float("nan"))
    )

    # Scarcity effects S(F) = D(F, 100) - D(F, FULL) (descriptive).
    s_rows: list[dict[str, object]] = []
    for baseline in ("XGB", "G0", "G1"):
        for feature_set in feature_sets:
            comparison_id = f"D-{baseline}-{feature_set}"
            point = float(fold_effects[comparison_id].mean())
            samples = draws_by_comparison[comparison_id]
            d_full = full_constants["D_full"][baseline][feature_set]
            s_point = point - d_full
            lower, upper = basic_interval(s_point, samples - d_full)
            s_rows.append(
                {
                    "baseline": baseline,
                    "feature_set": feature_set,
                    "D_full": d_full,
                    "D_100": point,
                    "scarcity_effect": s_point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    scarcity = pd.DataFrame(s_rows)

    # Write outputs.
    paths = {
        "inner_selection": output_root / "inner_selection.csv",
        "cell_scores.parquet": output_root / "cell_scores.parquet",
        "d_inference.csv": output_root / "d_inference.csv",
        "interaction_inference.csv": output_root / "interaction_inference.csv",
        "interaction_bootstrap.parquet": output_root / "interaction_bootstrap.parquet",
        "scarcity.csv": output_root / "scarcity.csv",
    }
    inner_selection_rows: list[dict[str, object]] = []
    for path in sorted(selection_dir.glob("*.json")):
        payload = json.loads(path.read_text("utf-8"))
        for family in INNER_FAMILIES:
            item = payload["selection"][family]
            inner_selection_rows.append(
                {
                    "feature_set": payload["feature_set"],
                    "outer_fold": payload["outer_fold"],
                    "trajectory": payload["trajectory"],
                    "family": family,
                    "config": item["config"],
                    "mean_inner_score": item["mean_inner_score"],
                    "best_epochs": ",".join(str(e) for e in item["best_epochs"]),
                }
            )
    pd.DataFrame(inner_selection_rows).to_csv(paths["inner_selection"], index=False)
    frame.to_parquet(paths["cell_scores.parquet"], index=False)
    d_inference.to_csv(paths["d_inference.csv"], index=False)
    interaction.to_csv(paths["interaction_inference.csv"], index=False)
    bootstrap.to_parquet(paths["interaction_bootstrap.parquet"], index=False)
    scarcity.to_csv(paths["scarcity.csv"], index=False)

    manifest = {
        "status": "success",
        "protocol_id": protocol_id,
        "amendment_version": "v1.2",
        "city": CITY,
        "budget": BUDGET,
        "freeze_config": str(freeze_path),
        "freeze_config_sha256": sha256_file(freeze_path),
        "freeze_sha": str(freeze["freeze_sha"]),
        "feature_sets": feature_sets,
        "confirmatory_baseline": confirmatory_baseline,
        "confirmatory_family": confirmatory_family,
        "secondary_contrasts": secondary_contrasts,
        "full_side_sha256": str(freeze["full_side"]["source_sha256"]),
        "bootstrap_replicates": replicates,
        "bootstrap_seed_name": seed_name,
        "outer_folds": list(outer_folds),
        "trajectories": list(trajectories),
        "units": len(completed_units),
        "environment": _versions(),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths.values()
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    (output_root / "report.md").write_text(
        _report_text(d_inference, interaction, scarcity, manifest), encoding="utf-8"
    )
    return manifest


def _report_text(
    d_inference: pd.DataFrame,
    interaction: pd.DataFrame,
    scarcity: pd.DataFrame,
    manifest: dict[str, object],
) -> str:
    lines = [
        "# Zhengzhou RQ2 — prospective scarcity x feature-set substitution",
        "",
        f"protocol `{manifest['protocol_id']}`, amendment v1.2, budget n+=100.",
        f"Freeze SHA `{manifest['freeze_sha']}`.",
        "",
        "## D(F, 100) = G2 - baseline (PB-AUPRC)",
        "",
        markdown_table(d_inference),
        "",
        "## Interactions (I = [D(Fx,100) - D(F18,100)] - [D(Fx,FULL) - D(F18,FULL)])",
        "",
        markdown_table(interaction),
        "",
        "## Scarcity effects S(F) = D(F,100) - D(F,FULL)",
        "",
        markdown_table(scarcity),
        "",
        "## Reading",
        "",
        "- G2 in the model tables denotes GraphSAGE on the G2 graph; the estimand concerns",
        "  this learned message-passing implementation, not the information content of G2 itself.",
        "- Fixed-filter also uses G2, with a different fixed neighbourhood transformation.",
        "  The comparison therefore concerns how the same graph relationship is used.",
        f"- Confirmatory family (Holm, alpha=0.05): {manifest['confirmatory_family']}.",
        "- XGBoost and G1 interaction versions are secondary/descriptive and do not enter the",
        "  Holm family.",
        "- The FULL side (D(F,FULL) and FULL feature-set interactions) is a frozen constant read",
        "  from the retrospective ablation; only the n+=100 side is bootstrapped here.",
        "  This is a fixed-FULL conditional interaction, not an unconditional budget interaction;",
        "  its interval does not propagate FULL-side uncertainty. The frozen primary is unchanged.",
        "- Removing the specified derived features tests whether that deletion restores a",
        "  GraphSAGE-on-G2 advantage. It cannot rule out all feature redundancy: two G2 hops",
        "  span at most about 566 m of geometric path, while flow accumulation summarizes",
        "  upstream terrain at larger scales. The model does not cover a complete catchment.",
        "- G1 versus G2 compares complete graph specifications: direction, edge count, in-degree,",
        "  connectivity and receptive field change together; this is not direction's net effect.",
        "- Read interval endpoints as bounds on supported increment sizes under this fixed",
        "  spatial design and fixed-FULL inference, not merely as significant/non-significant.",
        "  These bootstrap intervals do not establish population equivalence or zero graph information.",
        "- PB-AUPRC (PB-AP) measures event-conditioned ranking of recorded positives against",
        "  background, not calibrated flood probability, future warning or cross-city transfer.",
    ]
    return "\n".join(lines) + "\n"


def _parse_folds(value: str) -> tuple[int, ...]:
    folds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(set(folds)) != len(folds):
        raise ValueError("outer folds must be unique")
    return folds


def _parse_trajectories(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    parser.add_argument("--trajectories", default="1,2,3,4,5")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    result = run_rq2_scarcity(
        root,
        output,
        replicates=args.replicates,
        outer_folds=_parse_folds(args.outer_folds),
        trajectories=_parse_trajectories(args.trajectories),
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
