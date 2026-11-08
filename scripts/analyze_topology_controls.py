"""Analyze complete topology-control fits; graph realizations remain within-fold repeats."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_predictions import positive_statistics
from metrics import ranking_metrics
from run_topology_controls import cache_key, task_key

from floodgnn_ceus import e1
from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.training import PROBE_CELL, TaskKey, seed_from_name

METRICS = ["sampled_ap", "full_ap", "weighted_ap", "recall_10"]
UNIT = ["city", "outer_fold", "draw", "seed", "predictor"]
FIT = [*UNIT, "representation", "realization"]


def evaluate(frame: pd.DataFrame, positives: set) -> dict:
    """Public metric helpers, preserving complete threshold ties and fractional quotas."""
    full = frame.loc[frame.bank_id.eq("NA")].sort_values("grid_id")
    y = full.grid_id.isin(positives).to_numpy(np.int8)
    scores = full.score.to_numpy(float)
    require(set(full.loc[y == 1, "grid_id"]) == positives, "positive inventory differs")
    natural = positive_statistics(y, scores)
    weighted = positive_statistics(y, scores, 10 * y.sum() / (len(y) - y.sum()))
    banks = []
    for bank in ("1", "2", "3"):
        part = frame.loc[frame.bank_id.eq(bank)].sort_values("grid_id")
        banks.append(
            positive_statistics(
                part.sample_role.eq("positive").to_numpy(np.int8), part.score.to_numpy(float)
            )[:, 0].sum()
        )
    return {
        "sampled_ap": float(np.mean(banks)),
        "full_ap": float(natural[:, 0].sum()),
        "weighted_ap": float(weighted[:, 0].sum()),
        "recall_10": ranking_metrics(y, scores)["recall_top_10pct"],
    }


def verify_reference(root: Path, run: Path) -> dict:
    """Require unchanged inputs/source files and a valid baseline resource-probe binding."""
    require(run.resolve().is_relative_to(root.resolve()), "reference run must be inside root")
    path = run / "local_environment_lock.json"
    lock = read_json(path)
    hashes = {path.relative_to(root).as_posix(): sha256_file(path)}
    require(
        lock["protocol_lock_sha256"] == sha256_file(root / "artifacts/protocol_lock.json"),
        "reference protocol differs",
    )
    for item in lock["files"]:
        require(
            sha256_file(root / item["path"]) == item["sha256"], "reference source or input changed"
        )
        hashes[item["path"]] = item["sha256"]
    probe_path = run / "resource_probe.json"
    probe = read_json(probe_path)
    require(
        probe["lock_sha256"] == sha256_file(path)
        and probe["source_sha256"] == lock["source_sha256"]
        and probe["input_sha256"] == lock["input_sha256"],
        "reference probe binding differs",
    )
    require(
        probe["status"] == "success"
        and probe["performance_blind"]
        and not probe["metrics_computed"]
        and not probe["predictions_written"],
        "reference probe incomplete",
    )
    require(
        probe.get("epochs") == 1 and probe.get("cell") == PROBE_CELL,
        "incorrect reference probe cell or duration",
    )
    hashes[probe_path.relative_to(root).as_posix()] = sha256_file(probe_path)
    return hashes


def read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summarize(folds: pd.DataFrame, keys: list[str], *, contrast: bool = False) -> pd.DataFrame:
    draws = np.random.default_rng(20261011).integers(0, 5, (10000, 5))
    output = []
    for names, part in folds.groupby(keys, sort=True):
        part = part.sort_values("outer_fold")
        require(part.outer_fold.tolist() == list(range(5)), "exactly five spatial folds required")
        values = part.value.to_numpy(float)
        require(np.isfinite(values).all(), "non-finite fold metric")
        low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
        row = dict(zip(keys, names if isinstance(names, tuple) else (names,), strict=True))
        row.update(
            mean=values.mean(),
            range_low=low,
            range_high=high,
            n_folds=5,
            positive_folds=int((values > 0).sum()),
            negative_folds=int((values < 0).sum()),
            **{f"fold{i}": v for i, v in enumerate(values)},
        )
        if contrast:
            row.update(
                mean_pp=100 * values.mean(), range_low_pp=100 * low, range_high_pp=100 * high
            )
        output.append(row)
    return pd.DataFrame(output)


def aggregate_metrics(frame: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Require every paired fit; average metrics, never prediction scores."""
    require(not frame.duplicated(FIT).any(), "duplicate fit identity")
    require(np.isfinite(frame[METRICS].to_numpy(float)).all(), "non-finite fit metric")
    expected = set()
    for city, fold, draw, seed, variant in product(
        config["cities"],
        config["outer_folds"],
        config["trajectories"],
        config["outer_seeds"],
        config["variants"],
    ):
        base = (city, int(fold), str(draw), seed, variant["model"])
        expected.add((*base, "D8", 0))
        expected.update((*base, "G7", r) for r in config["graph_realizations"])
    actual = set(frame[FIT].itertuples(index=False, name=None))
    require(
        actual == expected,
        f"missing or unexpected fits: {len(expected - actual)}/{len(actual - expected)}",
    )
    reference = frame.loc[frame.representation.eq("D8"), UNIT + METRICS]
    control = frame.loc[frame.representation.eq("G7")]
    paired = control.merge(reference, on=UNIT, suffixes=("_control", "_D8"), validate="many_to_one")
    differences = paired[UNIT + ["realization"]].copy()
    for metric in METRICS:
        differences[metric] = paired[f"{metric}_D8"] - paired[f"{metric}_control"]
    long = differences.melt(
        id_vars=UNIT + ["realization"], value_vars=METRICS, var_name="metric", value_name="value"
    )
    # Graph variants are repeated realizations of one control, not additional folds.
    unit_means = long.groupby(UNIT + ["metric"], as_index=False).value.mean()
    paired_folds = unit_means.groupby(
        ["city", "predictor", "outer_fold", "metric"], as_index=False
    ).value.mean()
    realization_folds = long.groupby(
        ["city", "predictor", "realization", "outer_fold", "metric"], as_index=False
    ).value.mean()
    absolute_units = frame.groupby(UNIT + ["representation"], as_index=False)[METRICS].mean()
    absolute_long = absolute_units.melt(
        id_vars=UNIT + ["representation"], value_vars=METRICS, var_name="metric", value_name="value"
    )
    absolute_folds = absolute_long.groupby(
        ["city", "predictor", "representation", "outer_fold", "metric"], as_index=False
    ).value.mean()
    return {
        "absolute_folds": absolute_folds,
        "absolute_summary": summarize(
            absolute_folds, ["city", "predictor", "representation", "metric"]
        ),
        "paired_folds": paired_folds,
        "paired_summary": summarize(paired_folds, ["city", "predictor", "metric"], contrast=True),
        "graph_realization_folds": realization_folds,
        "graph_realization_contrasts": summarize(
            realization_folds, ["city", "predictor", "realization", "metric"], contrast=True
        ),
    }


def verify_provenance(root: Path, config: dict) -> tuple[Path, dict, list[dict]]:
    run = root / "artifacts/runs" / config["run_id"]
    lock_path = run / "local_environment_lock.json"
    lock = read_json(lock_path)
    verified = {}
    for name, expected in {**lock["source_files"], **lock["graph_files"]}.items():
        require(sha256_file(root / name) == expected, f"frozen hash mismatch: {name}")
        verified[name] = expected
    require(
        sha256_file(root / "artifacts/protocol_lock.json") == lock["protocol_lock_sha256"],
        "protocol lock changed",
    )
    local = root / lock["local_revision_lock"]
    require(
        sha256_file(local) == lock["local_revision_lock_sha256"], "local environment lock changed"
    )
    local_lock = read_json(local)
    for item in local_lock["files"]:
        require(
            sha256_file(root / item["path"]) == item["sha256"], f"input changed: {item['path']}"
        )
        verified[item["path"]] = item["sha256"]
    probe = read_json(run / "resource_probe.json")
    require(probe["topology_lock_sha256"] == sha256_file(lock_path), "probe bound to another lock")
    require(
        probe["performance_blind"]
        and not probe["metrics_computed"]
        and not probe["predictions_written"]
        and probe["status"] == "success",
        "resource probe is incomplete or not performance-blind",
    )
    require(
        probe.get("epochs") == 1 and probe.get("cell") == PROBE_CELL,
        "incorrect resource probe cell or duration",
    )
    expected_keys = set()
    for city, fold, realization, partition in product(
        config["cities"], config["outer_folds"], config["graph_realizations"], ["train", "test"]
    ):
        expected_keys.add(cache_key(city, fold, "NA", None, partition, realization))
        for draw, inner in product(config["trajectories"], config["inner_folds"]):
            expected_keys.add(cache_key(city, fold, draw, inner, partition, realization))
    audits, observed_keys = [], set()
    for path in sorted((run / "graphs").rglob("*.json")):
        record = read_json(path)
        require(record["key"] not in observed_keys, "duplicate graph cache identity")
        observed_keys.add(record["key"])
        require(
            path.relative_to(run / "graphs").with_suffix("").as_posix() == record["key"],
            "graph cache identity differs from path",
        )
        require(
            path.relative_to(root).as_posix() in lock["graph_files"]
            and path.with_suffix(".parquet").relative_to(root).as_posix() in lock["graph_files"],
            "graph absent from frozen lock",
        )
        require(record["accepted"] is True, f"graph rejected: {path}")
        for key in ("indegree_preserved", "outdegree_preserved", "distance_counts_preserved"):
            require(record[key] is True, f"graph invariant failed: {key}")
        require(
            record["changed_fraction"] >= config["graph"]["min_changed_fraction"],
            "graph perturbation too small",
        )
        require(
            sha256_file(path.with_suffix(".parquet")) == record["parquet_sha256"],
            "graph bytes changed",
        )
        parts = record["key"].split("/")
        scope = parts[2]
        audits.append(
            {
                "city": parts[0],
                "outer_fold": int(parts[1][1:]),
                "scope": "outer" if scope == "outer" else "inner",
                "draw": "NA" if scope == "outer" else scope.split("-")[1][1:],
                "inner_fold": -1 if scope == "outer" else int(scope.split("-")[2][1:]),
                "partition": parts[3].split("-r")[0],
                "realization": int(parts[3].split("-r")[1]),
                "edges": record["edges"],
                "changed_edges": record["changed_edges"],
                "changed_fraction": record["changed_fraction"],
                "diagonal_edges": record["diagonal_edges"],
                "cross_tile_edges": record["cross_tile_edges"],
                "solver_fallback_tiles": record["solver_fallback_tiles"],
                "exact_in_out_degree": True,
                "distance_counts_preserved": True,
            }
        )
    require(len(audits) == 960, "complete 960-graph cache required")
    require(observed_keys == expected_keys, "missing or unexpected graph cache identities")
    return (
        run,
        {
            "topology_lock_sha256": sha256_file(lock_path),
            "verified_files": verified,
            "local_environment": local_lock["environment"],
        },
        audits,
    )


def collect_receipts(
    root: Path, run: Path, config: dict, reference_run: Path
) -> tuple[list, list, dict]:
    """Check all expected receipts and payload hashes before computing any metrics."""
    grid = yaml.safe_load((root / "configs/model_grid.yaml").read_text("utf-8"))
    main = reference_run
    records, ledger, hashes = [], [], {}
    for city, fold, draw, variant in product(
        config["cities"], config["outer_folds"], config["trajectories"], config["variants"]
    ):
        predictor = variant["model"]
        for realization in [0, *config["graph_realizations"]]:
            if realization == 0:
                base_variant = variant["id"].replace("G7", "G2")
                selection_path = main / "selections" / f"{city}-f{fold}-b100-t{draw}.json"
                selected = read_json(selection_path)["variants"][base_variant]
            else:
                selection_path = (
                    run
                    / "selections"
                    / f"{city}-f{fold}-t{draw}-r{realization}-{variant['id']}.json"
                )
                selected = read_json(selection_path)
            require(selected["config"] in config["candidates"], "unknown selected candidate")
            require(
                len(selected["best_epochs"]) == 3 and min(selected["best_epochs"]) > 0,
                "three valid inner stopping epochs required",
            )
            require(
                int(np.median(selected["best_epochs"])) == selected["refit_epochs"],
                "selected duration differs from inner median",
            )
            hashes[selection_path.relative_to(root).as_posix()] = sha256_file(selection_path)
            for seed in config["outer_seeds"]:
                if realization:
                    task = task_key(
                        config,
                        grid,
                        city,
                        fold,
                        draw,
                        variant,
                        realization,
                        selected["config"],
                        seed,
                    )
                    status_path = e1._status_path(run / "outer", task)
                else:
                    task = TaskKey(
                        protocol_id=grid["protocol_id"],
                        experiment_id="reviewer-revision-v1",
                        city=city,
                        label_layer="STRICT-LOC",
                        reveal_mechanism="uniform",
                        outer_fold=str(fold),
                        budget="100",
                        trajectory=str(draw),
                        feature_set="FULL-COMMON",
                        model=predictor,
                        graph_id="G2",
                        realization_id=base_variant,
                        config=selected["config"],
                        seed=seed,
                    )
                    status_path = e1._status_path(main / "outer", task)
                receipt = read_json(status_path)
                require(
                    receipt["task"] == task.as_dict() and receipt["status"] == "success",
                    f"incomplete or mismatched receipt: {status_path}",
                )
                payload = status_path.with_name("oof.parquet")
                require(
                    sha256_file(payload) == receipt["oof_sha256"], f"OOF hash differs: {payload}"
                )
                for path in (status_path, payload):
                    hashes[path.relative_to(root).as_posix()] = sha256_file(path)
                identity = {
                    "city": city,
                    "outer_fold": int(fold),
                    "draw": str(draw),
                    "seed": seed,
                    "predictor": predictor,
                    "representation": "G7" if realization else "D8",
                    "realization": realization,
                }
                records.append((identity, task, payload))
                meta = dict(
                    identity,
                    selected_config=selected["config"],
                    best_inner_epochs=";".join(map(str, selected["best_epochs"])),
                    scheduled_refit_epochs=selected["refit_epochs"],
                    status="success",
                    oof_sha256=receipt["oof_sha256"],
                )
                namespace = grid["seeds"]["namespace"].format(
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
                meta.update(
                    seed_namespace=namespace,
                    seed_uint128=str(seed_from_name(namespace)),
                    torch_seed=seed_from_name(namespace) % (2**63),
                    seed_source="locked e1 optimizer seed rule",
                )
                if realization:
                    metadata_path = status_path.with_name("training_metadata.json")
                    training = read_json(metadata_path)
                    require(
                        training["selected_epochs"] == selected["refit_epochs"],
                        "training metadata duration differs",
                    )
                    for field in ("seed_uint128", "torch_seed", "seed_namespace"):
                        require(training[field] == meta[field], f"training seed differs: {field}")
                    hashes[metadata_path.relative_to(root).as_posix()] = sha256_file(metadata_path)
                ledger.append(meta)
    require(len(records) == 1200, "expected 900 controls and 300 D8 fits")
    return records, ledger, hashes


def metrics_from_records(root: Path, records: list) -> pd.DataFrame:
    design = {}
    for city in {x[0]["city"] for x in records}:
        directory = root / "data/interim" / city / "evaluation_design"
        outer = pd.read_parquet(directory / "outer_partitions.parquet")
        banks = pd.read_parquet(directory / "evaluation_banks.parquet")
        for fold in range(5):
            ids = set(outer.loc[outer.outer_fold.eq(fold) & outer.partition.eq("test"), "grid_id"])
            block = banks.loc[
                banks.scope.eq("outer")
                & banks.outer_fold.eq(fold)
                & banks.label_layer.eq("STRICT-LOC")
            ]
            positives = set(block.loc[block.sample_role.eq("positive"), "grid_id"])
            design[city, fold] = ids, positives, block
    rows = []
    for identity, task, payload in records:
        frame = pd.read_parquet(payload)
        for field, value in task.as_dict().items():
            require(
                frame[field].astype(str).eq(str(value)).all(), f"OOF task identity differs: {field}"
            )
        require(not frame.duplicated(["bank_id", "grid_id"]).any(), "duplicate OOF rows")
        require(frame.status.eq("success").all(), "unsuccessful OOF rows")
        require(np.isfinite(frame.score).all(), "non-finite scores")
        ids, positives, banks = design[identity["city"], identity["outer_fold"]]
        full = frame.loc[frame.bank_id.eq("NA")]
        require(set(full.grid_id) == ids, "incomplete full test grid")
        require(set(frame.bank_id) == {"NA", "1", "2", "3"}, "missing or extra evaluation bank")
        for bank in ("1", "2", "3"):
            actual = frame.loc[frame.bank_id.eq(bank), ["grid_id", "sample_role"]]
            expected = banks.loc[banks.bank_id.astype(str).eq(bank), ["grid_id", "sample_role"]]
            require(
                set(actual.itertuples(index=False, name=None))
                == set(expected.itertuples(index=False, name=None)),
                "evaluation bank changed",
            )
            aligned = actual.grid_id.map(full.set_index("grid_id").score)
            require(
                np.array_equal(
                    aligned.to_numpy(), frame.loc[frame.bank_id.eq(bank), "score"].to_numpy()
                ),
                "bank scores differ from full predictions",
            )
        values = evaluate(frame, positives)
        rows.append({**identity, **{k: values[k] for k in METRICS}})
        if len(rows) % 100 == 0:
            print(f"Verified and evaluated {len(rows)}/{len(records)} fits", flush=True)
    return pd.DataFrame(rows)


def existing_g6_tables(root: Path) -> tuple[dict, dict]:
    """Carry forward verified aggregate comparisons without refitting or pooling controls."""
    tables, hashes = {}, {}
    for kind in ("summary", "folds"):
        path = root / f"results/minor_revision_final_20261029/contrasts_{kind}.csv"
        if not path.exists():
            continue
        source = pd.read_csv(path)
        selected = source.loc[
            source.series.eq("main")
            & source.contrast.isin(["FF-G2-1 - FF-G6-1", "SAGE-G2-2 - SAGE-G6-2"])
            & source.metric.isin(METRICS)
        ].copy()
        require(
            len(selected) == (16 if kind == "summary" else 80),
            f"existing G6 {kind} comparisons incomplete",
        )
        selected["source_path"] = path.relative_to(root).as_posix()
        selected["source_sha256"] = sha256_file(path)
        selected["provenance"] = "existing verified aggregate; not refit or pooled with G7"
        tables[f"existing_G6_comparisons_{kind}"] = selected
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return tables, hashes


def analyze(root: Path, output: Path, reference_run: Path) -> dict:
    config = yaml.safe_load((root / "configs/topology_controls_v1.yaml").read_text("utf-8"))
    run, provenance, graph_rows = verify_provenance(root, config)
    reference_hashes = verify_reference(root, reference_run)
    records, ledger, hashes = collect_receipts(root, run, config, reference_run)
    hashes.update(reference_hashes)
    frame = metrics_from_records(root, records)
    tables = aggregate_metrics(frame, config)
    graph = pd.DataFrame(graph_rows)
    tables["graph_quality"] = graph
    quality = []
    for keys, part in graph.groupby(["city", "scope", "partition"], sort=True):
        quality.append(
            dict(zip(("city", "scope", "partition"), keys, strict=True))
            | {
                "graphs": len(part),
                "minimum_changed_fraction": part.changed_fraction.min(),
                "maximum_changed_fraction": part.changed_fraction.max(),
                "edge_weighted_changed_fraction": part.changed_edges.sum() / part.edges.sum(),
                "solver_fallback_tiles": int(part.solver_fallback_tiles.sum()),
            }
        )
    tables["graph_quality_summary"] = pd.DataFrame(quality)
    tables["graph_quality_outer_folds"] = (
        graph.loc[graph.scope.eq("outer")]
        .groupby(["city", "outer_fold", "partition"], as_index=False)
        .agg(
            graphs=("realization", "size"),
            minimum_changed_fraction=("changed_fraction", "min"),
            mean_changed_fraction=("changed_fraction", "mean"),
            maximum_changed_fraction=("changed_fraction", "max"),
            solver_fallback_tiles=("solver_fallback_tiles", "sum"),
        )
    )
    tables["training_records"] = pd.DataFrame(ledger)
    previous, previous_hashes = existing_g6_tables(root)
    tables.update(previous)
    hashes.update(previous_hashes)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    manifest = {
        "status": "complete",
        "control_fits": 900,
        "reference_fits": 300,
        "accepted_graphs": 960,
        "bootstrap_seed": 20261011,
        "bootstrap_samples": 10000,
        "independent_summary_units": "five spatial folds",
        "graph_repetitions": "averaged within fold/draw/refit; not independent folds",
        "contrast": "D8 minus G7",
        "metric_units": "fractions; paired *_pp columns multiply by 100",
        "scores_ensembled": False,
        "metrics_recomputed": True,
        "training_performed_by_analysis": False,
        **provenance,
        "input_hashes": hashes,
        "analysis_sha256": sha256_file(Path(__file__)),
        "reference_run": reference_run.relative_to(root).as_posix(),
        "metric_helper_sha256": {
            name: sha256_file(root / name)
            for name in (
                "scripts/metrics.py",
                "scripts/evaluate_predictions.py",
                "src/floodgnn_ceus/inference.py",
            )
        },
        "outputs": {name: sha256_file(output / f"{name}.csv") for name in tables},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {k: manifest[k] for k in ("status", "control_fits", "reference_fits", "accepted_graphs")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("results/topology_controls_20261108"))
    parser.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="Completed matched-model run directory, relative to --root",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    reference_run = (root / args.reference_run).resolve()
    require(reference_run.is_relative_to(root), "reference run must be inside repository root")
    print(json.dumps(analyze(root, output, reference_run)))


if __name__ == "__main__":
    main()
