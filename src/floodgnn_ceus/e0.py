"""Performance-blind E0 audit and compact protocol lock."""

from __future__ import annotations

import json
import os
import stat
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path

import pandas as pd
import psutil
import pyarrow.parquet as pq
import yaml

from floodgnn_ceus.audit import inspect_crowdflood_archive
from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file
from floodgnn_ceus.protocol import (
    EXPECTED_CITIES,
    EXPECTED_FEATURE_NAMES,
    effective_protocol_id,
    load_protocol,
)

EXPECTED_NODES = {"zhengzhou": 25_155, "beijing": 61_443}
ALLOWED_STATUSES = {"success", "not-estimable"}
DESIGN_FILES = (
    "outer_partitions.parquet",
    "inner_blocks.parquet",
    "reveals.parquet",
    "u_master_index.parquet",
    "u_samples.parquet",
    "evaluation_banks.parquet",
)
DESIGN_SCHEMAS = {
    "outer_partitions.parquet": [
        "city_id",
        "outer_fold",
        "grid_id",
        "block_row",
        "block_col",
        "partition",
    ],
    "inner_blocks.parquet": [
        "city_id",
        "outer_fold",
        "label_layer",
        "budget",
        "trajectory_id",
        "block_row",
        "block_col",
        "positive_count",
        "node_count",
        "inner_fold",
    ],
    "reveals.parquet": [
        "city_id",
        "label_layer",
        "reveal_mechanism",
        "outer_fold",
        "budget",
        "trajectory_id",
        "reveal_rank",
        "grid_id",
    ],
    "u_master_index.parquet": [
        "city_id",
        "scope",
        "label_layer",
        "outer_fold",
        "budget",
        "trajectory_id",
        "inner_fold",
        "candidate_count",
        "seed_name",
        "seed_uint128",
        "candidate_sha256",
        "order_sha256",
    ],
    "u_samples.parquet": [
        "city_id",
        "scope",
        "label_layer",
        "outer_fold",
        "budget",
        "trajectory_id",
        "inner_fold",
        "u_rank",
        "grid_id",
    ],
    "evaluation_banks.parquet": [
        "city_id",
        "scope",
        "label_layer",
        "outer_fold",
        "budget",
        "trajectory_id",
        "inner_fold",
        "bank_id",
        "sample_role",
        "sample_rank",
        "grid_id",
    ],
}
GRAPH_FILES = (
    "base_edges.parquet",
    "outer_edges.parquet",
    "graph_index.parquet",
    "g5_audit.parquet",
)
CITY_FILES = (
    "admin_aoi.parquet",
    "final_scorable_grid.parquet",
    "final_label_ledger.parquet",
    "final_label_qc_ledger.parquet",
    "dsm_200m_float64.tif",
    "filled_dsm_200m_float64.tif",
    "flat_mask_200m_int64.tif",
    "d8_direction_200m_int16.tif",
    "flow_accumulation_200m_int64.tif",
    "d8_edges.parquet",
    "terrain_features.parquet",
    "features_18.parquet",
)
SOURCE_REPORTS = (
    "public_sources.json",
    "crowdflood_audit.json",
    "worldcover.json",
    "osm.json",
    "worldpop.json",
    "imerg_inventory.json",
    "imerg.json",
    "dsm_final_aoi.json",
)
CITY_REPORTS = (
    "admin_aoi.json",
    "candidate_grid.json",
    "label_qc.json",
    "terrain.json",
    "features_18.json",
)
FORBIDDEN_MODEL_COLUMNS = {
    "raw_id",
    "member_row_hashes",
    "member_count",
    "text",
    "toponym",
    "SSA",
    "similarity",
    "flood_prob",
    "center_x",
    "center_y",
    "row",
    "column",
}
EXPECTED_FORBIDDEN_DESIGN_COLUMNS = (
    FORBIDDEN_MODEL_COLUMNS - {"SSA", "similarity", "row", "column"}
) | {"geometry"}


def _read_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a mapping")
    return value


def validate_label_schema_contract(schema: Mapping[str, object]) -> set[str]:
    """Validate the design-only forbidden-column contract frozen in YAML."""
    configured = schema.get("forbidden_design_columns")
    if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
        raise TypeError("forbidden design columns must be a string list")
    if len(configured) != len(set(configured)):
        raise ValueError("forbidden design columns contain duplicates")
    values = set(configured)
    if values != EXPECTED_FORBIDDEN_DESIGN_COLUMNS:
        raise ValueError("forbidden design columns differ from the audited design contract")
    return values


def validate_analysis_contract(contract: Mapping[str, object]) -> None:
    """Validate only rules that can be frozen before training exists."""
    preprocessing = contract.get("preprocessing")
    if not isinstance(preprocessing, Mapping) or preprocessing.get("fit_rows") != (
        "train P_total union U"
    ):
        raise ValueError("invalid preprocessing fit scope")
    if preprocessing.get("validation_or_test_statistics") != "forbidden":
        raise ValueError("invalid preprocessing validation/test rule")
    oof = contract.get("oof")
    if not isinstance(oof, Mapping) or oof.get("coverage") != (
        "each grid_id exactly once across outer-test folds"
    ):
        raise ValueError("invalid OOF coverage contract")
    if oof.get("E0_prediction_materialization") != "forbidden":
        raise ValueError("E0 must not materialize OOF predictions")
    task_key = contract.get("task_key")
    expected_oof_schema = [
        *(task_key if isinstance(task_key, list) else []),
        "bank_id",
        "grid_id",
        "sample_role",
        "score",
        "status",
    ]
    if oof.get("schema") != expected_oof_schema:
        raise ValueError("invalid OOF schema contract")
    bootstrap = contract.get("bootstrap")
    expected_pairing = ["city", "budget", "trajectory", "seed", "outer_block", "bank_id"]
    if not isinstance(bootstrap, Mapping) or bootstrap.get("replicates") != 10_000:
        raise ValueError("invalid bootstrap replicate contract")
    if bootstrap.get("paired_by") != expected_pairing:
        raise ValueError("invalid bootstrap pairing contract")
    if bootstrap.get("fixed_designs") != ["evaluation_banks", "G5_realizations"]:
        raise ValueError("invalid bootstrap fixed-design contract")


def _record(root: Path, path: Path, category: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "bytes": path.stat().st_size,
        "category": category,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }


def _bundle(records: Iterable[Mapping[str, object]]) -> str:
    values = [
        {"bytes": item["bytes"], "path": item["path"], "sha256": item["sha256"]} for item in records
    ]
    values.sort(key=lambda item: str(item["path"]))
    return sha256_bytes(canonical_json_bytes(values))


def _raw_records(root: Path, expected_sources: set[str]) -> list[dict[str, object]]:
    manifests = (
        root / "artifacts" / "manifests" / "raw_source_manifest.parquet",
        root / "artifacts" / "manifests" / "glo30_dsm_tile_manifest.parquet",
    )
    frames = [pd.read_parquet(path) for path in manifests]
    table = pd.concat(frames, ignore_index=True)
    required = {"source_id", "relative_path", "bytes", "sha256", "privacy_status"}
    if not required.issubset(table.columns):
        raise ValueError("source manifest schema is incomplete")
    if table["relative_path"].duplicated().any():
        raise ValueError("source manifest paths must be unique")
    if set(table["source_id"]) != expected_sources:
        raise ValueError("source manifest does not cover all seven sources")
    raw_root = (root / "data" / "raw").resolve()
    records = []
    for row in table.sort_values("relative_path", kind="stable").itertuples(index=False):
        path = (raw_root / row.relative_path).resolve()
        if raw_root not in path.parents:
            raise ValueError(f"source path escapes data/raw: {row.relative_path}")
        record = _record(root, path, "source")
        if record["bytes"] != int(row.bytes) or record["sha256"] != row.sha256:
            raise ValueError(f"source manifest mismatch: {row.relative_path}")
        records.append(record)
    records.extend(_record(root, path, "source") for path in manifests)
    for path in sorted((root / "artifacts" / "manifests").glob("*.parquet")):
        if path not in manifests:
            records.append(_record(root, path, "source"))
    for name in SOURCE_REPORTS:
        records.append(_record(root, root / "artifacts" / "reports" / name, "source"))
    return records


def _parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _validate_interface_columns(
    path: Path,
    expected: list[str],
    interface: str,
    forbidden: Iterable[str] = (),
) -> None:
    columns = _parquet_columns(path)
    if columns != expected:
        raise ValueError(f"invalid {interface} schema: {path}")
    leaked = set(columns).intersection(forbidden)
    if leaked:
        raise ValueError(f"forbidden columns entered {interface}: {sorted(leaked)}")


def _validate_performance_report(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report must contain an object: {path}")
    if "performance_blind" in value and value["performance_blind"] is not True:
        raise ValueError(f"report is not performance-blind: {path}")
    visible = ("predictions_written", "metrics_written", "figures_written")
    if any(value.get(key) is True for key in visible):
        raise ValueError(f"performance output is visible during E0: {path}")
    if value.get("outer_test_used_for_selection") is True:
        raise ValueError(f"outer-test selection is visible during E0: {path}")


def _validate_features(city_root: Path, grid_ids: set[str]) -> dict[str, object]:
    path = city_root / "features_18.parquet"
    expected = ["grid_id", *EXPECTED_FEATURE_NAMES, "geometry"]
    if _parquet_columns(path) != expected:
        raise ValueError(f"invalid 18-feature schema for {city_root.name}")
    if FORBIDDEN_MODEL_COLUMNS.intersection(expected):
        raise ValueError("forbidden column entered the feature interface")
    features = pd.read_parquet(path, columns=["grid_id", *EXPECTED_FEATURE_NAMES])
    if features["grid_id"].duplicated().any() or set(features["grid_id"]) != grid_ids:
        raise ValueError(f"feature/grid mismatch for {city_root.name}")
    ranges = {}
    missing = {}
    for name in EXPECTED_FEATURE_NAMES:
        values = features[name]
        finite = values.dropna()
        ranges[name] = [float(finite.min()), float(finite.max())] if len(finite) else [None, None]
        missing[name] = int(values.isna().sum())
    return {"missing": missing, "ranges": ranges}


def _validate_outer(city_root: Path, grid_ids: set[str]) -> None:
    path = city_root / "evaluation_design" / "outer_partitions.parquet"
    outer = pd.read_parquet(path, columns=["outer_fold", "grid_id", "partition"])
    if set(outer["partition"]) != {"train", "buffer", "test"}:
        raise ValueError("outer partitions changed")
    if outer.duplicated(["outer_fold", "grid_id"]).any():
        raise ValueError("split leakage: duplicate outer assignment")
    if set(outer["outer_fold"]) != set(range(5)):
        raise ValueError("outer folds must be 0 through 4")
    if not all(set(frame["grid_id"]) == grid_ids for _, frame in outer.groupby("outer_fold")):
        raise ValueError("outer partition coverage is incomplete")
    test_counts = outer.loc[outer["partition"].eq("test")].groupby("grid_id").size()
    if len(test_counts) != len(grid_ids) or not test_counts.eq(1).all():
        raise ValueError("OOF static coverage requires one outer-test fold per node")


def _validate_reveals(city_root: Path) -> None:
    reveals = pd.read_parquet(city_root / "evaluation_design" / "reveals.parquet")
    if FORBIDDEN_MODEL_COLUMNS.intersection(reveals.columns):
        raise ValueError("forbidden column entered reveal design")
    key = ["label_layer", "outer_fold", "budget", "trajectory_id"]
    for _, frame in reveals.groupby(key, dropna=False):
        if frame["grid_id"].duplicated().any() or sorted(frame["reveal_rank"]) != list(
            range(1, len(frame) + 1)
        ):
            raise ValueError("invalid reveal ranks")
    strict = reveals.loc[
        reveals["label_layer"].eq("STRICT-LOC") & reveals["budget"].isin(["20", "50", "100", "200"])
    ]
    for (_, outer_fold, trajectory_id), frame in strict.groupby(
        ["label_layer", "outer_fold", "trajectory_id"]
    ):
        by_budget = {
            int(budget): set(group["grid_id"]) for budget, group in frame.groupby("budget")
        }
        ordered = sorted(by_budget)
        for lower, upper in pairwise(ordered):
            if not by_budget[lower].issubset(by_budget[upper]):
                raise ValueError(
                    f"reveals are not nested for fold {outer_fold} trajectory {trajectory_id}"
                )


def _validate_u_and_banks(city_root: Path) -> None:
    design_root = city_root / "evaluation_design"
    samples = pd.read_parquet(design_root / "u_samples.parquet")
    u_key = [
        "scope",
        "label_layer",
        "outer_fold",
        "budget",
        "trajectory_id",
        "inner_fold",
    ]
    sizes = samples.groupby(u_key, dropna=False).size()
    if not sizes.eq(2000).all() or samples.duplicated([*u_key, "grid_id"]).any():
        raise ValueError("U samples must contain 2,000 unique nodes per task")
    banks = pd.read_parquet(design_root / "evaluation_banks.parquet")
    bank_key = [*u_key, "bank_id"]
    if banks.duplicated([*bank_key, "grid_id"]).any():
        raise ValueError("evaluation banks sample a node more than once")
    counts = banks.groupby([*bank_key, "sample_role"], dropna=False).size().unstack(fill_value=0)
    if not counts["background"].eq(10 * counts["positive"]).all():
        raise ValueError("evaluation banks must preserve 1:10 pairing")


def _validate_graphs(city_root: Path) -> None:
    graph_root = city_root / "graphs"
    base = pd.read_parquet(graph_root / "base_edges.parquet")
    g2 = base.loc[base["graph_id"].eq("G2")]
    if not g2.groupby("source_grid_id").size().le(1).all():
        raise ValueError("G2 outdegree exceeds one")
    g4 = set(
        base.loc[base["graph_id"].eq("G4"), ["source_grid_id", "target_grid_id"]].itertuples(
            index=False, name=None
        )
    )
    if g4 != {
        (target, source)
        for source, target in g2[["source_grid_id", "target_grid_id"]].itertuples(
            index=False, name=None
        )
    }:
        raise ValueError("G4 is not the exact G2 reverse")
    outer = pd.read_parquet(graph_root / "outer_edges.parquet")
    if not set(outer["partition"]).issubset({"train", "test"}):
        raise ValueError("outer graph contains buffer edges")
    partitions = pd.read_parquet(
        city_root / "evaluation_design" / "outer_partitions.parquet",
        columns=["outer_fold", "grid_id", "partition"],
    )
    for outer_fold, edges in outer.groupby("outer_fold"):
        lookup = partitions.loc[partitions["outer_fold"].eq(outer_fold)].set_index("grid_id")[
            "partition"
        ]
        source = edges["source_grid_id"].map(lookup)
        target = edges["target_grid_id"].map(lookup)
        if source.isna().any() or target.isna().any() or not source.eq(target).all():
            raise ValueError("outer graph crosses partitions")


def _status_summary(root: Path) -> dict[str, object]:
    statuses = []
    design = json.loads(
        (root / "artifacts" / "reports" / "evaluation_design.json").read_text(encoding="utf-8")
    )
    for city in design["cities"]:
        statuses.extend(item["status"] for item in city["statuses"])
    for city_id in EXPECTED_CITIES:
        index = pd.read_parquet(
            root / "data" / "interim" / city_id / "graphs" / "graph_index.parquet",
            columns=["status"],
        )
        statuses.extend(index["status"])
    invalid = set(statuses) - ALLOWED_STATUSES
    if invalid:
        raise ValueError(f"failed or invalid feasibility status: {sorted(invalid)}")
    counts = Counter(statuses)
    return {name: int(counts.get(name, 0)) for name in ("success", "not-estimable", "failed")}


def _artifact_records(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    protocol = load_protocol(root / "configs")
    label_schema = _read_yaml(root / "configs" / "label_schema.yaml")
    forbidden_design = validate_label_schema_contract(label_schema)
    records = _raw_records(root, set(protocol.sources))
    for path in sorted((root / "configs").rglob("*.yaml")):
        records.append(_record(root, path, "config"))
    for name in CITY_REPORTS:
        records.append(_record(root, root / "artifacts" / "reports" / name, "city_data"))
    records.append(
        _record(root, root / "artifacts" / "reports" / "evaluation_design.json", "design")
    )
    records.append(
        _record(root, root / "artifacts" / "reports" / "data_feasibility_report.md", "design")
    )
    records.append(
        _record(root, root / "artifacts" / "reports" / "graph_counterfactuals.json", "graph")
    )
    city_qa = {}
    for city_id in EXPECTED_CITIES:
        city_root = root / "data" / "interim" / city_id
        grid = pd.read_parquet(city_root / "final_scorable_grid.parquet", columns=["grid_id"])
        if len(grid) != EXPECTED_NODES[city_id] or grid["grid_id"].duplicated().any():
            raise ValueError(f"final scorable node baseline changed for {city_id}")
        grid_ids = set(grid["grid_id"])
        feature_qa = _validate_features(city_root, grid_ids)
        for name, schema in DESIGN_SCHEMAS.items():
            _validate_interface_columns(
                city_root / "evaluation_design" / name,
                schema,
                "design",
                forbidden_design,
            )
        _validate_outer(city_root, grid_ids)
        _validate_reveals(city_root)
        _validate_u_and_banks(city_root)
        _validate_graphs(city_root)
        city_qa[city_id] = {"nodes": len(grid), "features": feature_qa}
        for name in CITY_FILES:
            records.append(_record(root, city_root / name, "city_data"))
        for name in DESIGN_FILES:
            records.append(_record(root, city_root / "evaluation_design" / name, "design"))
        for name in GRAPH_FILES:
            records.append(_record(root, city_root / "graphs" / name, "graph"))
    code_paths = [
        root / "pyproject.toml",
        root / "requirements.txt",
        root / "requirements-train.txt",
    ]
    code_paths.extend(sorted((root / "src").rglob("*.py")))
    code_paths.extend(sorted((root / "tests").rglob("*.py")))
    records.extend(_record(root, path, "code") for path in code_paths)
    return records, city_qa


def _reject_performance_outputs(root: Path) -> None:
    forbidden = set(load_protocol(root / "configs").protocol["e0"]["forbidden_performance_outputs"])
    for path in (root / "artifacts").rglob("*"):
        if path.is_file() and forbidden.intersection(part.lower() for part in path.parts):
            raise ValueError(f"performance output is visible during E0: {path}")
    for path in (root / "artifacts" / "reports").glob("*.json"):
        if path.name != "e0_audit.json":
            _validate_performance_report(path)


def _write_runtime(root: Path, started: float) -> None:
    process = psutil.Process()
    usage = psutil.disk_usage(str(root))
    payload = {
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "peak_rss_bytes": process.memory_info().peak_wset
        if os.name == "nt"
        else process.memory_info().rss,
        "wall_time_seconds": time.perf_counter() - started,
    }
    path = root / "artifacts" / "runtime" / "e0_resources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def audit_e0(root: Path) -> dict[str, object]:
    """Recompute the complete deterministic E0 audit and its bundle hashes."""
    root = root.resolve()
    started = time.perf_counter()
    protocol = load_protocol(root / "configs")
    analysis = _read_yaml(root / "configs" / "analysis_contract.yaml")
    validate_analysis_contract(analysis)
    _reject_performance_outputs(root)
    crowd = protocol.sources["crowdflood_v1"]
    if crowd["privacy_status"] != "processing-only":
        raise ValueError("CrowdFlood must remain processing-only")
    archive = (
        root / "data" / "raw" / "crowdflood_v1" / str(crowd["record_id"]) / str(crowd["filename"])
    )
    crowd_audit = inspect_crowdflood_archive(
        archive,
        {city_id: str(protocol.cities[city_id]["city_event"]) for city_id in EXPECTED_CITIES},
    )
    records, city_qa = _artifact_records(root)
    records.sort(key=lambda item: (str(item["category"]), str(item["path"])))
    bundle_sha256 = {
        category: _bundle(item for item in records if item["category"] == category)
        for category in ("config", "source", "city_data", "design", "graph", "code")
    }
    feasibility = _status_summary(root)
    payload = {
        "base_protocol_id": protocol.protocol["base_id"],
        "bundle_sha256": bundle_sha256,
        "cities": city_qa,
        "crowdflood": {
            "archive_sha256": crowd_audit["archive_sha256"],
            "privacy_status": crowd["privacy_status"],
            "selected_events": crowd_audit["selected_events"],
        },
        "feasibility": feasibility,
        "file_index": records,
        "labels_read_for_design": ["grid_id", "label_layer"],
        "performance_blind": True,
        "predictions_written": False,
        "metrics_written": False,
        "schema_version": 1,
        "status": "success",
    }
    destination = root / "artifacts" / "reports" / "e0_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, destination)
    _write_runtime(root, started)
    return payload


def lock_protocol(root: Path) -> dict[str, object]:
    """Rerun E0 and exclusively create the compact read-only protocol lock."""
    root = root.resolve()
    audit = audit_e0(root)
    audit_path = root / "artifacts" / "reports" / "e0_audit.json"
    bundles = audit["bundle_sha256"]
    lock_inputs = {
        f"{category}_bundle_sha256": bundles[category]
        for category in ("config", "source", "city_data", "design", "graph", "code")
    }
    base_id = str(audit["base_protocol_id"])
    payload = {
        "base_protocol_id": base_id,
        "effective_protocol_id": effective_protocol_id(base_id, lock_inputs),
        "e0_audit": {
            "path": audit_path.relative_to(root).as_posix(),
            "sha256": sha256_file(audit_path),
        },
        **lock_inputs,
        "feasibility": audit["feasibility"],
        "environment_lock_required": True,
        "schema_version": 1,
    }
    destination = root / "artifacts" / "protocol_lock.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload))
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    os.chmod(destination, stat.S_IREAD)
    return payload
