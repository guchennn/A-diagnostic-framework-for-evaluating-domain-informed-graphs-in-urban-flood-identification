"""STRICT-LOC and ALL-LOC label preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely import from_wkt
from shapely.errors import GEOSException
from shapely.geometry import Point

from floodgnn_ceus.audit import CROWD_FLOOD_FIELDS
from floodgnn_ceus.io import canonical_json_bytes, sha256_file

GEOMETRY_TOLERANCE_DEGREES = 1e-6
PARSER_VERSION = "crowdflood-labels-v1"
QC_PRIORITY = (
    "invalid_coordinate",
    "geometry_invalid",
    "geometry_mismatch",
    "outside_admin",
    "outside_AOI",
    "invalid_pt_type_1",
    "not_all_type_1",
    "duplicate_id",
    "duplicate_coordinate",
    "duplicate_grid",
)
_INTEGER_LIST = re.compile(r"^\[\s*[+-]?\d+(?:\s*,\s*[+-]?\d+)*\s*\]$")


def parse_pt_type_1(value: str) -> tuple[list[int] | None, str]:
    """Parse only JSON integers or the frozen integer-list fallback."""
    text = value.strip()
    if not text:
        return None, "invalid"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if not _INTEGER_LIST.fullmatch(text):
            return None, "invalid"
        parsed = [int(item.strip()) for item in text[1:-1].split(",")]
    if type(parsed) is int:
        parsed = [parsed]
    if type(parsed) is not list or not parsed or any(type(item) is not int for item in parsed):
        return None, "invalid"
    return parsed, "valid"


def parse_geometry(
    value: str,
    longitude: float | None,
    latitude: float | None,
) -> str:
    """Validate an optional WGS84 POINT against authoritative coordinates."""
    if not value.strip():
        return "missing"
    try:
        geometry = from_wkt(value)
    except GEOSException:
        return "invalid"
    if geometry.geom_type != "Point" or geometry.is_empty or geometry.has_z:
        return "invalid"
    if longitude is None or latitude is None:
        return "valid"
    if (
        abs(geometry.x - longitude) > GEOMETRY_TOLERANCE_DEGREES
        or abs(geometry.y - latitude) > GEOMETRY_TOLERANCE_DEGREES
    ):
        return "mismatch"
    return "valid"


def _parse_coordinates(row: Mapping[str, str]) -> tuple[float | None, float | None]:
    try:
        longitude = float(row["wgs_lng"])
        latitude = float(row["wgs_lat"])
    except (TypeError, ValueError):
        return None, None
    if (
        not math.isfinite(longitude)
        or not math.isfinite(latitude)
        or not -180 <= longitude <= 180
        or not -90 <= latitude <= 90
    ):
        return None, None
    return longitude, latitude


def _first_reason(flags: set[str]) -> str | None:
    return next((reason for reason in QC_PRIORITY if reason in flags), None)


def _deduplicate(
    records: list[dict[str, object]],
    candidates: list[int],
) -> tuple[set[int], dict[str, list[int]], dict[int, set[str]]]:
    flags = {index: set() for index in range(len(records))}
    survivors = candidates
    for reason, field in (("duplicate_id", "raw_id"), ("duplicate_coordinate", "coordinate")):
        seen: set[object] = set()
        retained: list[int] = []
        for index in survivors:
            key = records[index][field]
            if field == "raw_id" and key == "":
                retained.append(index)
            elif key in seen:
                flags[index].add(reason)
            else:
                seen.add(key)
                retained.append(index)
        survivors = retained
    groups: dict[str, list[int]] = {}
    representatives: set[int] = set()
    for index in survivors:
        grid_id = str(records[index]["grid_id"])
        members = groups.setdefault(grid_id, [])
        if members:
            flags[index].add("duplicate_grid")
        else:
            representatives.add(index)
        members.append(index)
    return representatives, groups, flags


def _grid_origin(grid: gpd.GeoDataFrame, resolution_m: int) -> tuple[float, float]:
    origin_x = grid["center_x"] - (grid["column"] + 0.5) * resolution_m
    origin_y = grid["center_y"] + (grid["row"] + 0.5) * resolution_m
    if origin_x.nunique() != 1 or origin_y.nunique() != 1:
        raise ValueError("candidate grid has inconsistent origins")
    return float(origin_x.iloc[0]), float(origin_y.iloc[0])


def build_label_ledgers(
    rows: Sequence[Mapping[str, str]],
    *,
    city_id: str,
    city_event: str,
    source_sha256: str,
    aoi: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    resolution_m: int,
    final_scorable: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse, spatially assign, and deduplicate one fixed city-event file."""
    if len(aoi) != 1 or aoi.crs is None or grid.empty or grid.crs != aoi.crs:
        raise ValueError("AOI and candidate grid must share one projected CRS")
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a full digest")
    origin_x, origin_y = _grid_origin(grid, resolution_m)
    grid_ids = set(grid["grid_id"])
    transformer = Transformer.from_crs(4326, aoi.crs, always_xy=True)
    admin_geometry = aoi.geometry.iloc[0]
    records: list[dict[str, object]] = []
    for source_row_number, row in enumerate(rows, start=1):
        if set(row) != set(CROWD_FLOOD_FIELDS):
            raise ValueError(f"unexpected fields at source row {source_row_number}")
        longitude, latitude = _parse_coordinates(row)
        flags: set[str] = set()
        if longitude is None or latitude is None:
            flags.add("invalid_coordinate")
        geometry_status = parse_geometry(row["geometry"], longitude, latitude)
        if geometry_status == "missing":
            flags.add("geometry_missing")
        elif geometry_status == "invalid":
            flags.add("geometry_invalid")
        elif geometry_status == "mismatch":
            flags.add("geometry_mismatch")
        parsed_type, type_status = parse_pt_type_1(row["pt_type_1"])
        if type_status == "invalid":
            flags.add("invalid_pt_type_1")
        elif not all(item == 1 for item in parsed_type or []):
            flags.add("not_all_type_1")
        grid_id: str | None = None
        admin_covered: bool | None = None
        coordinate: tuple[float, float] | None = None
        if longitude is not None and latitude is not None:
            coordinate = (longitude, latitude)
            x, y = transformer.transform(longitude, latitude)
            admin_covered = bool(admin_geometry.covers(Point(x, y)))
            if not admin_covered:
                flags.add("outside_AOI" if final_scorable else "outside_admin")
            else:
                column = math.floor((x - origin_x) / resolution_m)
                row_index = math.floor((origin_y - y) / resolution_m)
                candidate_id = f"{city_id}-r{row_index:06d}-c{column:06d}"
                if candidate_id in grid_ids:
                    grid_id = candidate_id
                else:
                    flags.add("outside_AOI")
        raw_id = row["id"]
        raw_id_hash = hashlib.sha256(raw_id.encode()).hexdigest() if raw_id else None
        member_row_hash = hashlib.sha256(
            f"{source_sha256}:{source_row_number}".encode()
        ).hexdigest()
        records.append(
            {
                "admin_covered": admin_covered,
                "coordinate": coordinate,
                "coordinate_status": "valid" if coordinate else "invalid",
                "flags": flags,
                "geometry_status": geometry_status,
                "grid_id": grid_id,
                "member_row_hash": member_row_hash,
                "parsed_pt_type_1": parsed_type,
                "pt_type_1_status": type_status,
                "raw_id": raw_id,
                "raw_id_hash": raw_id_hash,
                "source_row_number": source_row_number,
            }
        )
    spatial_reasons = {
        "invalid_coordinate",
        "geometry_invalid",
        "geometry_mismatch",
        "outside_admin",
        "outside_AOI",
    }
    all_candidates = [
        index for index, record in enumerate(records) if not (record["flags"] & spatial_reasons)
    ]
    strict_candidates = [
        index
        for index in all_candidates
        if not records[index]["flags"] & {"invalid_pt_type_1", "not_all_type_1"}
    ]
    strict_included, strict_groups, strict_dedupe = _deduplicate(records, strict_candidates)
    all_included, all_groups, all_dedupe = _deduplicate(records, all_candidates)
    aoi_stage = "final_scorable" if final_scorable else "administrative"
    grid_stage = "final_scorable" if final_scorable else "administrative_candidate"
    qc_rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        common_flags = set(record["flags"])
        strict_flags = common_flags | strict_dedupe[index]
        all_flags = (common_flags - {"invalid_pt_type_1", "not_all_type_1"}) | all_dedupe[index]
        visible_flags = common_flags | strict_dedupe[index] | all_dedupe[index]
        flag_order = {name: position for position, name in enumerate(QC_PRIORITY)}
        qc_rows.append(
            {
                "city_id": city_id,
                "city_event": city_event,
                "source_row_number": record["source_row_number"],
                "member_row_hash": record["member_row_hash"],
                "raw_id_hash": record["raw_id_hash"],
                "coordinate_status": record["coordinate_status"],
                "geometry_status": record["geometry_status"],
                "pt_type_1_status": record["pt_type_1_status"],
                "parsed_pt_type_1": record["parsed_pt_type_1"],
                "admin_covered": record["admin_covered"],
                "grid_id": record["grid_id"],
                "qc_flags": sorted(visible_flags, key=lambda item: flag_order.get(item, -1)),
                "strict_primary_reason": _first_reason(strict_flags - {"geometry_missing"}),
                "all_primary_reason": _first_reason(all_flags - {"geometry_missing"}),
                "strict_included": index in strict_included,
                "all_included": index in all_included,
                "aoi_stage": aoi_stage,
                "grid_stage": grid_stage,
            }
        )
    label_rows: list[dict[str, object]] = []
    for label_layer, groups in (("STRICT-LOC", strict_groups), ("ALL-LOC", all_groups)):
        for grid_id, member_indices in groups.items():
            label_rows.append(
                {
                    "city_id": city_id,
                    "city_event": city_event,
                    "grid_id": grid_id,
                    "label_layer": label_layer,
                    "member_row_hashes": [
                        records[index]["member_row_hash"] for index in member_indices
                    ],
                    "member_count": len(member_indices),
                    "aoi_stage": aoi_stage,
                    "grid_stage": grid_stage,
                }
            )
    return pd.DataFrame(qc_rows), pd.DataFrame(label_rows)


def read_crowdflood_rows(path: Path) -> list[dict[str, str]]:
    """Read one fixed UTF-8 CrowdFlood event without normalizing source values."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CROWD_FLOOD_FIELDS:
            raise ValueError(f"unexpected CrowdFlood fields: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"invalid CrowdFlood row width: {path}")
    return rows


def write_label_ledgers(
    qc: pd.DataFrame,
    labels: pd.DataFrame,
    qc_path: Path,
    label_path: Path,
) -> tuple[Path, Path]:
    """Write restricted row QC and grid labels as deterministic Parquet."""
    forbidden = {
        "id",
        "wgs_lng",
        "wgs_lat",
        "geometry",
        "toponym",
        "SSA",
        "similarity",
        "flood_prob",
    }
    if forbidden & set(qc.columns) or forbidden & set(labels.columns):
        raise ValueError("raw or report-derived fields are forbidden in label outputs")
    for frame, destination in ((qc, qc_path), (labels, label_path)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        frame.to_parquet(
            partial,
            index=False,
            compression="zstd",
            engine="pyarrow",
            row_group_size=65_536,
        )
        os.replace(partial, destination)
    return qc_path, label_path


def _counts(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def label_qc_record(
    qc: pd.DataFrame,
    labels: pd.DataFrame,
    source_path: Path,
    qc_path: Path,
    label_path: Path,
) -> dict[str, object]:
    """Summarize label preparation without row identities or locations."""
    aoi_stage = str(qc["aoi_stage"].iloc[0])
    grid_stage = str(qc["grid_stage"].iloc[0])
    if qc["aoi_stage"].nunique() != 1 or qc["grid_stage"].nunique() != 1:
        raise ValueError("label ledgers must use one spatial stage")
    flag_counts = _counts(qc["qc_flags"].explode().dropna())
    label_counts = _counts(labels["label_layer"])
    strict_grids = set(labels.loc[labels["label_layer"] == "STRICT-LOC", "grid_id"])
    all_grids = set(labels.loc[labels["label_layer"] == "ALL-LOC", "grid_id"])
    return {
        "all_primary_reason_counts": _counts(qc["all_primary_reason"].fillna("included")),
        "aoi_stage": aoi_stage,
        "city_event": str(qc["city_event"].iloc[0]),
        "city_id": str(qc["city_id"].iloc[0]),
        "coverage_predicate": "covers",
        "final_scorable": aoi_stage == "final_scorable" and grid_stage == "final_scorable",
        "geometry_tolerance_degrees": GEOMETRY_TOLERANCE_DEGREES,
        "grid_mapping_tie_break": "east-and-south on internal boundaries",
        "grid_stage": grid_stage,
        "input_bytes": source_path.stat().st_size,
        "input_encoding": "UTF-8 with BOM",
        "input_sha256": sha256_file(source_path),
        "label_cell_counts": label_counts,
        "label_ledger_bytes": label_path.stat().st_size,
        "label_ledger_sha256": sha256_file(label_path),
        "label_sets_identical": strict_grids == all_grids,
        "parser_version": PARSER_VERSION,
        "privacy_status": "processing-only",
        "qc_flag_counts": flag_counts,
        "qc_ledger_bytes": qc_path.stat().st_size,
        "qc_ledger_sha256": sha256_file(qc_path),
        "raw_row_count": len(qc),
        "source_row_number_base": 1,
        "strict_primary_reason_counts": _counts(qc["strict_primary_reason"].fillna("included")),
    }


def write_label_qc_report(
    destination: Path,
    records: list[dict[str, object]],
) -> Path:
    """Write the public aggregate QC ledger report."""
    payload = {
        "cities": records,
        "final_scorable": False,
        "performance_blind": True,
        "privacy_status": "processing-only",
        "raw_values_committed": False,
        "schema_version": 1,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, destination)
    return destination
