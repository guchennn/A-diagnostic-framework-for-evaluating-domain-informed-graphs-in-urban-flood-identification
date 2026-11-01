from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon

from floodgnn_ceus.audit import CROWD_FLOOD_FIELDS
from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.labels import (
    QC_PRIORITY,
    build_label_ledgers,
    label_qc_record,
    parse_geometry,
    parse_pt_type_1,
    read_crowdflood_rows,
    write_label_ledgers,
    write_label_qc_report,
)
from floodgnn_ceus.spatial import build_candidate_grid


def test_pt_type_parser_accepts_only_frozen_integer_forms() -> None:
    assert parse_pt_type_1("[1, 1]") == ([1, 1], "valid")
    assert parse_pt_type_1("1") == ([1], "valid")
    assert parse_pt_type_1("[+1, -2]") == ([1, -2], "valid")
    for value in ("", "true", "1.0", "[1.0]", "__import__('os')", "[1,]"):
        assert parse_pt_type_1(value) == (None, "invalid")


def test_geometry_parser_distinguishes_missing_invalid_and_mismatch() -> None:
    assert parse_geometry("", 116.4, 39.9) == "missing"
    assert parse_geometry("POINT (116.4 39.9)", 116.4, 39.9) == "valid"
    assert parse_geometry("POINT (116.5 39.9)", 116.4, 39.9) == "mismatch"
    assert parse_geometry("LINESTRING (116.4 39.9, 116.5 40)", 116.4, 39.9) == "invalid"


def raw_row(
    raw_id: str,
    longitude: object,
    latitude: object,
    *,
    geometry: str = "",
    pt_type_1: str = "[1]",
) -> dict[str, str]:
    return {
        "id": raw_id,
        "wgs_lng": str(longitude),
        "wgs_lat": str(latitude),
        "geometry": geometry,
        "pt_type_1": pt_type_1,
        "toponym": "restricted place",
        "SSA": "restricted text",
        "similarity": "1",
        "flood_prob": "1",
    }


def wgs84_point(x: float, y: float) -> tuple[float, float]:
    transformer = Transformer.from_crs(3857, 4326, always_xy=True)
    return transformer.transform(x, y)


def point_row(
    raw_id: str,
    x: float,
    y: float,
    *,
    pt_type_1: str = "[1]",
    geometry: str | None = None,
) -> dict[str, str]:
    longitude, latitude = wgs84_point(x, y)
    point_wkt = geometry if geometry is not None else f"POINT ({longitude} {latitude})"
    return raw_row(raw_id, longitude, latitude, geometry=point_wkt, pt_type_1=pt_type_1)


def ledger_fixture() -> tuple[list[Mapping[str, str]], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    aoi = gpd.GeoDataFrame(
        {"city_id": ["test_city"], "geometry": [Polygon([(0, 0), (400, 0), (400, 400), (0, 400)])]},
        crs="EPSG:3857",
    )
    grid = build_candidate_grid(aoi, resolution_m=200)
    grid = grid.loc[grid["grid_id"] != "test_city-r000000-c000001"].reset_index(drop=True)
    first = point_row("same", 100, 100)
    rows = [
        first,
        point_row("same", 140, 140),
        dict(first, id=""),
        point_row("grid-duplicate", 160, 160, geometry=""),
        point_row("all-only", 300, 100, pt_type_1="[2]"),
        raw_row("bad-coordinate", "nan", 0, pt_type_1="true"),
        point_row("mismatch", 100, 300, geometry="POINT (0 0)"),
        point_row("outside", 600, 100),
        point_row("missing-grid", 300, 300),
        point_row("bad-geometry", 100, 300, geometry="LINESTRING (0 0, 1 1)"),
    ]
    return rows, aoi, grid


def test_label_ledgers_apply_qc_priority_and_three_stage_deduplication() -> None:
    rows, aoi, grid = ledger_fixture()

    qc, labels = build_label_ledgers(
        rows,
        city_id="test_city",
        city_event="Test City_2020_1_1",
        source_sha256="a" * 64,
        aoi=aoi,
        grid=grid,
        resolution_m=200,
    )

    by_row = qc.set_index("source_row_number")
    assert by_row.loc[2, "strict_primary_reason"] == "duplicate_id"
    assert by_row.loc[3, "strict_primary_reason"] == "duplicate_coordinate"
    assert by_row.loc[4, "strict_primary_reason"] == "duplicate_grid"
    assert by_row.loc[5, "strict_primary_reason"] == "not_all_type_1"
    assert pd.isna(by_row.loc[5, "all_primary_reason"])
    assert by_row.loc[6, "strict_primary_reason"] == "invalid_coordinate"
    assert by_row.loc[6, "all_primary_reason"] == "invalid_coordinate"
    assert by_row.loc[7, "strict_primary_reason"] == "geometry_mismatch"
    assert by_row.loc[8, "strict_primary_reason"] == "outside_admin"
    assert by_row.loc[9, "strict_primary_reason"] == "outside_AOI"
    assert by_row.loc[10, "strict_primary_reason"] == "geometry_invalid"
    assert qc.loc[qc["source_row_number"] == 4, "geometry_status"].item() == "missing"
    assert qc["strict_included"].sum() == 1
    assert qc["all_included"].sum() == 2
    assert labels.groupby("label_layer").size().to_dict() == {"ALL-LOC": 2, "STRICT-LOC": 1}
    strict = labels.loc[labels["label_layer"] == "STRICT-LOC"].iloc[0]
    assert len(strict["member_row_hashes"]) == 2
    assert set(QC_PRIORITY) == {
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
    }
    assert {"id", "wgs_lng", "wgs_lat", "geometry", "toponym", "SSA"}.isdisjoint(qc.columns)


def test_final_label_remap_unifies_valid_spatial_gate_failures() -> None:
    rows, aoi, grid = ledger_fixture()

    qc, labels = build_label_ledgers(
        rows,
        city_id="test_city",
        city_event="Test City_2020_1_1",
        source_sha256="a" * 64,
        aoi=aoi,
        grid=grid,
        resolution_m=200,
        final_scorable=True,
    )

    by_row = qc.set_index("source_row_number")
    assert by_row.loc[8, "strict_primary_reason"] == "outside_AOI"
    assert by_row.loc[9, "strict_primary_reason"] == "outside_AOI"
    assert qc["aoi_stage"].eq("final_scorable").all()
    assert qc["grid_stage"].eq("final_scorable").all()
    assert labels["aoi_stage"].eq("final_scorable").all()
    assert labels["grid_stage"].eq("final_scorable").all()


def test_label_outputs_are_deterministic_and_reports_are_aggregate(tmp_path: Path) -> None:
    rows, aoi, grid = ledger_fixture()
    qc, labels = build_label_ledgers(
        rows,
        city_id="test_city",
        city_event="Test City_2020_1_1",
        source_sha256="a" * 64,
        aoi=aoi,
        grid=grid,
        resolution_m=200,
    )
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CROWD_FLOOD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    assert len(read_crowdflood_rows(source)) == len(rows)
    first_qc = tmp_path / "first-qc.parquet"
    first_labels = tmp_path / "first-labels.parquet"
    second_qc = tmp_path / "second-qc.parquet"
    second_labels = tmp_path / "second-labels.parquet"

    write_label_ledgers(qc, labels, first_qc, first_labels)
    write_label_ledgers(qc, labels, second_qc, second_labels)

    assert sha256_file(first_qc) == sha256_file(second_qc)
    assert sha256_file(first_labels) == sha256_file(second_labels)
    record = label_qc_record(qc, labels, source, first_qc, first_labels)
    assert record["raw_row_count"] == 10
    assert record["label_cell_counts"] == {"ALL-LOC": 2, "STRICT-LOC": 1}
    report = tmp_path / "label_qc.json"
    write_label_qc_report(report, [record])
    text = report.read_text(encoding="utf-8")
    assert "restricted place" not in text
    assert "restricted text" not in text
    assert '"final_scorable":false' in text
