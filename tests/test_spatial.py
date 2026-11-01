from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.spatial import (
    admin_aoi_record,
    build_admin_aoi,
    build_candidate_grid,
    candidate_grid_record,
    write_admin_aoi,
    write_admin_aoi_report,
    write_candidate_grid,
    write_candidate_grid_report,
)


def city_config() -> dict[str, object]:
    return {
        "city_id": "test_city",
        "crs_epsg": 32631,
        "geoboundaries": {"shapeID": "target", "shapeName": "Target"},
    }


def boundaries() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "shapeID": ["other", "target"],
            "shapeName": ["Other", "Target"],
            "geometry": [
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            ],
        },
        crs="EPSG:4326",
    )


def test_build_admin_aoi_selects_exact_shape_and_local_crs() -> None:
    aoi = build_admin_aoi(boundaries(), city_config())

    assert list(aoi.columns) == [
        "city_id",
        "shapeID",
        "shapeName",
        "aoi_stage",
        "is_final_scorable",
        "area_km2",
        "geometry",
    ]
    assert aoi.crs.to_epsg() == 32631
    assert aoi.loc[0, "shapeID"] == "target"
    assert aoi.loc[0, "aoi_stage"] == "administrative"
    assert not bool(aoi.loc[0, "is_final_scorable"])
    assert aoi.loc[0, "area_km2"] > 0
    assert aoi.geometry.iloc[0].is_valid


def test_build_admin_aoi_requires_one_exact_valid_shape() -> None:
    duplicate = gpd.GeoDataFrame(
        {
            "shapeID": ["target", "target"],
            "shapeName": ["Target", "Target"],
            "geometry": boundaries().geometry,
        },
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="exactly one"):
        build_admin_aoi(duplicate, city_config())


def test_write_admin_aoi_is_reproducible(tmp_path: Path) -> None:
    aoi = build_admin_aoi(boundaries(), city_config())
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"

    write_admin_aoi(aoi, first)
    write_admin_aoi(aoi, second)

    assert sha256_file(first) == sha256_file(second)
    restored = gpd.read_parquet(first)
    assert restored.crs == aoi.crs
    assert restored.geometry.iloc[0].equals_exact(aoi.geometry.iloc[0], tolerance=0)
    record = admin_aoi_record(aoi, first)
    assert record["covers_boundary"] is True
    assert record["crs"] == "EPSG:32631"
    assert record["output_sha256"] == sha256_file(first)
    assert record["is_final_scorable"] is False
    report = tmp_path / "admin_aoi.json"
    write_admin_aoi_report(report, [record], source_commit="abc123")
    assert '"source_commit":"abc123"' in report.read_text(encoding="utf-8")


def test_candidate_grid_uses_aligned_origin_and_boundary_inclusive_centers() -> None:
    aoi = gpd.GeoDataFrame(
        {
            "city_id": ["test_city"],
            "geometry": [Polygon([(100, 100), (300, 100), (300, 300), (100, 300)])],
        },
        crs="EPSG:32631",
    )

    grid = build_candidate_grid(aoi, resolution_m=200)

    assert grid[["row", "column"]].to_records(index=False).tolist() == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert grid["grid_id"].tolist() == [
        "test_city-r000000-c000000",
        "test_city-r000000-c000001",
        "test_city-r000001-c000000",
        "test_city-r000001-c000001",
    ]
    assert grid[["center_x", "center_y"]].to_records(index=False).tolist() == [
        (100.0, 300.0),
        (300.0, 300.0),
        (100.0, 100.0),
        (300.0, 100.0),
    ]
    assert grid.geometry.area.eq(40_000).all()
    assert grid["grid_stage"].eq("administrative_candidate").all()
    assert not grid["is_final_scorable"].any()


def test_candidate_grid_write_and_record_are_reproducible(tmp_path: Path) -> None:
    aoi = gpd.GeoDataFrame(
        {"city_id": ["test_city"], "geometry": [Polygon([(0, 0), (400, 0), (400, 400)])]},
        crs="EPSG:32631",
    )
    grid = build_candidate_grid(aoi, resolution_m=200)
    first = tmp_path / "grid-first.parquet"
    second = tmp_path / "grid-second.parquet"

    write_candidate_grid(grid, first)
    write_candidate_grid(grid, second)

    assert sha256_file(first) == sha256_file(second)
    record = candidate_grid_record(grid, aoi, first, resolution_m=200)
    assert record["all_centers_covered"] is True
    assert record["node_count"] == 3
    assert record["origin_x"] == 0.0
    assert record["origin_y"] == 400.0
    report = tmp_path / "candidate_grid.json"
    write_candidate_grid_report(report, [record])
    assert '"grid_stage":"administrative_candidate"' in report.read_text(encoding="utf-8")
