"""Deterministic spatial foundations for both study cities."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

from floodgnn_ceus.io import canonical_json_bytes, sha256_file


def build_admin_aoi(
    boundaries: gpd.GeoDataFrame,
    city: Mapping[str, object],
) -> gpd.GeoDataFrame:
    """Select one frozen administrative shape and project it to the city CRS."""
    if boundaries.crs is None or boundaries.crs.to_epsg() != 4326:
        raise ValueError("geoBoundaries input must use EPSG:4326")
    selector = city.get("geoboundaries")
    if not isinstance(selector, Mapping):
        raise TypeError("city geoboundaries selector must be a mapping")
    city_id = city.get("city_id")
    shape_id = selector.get("shapeID")
    shape_name = selector.get("shapeName")
    crs_epsg = city.get("crs_epsg")
    if (
        not isinstance(city_id, str)
        or not isinstance(shape_id, str)
        or not isinstance(shape_name, str)
    ):
        raise TypeError("city and geoBoundaries identifiers must be strings")
    if not isinstance(crs_epsg, int):
        raise TypeError("city crs_epsg must be an integer")
    selected = boundaries.loc[boundaries["shapeID"] == shape_id]
    if len(selected) != 1:
        raise ValueError(f"shapeID {shape_id} must match exactly one boundary")
    if selected.iloc[0]["shapeName"] != shape_name:
        raise ValueError(f"shapeName mismatch for {shape_id}")
    geometry = selected.geometry.iloc[0]
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"invalid administrative geometry for {city_id}")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"administrative geometry must be polygonal for {city_id}")
    projected = (
        selected[["shapeID", "shapeName", "geometry"]].to_crs(crs_epsg).reset_index(drop=True)
    )
    area_km2 = float(projected.geometry.area.iloc[0] / 1_000_000)
    if not math.isfinite(area_km2) or area_km2 <= 0:
        raise ValueError(f"invalid administrative area for {city_id}")
    projected.insert(0, "city_id", city_id)
    projected.insert(3, "aoi_stage", "administrative")
    projected.insert(4, "is_final_scorable", False)
    projected.insert(5, "area_km2", area_km2)
    return projected


def write_admin_aoi(aoi: gpd.GeoDataFrame, destination: Path) -> Path:
    """Write one administrative AOI as deterministic GeoParquet."""
    if len(aoi) != 1 or aoi.crs is None:
        raise ValueError("administrative AOI must contain one projected geometry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    aoi.to_parquet(partial, index=False, compression="zstd")
    os.replace(partial, destination)
    return destination


def admin_aoi_record(aoi: gpd.GeoDataFrame, path: Path) -> dict[str, object]:
    """Summarize one administrative AOI without serializing geometry coordinates."""
    geometry = aoi.geometry.iloc[0]
    bounds_wgs84 = aoi.to_crs(4326).total_bounds.tolist()
    return {
        "area_km2": float(aoi.loc[0, "area_km2"]),
        "bounds_projected_m": aoi.total_bounds.tolist(),
        "bounds_wgs84": bounds_wgs84,
        "city_id": str(aoi.loc[0, "city_id"]),
        "covers_boundary": bool(geometry.covers(geometry.boundary)),
        "crs": aoi.crs.to_string(),
        "geometry_type": geometry.geom_type,
        "is_final_scorable": bool(aoi.loc[0, "is_final_scorable"]),
        "is_valid": bool(geometry.is_valid),
        "output_sha256": sha256_file(path),
        "shapeID": str(aoi.loc[0, "shapeID"]),
        "shapeName": str(aoi.loc[0, "shapeName"]),
    }


def write_admin_aoi_report(
    destination: Path,
    records: list[dict[str, object]],
    *,
    source_commit: str,
) -> Path:
    """Write compact provenance and geometry checks for administrative AOIs."""
    payload = {
        "aoi_stage": "administrative",
        "cities": records,
        "final_scorable": False,
        "next_gate": ["WorldCover built-up mask", "DEM coverage", "water fraction"],
        "schema_version": 1,
        "source_commit": source_commit,
        "source_id": "geoboundaries_chn_adm2",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, destination)
    return destination


def build_candidate_grid(
    aoi: gpd.GeoDataFrame,
    *,
    resolution_m: int,
) -> gpd.GeoDataFrame:
    """Build the row-major administrative grid using center-point covers."""
    if len(aoi) != 1 or aoi.crs is None or not aoi.crs.is_projected:
        raise ValueError("grid AOI must contain one projected geometry")
    if not isinstance(resolution_m, int) or resolution_m <= 0:
        raise ValueError("grid resolution must be a positive integer")
    city_id = aoi.loc[aoi.index[0], "city_id"]
    if not isinstance(city_id, str):
        raise TypeError("grid city_id must be a string")
    geometry = aoi.geometry.iloc[0]
    min_x, min_y, max_x, max_y = geometry.bounds
    origin_x = math.floor(min_x / resolution_m) * resolution_m
    origin_y = math.ceil(max_y / resolution_m) * resolution_m
    column_count = math.ceil((max_x - origin_x) / resolution_m)
    row_count = math.ceil((origin_y - min_y) / resolution_m)
    rows = np.repeat(np.arange(row_count, dtype=np.int32), column_count)
    columns = np.tile(np.arange(column_count, dtype=np.int32), row_count)
    center_x = origin_x + (columns.astype(np.float64) + 0.5) * resolution_m
    center_y = origin_y - (rows.astype(np.float64) + 0.5) * resolution_m
    centers = shapely.points(center_x, center_y)
    keep = shapely.covers(geometry, centers)
    rows = rows[keep]
    columns = columns[keep]
    center_x = center_x[keep]
    center_y = center_y[keep]
    cells = shapely.box(
        center_x - resolution_m / 2,
        center_y - resolution_m / 2,
        center_x + resolution_m / 2,
        center_y + resolution_m / 2,
    )
    grid_ids = [f"{city_id}-r{row:06d}-c{column:06d}" for row, column in zip(rows, columns)]
    return gpd.GeoDataFrame(
        {
            "city_id": city_id,
            "grid_id": grid_ids,
            "row": rows,
            "column": columns,
            "center_x": center_x,
            "center_y": center_y,
            "grid_stage": "administrative_candidate",
            "is_final_scorable": False,
        },
        geometry=cells,
        crs=aoi.crs,
    )


def write_candidate_grid(grid: gpd.GeoDataFrame, destination: Path) -> Path:
    """Write a row-major administrative candidate grid as GeoParquet."""
    if grid.empty or grid.crs is None:
        raise ValueError("candidate grid must be nonempty and projected")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    grid.to_parquet(partial, index=False, compression="zstd", row_group_size=65_536)
    os.replace(partial, destination)
    return destination


def candidate_grid_record(
    grid: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    path: Path,
    *,
    resolution_m: int,
) -> dict[str, object]:
    """Summarize grid alignment, coverage, and reproducibility."""
    min_x, _, _, max_y = aoi.geometry.iloc[0].bounds
    origin_x = math.floor(min_x / resolution_m) * resolution_m
    origin_y = math.ceil(max_y / resolution_m) * resolution_m
    centers = shapely.points(grid["center_x"].to_numpy(), grid["center_y"].to_numpy())
    return {
        "all_centers_covered": bool(shapely.covers(aoi.geometry.iloc[0], centers).all()),
        "city_id": str(grid["city_id"].iloc[0]),
        "column_direction": "east",
        "crs": grid.crs.to_string(),
        "grid_stage": "administrative_candidate",
        "is_final_scorable": False,
        "node_count": len(grid),
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "output_sha256": sha256_file(path),
        "resolution_m": resolution_m,
        "row_direction": "south",
    }


def write_candidate_grid_report(
    destination: Path,
    records: list[dict[str, object]],
) -> Path:
    """Write compact alignment and hash evidence for both candidate grids."""
    payload = {
        "cities": records,
        "final_scorable": False,
        "grid_stage": "administrative_candidate",
        "schema_version": 1,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, destination)
    return destination
