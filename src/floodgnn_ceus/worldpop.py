"""Download and aggregate the fixed WorldPop population rasters."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import requests
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file
from floodgnn_ceus.protocol import LoadedProtocol, load_protocol
from floodgnn_ceus.sources import (
    RAW_SOURCE_COLUMNS,
    RAW_SOURCE_SCHEMA,
    SourceRecord,
    build_raw_source_manifest,
    download_http,
    ensure_free_space,
    write_source_receipt,
)

RELEASE_STATEMENT = (
    "https://data.worldpop.org/repo/prj/Global_2015_2030/R2024B/doc/"
    "Global2_Release_Statement_R2024B_v1.pdf"
)
EFFECTIVE_LICENSE = "ODbL-1.0 (WorldPop general license CC-BY-4.0)"


def population_feature(
    population_count: np.ndarray,
    valid_fraction: np.ndarray,
    *,
    cell_area_km2: float,
    minimum_coverage: float,
) -> np.ndarray:
    """Return log1p population density after the fixed valid-area gate."""
    population = np.asarray(population_count, dtype=np.float64)
    coverage = np.asarray(valid_fraction, dtype=np.float64)
    if (
        population.shape != coverage.shape
        or population.ndim != 2
        or cell_area_km2 <= 0
        or not 0 <= minimum_coverage <= 1
    ):
        raise ValueError("population feature inputs are invalid")
    result = np.full(population.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(population)
        & (population >= 0)
        & np.isfinite(coverage)
        & (coverage >= minimum_coverage)
        & (coverage <= 1)
    )
    density = np.zeros(population.shape, dtype=np.float64)
    np.divide(
        population,
        coverage * cell_area_km2,
        out=density,
        where=valid,
    )
    result[valid] = np.log1p(density[valid]).astype(np.float32)
    return result


def _source_window(
    source: rasterio.DatasetReader, bounds: tuple[float, ...], dst_crs: object
) -> Window:
    source_bounds = transform_bounds(dst_crs, source.crs, *bounds, densify_pts=21)
    raw = from_bounds(*source_bounds, transform=source.transform)
    column_start = max(0, math.floor(raw.col_off) - 1)
    row_start = max(0, math.floor(raw.row_off) - 1)
    column_stop = min(source.width, math.ceil(raw.col_off + raw.width) + 1)
    row_stop = min(source.height, math.ceil(raw.row_off + raw.height) + 1)
    if column_stop <= column_start or row_stop <= row_start:
        raise ValueError("WorldPop raster does not overlap the target grid")
    return Window(
        column_start,
        row_start,
        column_stop - column_start,
        row_stop - row_start,
    )


def reproject_population_counts(
    source_path: Path,
    *,
    dst_crs: object,
    bounds: tuple[float, float, float, float],
    resolution_m: int,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Area-weight population counts and validity onto an aligned projected grid."""
    left, bottom, right, top = bounds
    if resolution_m <= 0 or right <= left or top <= bottom:
        raise ValueError("WorldPop target grid is invalid")
    width = round((right - left) / resolution_m)
    height = round((top - bottom) / resolution_m)
    transform = from_origin(left, top, resolution_m, resolution_m)
    population = np.zeros((height, width), dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(source_path) as source:
        if source.crs is None or source.count != 1:
            raise ValueError("WorldPop source raster is invalid")
        window = _source_window(source, bounds, dst_crs)
        values = source.read(1, window=window).astype(np.float64)
        valid = (source.read_masks(1, window=window) > 0) & np.isfinite(values) & (values >= 0)
        values[~valid] = 0
        source_transform = source.window_transform(window)
        reproject(
            values,
            population,
            src_transform=source_transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=None,
            dst_nodata=0,
            resampling=Resampling.sum,
            init_dest_nodata=True,
            num_threads=1,
        )
        reproject(
            valid.astype(np.float32),
            coverage,
            src_transform=source_transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=None,
            dst_nodata=0,
            resampling=Resampling.average,
            init_dest_nodata=True,
            num_threads=1,
        )
    np.maximum(population, 0, out=population)
    np.clip(coverage, 0, 1, out=coverage)
    return population, coverage, transform


def _existing_record(destination: Path, raw_root: Path, year: str) -> SourceRecord | None:
    receipt = destination.with_name(f"{destination.name}.source.json")
    if not receipt.exists():
        return None
    try:
        record = SourceRecord(**json.loads(receipt.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("existing WorldPop source receipt is invalid") from None
    if (
        record.source_id != "worldpop_r2024b"
        or record.asset_key != year
        or record.relative_path != destination.relative_to(raw_root).as_posix()
    ):
        raise ValueError("existing WorldPop source receipt does not match its asset")
    return record


def _write_manifest(records: list[SourceRecord], destination: Path) -> Path:
    rows = [
        {name: getattr(record, name) for name in RAW_SOURCE_COLUMNS}
        for record in sorted(records, key=lambda item: item.asset_key)
    ]
    table = pa.Table.from_pylist(rows, schema=RAW_SOURCE_SCHEMA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    pq.write_table(table, partial, compression="zstd", use_dictionary=False, write_statistics=False)
    os.replace(partial, destination)
    return destination


def fetch_worldpop(
    root: Path,
    session: requests.Session,
) -> tuple[dict[str, SourceRecord], dict[str, str | None], Path]:
    """Download and register the two frozen China population-count assets."""
    protocol = load_protocol(root / "configs")
    source = protocol.sources["worldpop_r2024b"]
    assets = source["assets"]
    raw_root = root / "data" / "raw"
    ensure_free_space(raw_root, sum(int(asset["bytes"]) for asset in assets.values()))
    records: dict[str, SourceRecord] = {}
    etags: dict[str, str | None] = {}
    for year in sorted(assets):
        asset = assets[year]
        url = str(asset["url"])
        filename = Path(urlsplit(url).path).name
        destination = (
            raw_root
            / "worldpop_r2024b"
            / str(source["release"])
            / str(source["version"])
            / "100m"
            / "unconstrained"
            / filename
        )
        existing = _existing_record(destination, raw_root, year)
        download = download_http(
            session,
            url,
            destination,
            asset_identity=f"worldpop:{source['release']}:{source['version']}:{year}:CHN:UC:100m",
            expected_size=int(asset["bytes"]),
            expected_checksums={"sha256": existing.sha256} if existing else None,
            timeout=(20, 300),
        )
        retrieved_at = (
            existing.retrieved_at
            if existing
            else datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
        record = SourceRecord(
            source_id="worldpop_r2024b",
            city_id="all",
            asset_key=year,
            provider=str(source["provider"]),
            version_or_release=f"{source['release']}/{source['version']}",
            source_url_or_doi=url,
            license=EFFECTIVE_LICENSE,
            privacy_status=str(source["privacy_status"]),
            reference_date_or_period=year,
            relative_path=destination.relative_to(raw_root).as_posix(),
            retrieved_at=retrieved_at,
            bytes=download.bytes_written,
            sha256=download.sha256,
            request_sha256=sha256_bytes(canonical_json_bytes({"method": "GET", "url": url})),
        )
        write_source_receipt(record, raw_root, protocol)
        records[year] = record
        etags[year] = download.etag
    manifest = _write_manifest(
        list(records.values()),
        root / "artifacts" / "manifests" / "worldpop_asset_manifest.parquet",
    )
    build_raw_source_manifest(raw_root, root / "artifacts", protocol)
    return records, etags, manifest


def _write_geoparquet(frame: gpd.GeoDataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    frame.to_parquet(partial, index=False, compression="zstd", row_group_size=65_536)
    os.replace(partial, destination)
    return destination


def _write_raster(
    destination: Path,
    arrays: tuple[np.ndarray, ...],
    *,
    crs: object,
    transform: object,
    descriptions: tuple[str, ...],
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    with rasterio.open(
        partial,
        "w",
        driver="GTiff",
        width=arrays[0].shape[1],
        height=arrays[0].shape[0],
        count=len(arrays),
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999,
        compress="DEFLATE",
        zlevel=6,
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as target:
        for band, (array, description) in enumerate(zip(arrays, descriptions, strict=True), 1):
            target.write(array.astype(np.float32, copy=False), band)
            target.set_band_description(band, description)
        target.update_tags(source_id="worldpop_r2024b", resampling="sum_and_valid_area")
    os.replace(partial, destination)
    return destination


def _file_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _process_city(
    root: Path,
    city_id: str,
    year: str,
    record: SourceRecord,
    protocol: LoadedProtocol,
) -> dict[str, object]:
    city_root = root / "data" / "interim" / city_id
    grid = gpd.read_parquet(city_root / "candidate_grid.parquet")
    nodes = gpd.read_parquet(city_root / "candidate_urban_domain.parquet")
    grid = grid.sort_values(["row", "column"], kind="stable").reset_index(drop=True)
    nodes = nodes.sort_values(["row", "column"], kind="stable").reset_index(drop=True)
    resolution = int(protocol.protocol["grid"]["resolution_m"])
    bounds = tuple(float(value) for value in grid.total_bounds)
    source_path = root / "data" / "raw" / record.relative_path
    population, coverage, transform = reproject_population_counts(
        source_path,
        dst_crs=grid.crs,
        bounds=bounds,
        resolution_m=resolution,
    )
    all_features = population_feature(
        population,
        coverage,
        cell_area_km2=(resolution / 1000) ** 2,
        minimum_coverage=0.5,
    )
    left, _, _, top = bounds[0], bounds[1], bounds[2], bounds[3]
    rows = np.floor((top - nodes["center_y"].to_numpy()) / resolution).astype(int)
    columns = np.floor((nodes["center_x"].to_numpy() - left) / resolution).astype(int)
    feature = nodes[
        ["city_id", "grid_id", "row", "column", "center_x", "center_y", "geometry"]
    ].copy()
    feature["log1p_population_density"] = all_features[rows, columns]
    feature["worldpop_valid_fraction"] = coverage[rows, columns]
    feature["feature_stage"] = "worldcover_candidate"
    feature["is_final_scorable"] = False
    feature_path = _write_geoparquet(feature, city_root / "worldpop_features.parquet")
    nodata = np.float32(-9999)
    raster_path = _write_raster(
        city_root / "worldpop_200m.tif",
        (
            np.where(coverage > 0, population, nodata),
            np.where(coverage > 0, coverage, nodata),
            np.where(np.isfinite(all_features), all_features, nodata),
        ),
        crs=grid.crs,
        transform=transform,
        descriptions=(
            "population_count",
            "worldpop_valid_fraction",
            "log1p_population_density",
        ),
    )
    finite = np.isfinite(feature["log1p_population_density"].to_numpy())
    values = feature.loc[finite, "log1p_population_density"].to_numpy()
    return {
        "city_id": city_id,
        "reference_year": int(year),
        "candidate_nodes": len(feature),
        "feature_complete_nodes": int(np.count_nonzero(finite)),
        "coverage_below_0_5": int(np.count_nonzero(~finite)),
        "valid_fraction_range": [
            float(feature["worldpop_valid_fraction"].min()),
            float(feature["worldpop_valid_fraction"].max()),
        ],
        "log1p_population_density_range": [float(values.min()), float(values.max())],
        "outputs": [_file_record(root, feature_path), _file_record(root, raster_path)],
    }


def prepare_worldpop(root: Path, session: requests.Session) -> dict[str, object]:
    """Download, verify, and prepare population features for both candidate domains."""
    protocol = load_protocol(root / "configs")
    records, etags, manifest = fetch_worldpop(root, session)
    city_years = {
        city_id: str(protocol.cities[city_id]["worldpop_year"])
        for city_id in ("zhengzhou", "beijing")
    }
    cities = [
        _process_city(root, city_id, year, records[year], protocol)
        for city_id, year in city_years.items()
    ]
    source = protocol.sources["worldpop_r2024b"]
    assets = []
    for year, record in sorted(records.items()):
        configured = source["assets"][year]
        receipt = root / "data" / "raw" / f"{record.relative_path}.source.json"
        assets.append(
            {
                "year": int(year),
                "path": f"data/raw/{record.relative_path}",
                "bytes": record.bytes,
                "sha256": record.sha256,
                "etag": etags[year],
                "etag_is_checksum": False,
                "etag_matches_frozen_observation": etags[year] == configured["observed_etag"],
                "source_receipt": _file_record(root, receipt),
            }
        )
    payload = {
        "schema_version": 1,
        "source_id": "worldpop_r2024b",
        "release": source["release"],
        "version": source["version"],
        "status": source["status"],
        "privacy_status": source["privacy_status"],
        "license": {
            "worldpop_general": source["general_license"],
            "effective_for_derived_outputs": "ODbL-1.0",
            "reason": "R2024B uses Microsoft Building Footprints; WorldPop applies ODbL to such derived datasets.",
            "release_statement": RELEASE_STATEMENT,
        },
        "processing": {
            "source_units": source["units"],
            "count_resampling": "area-weighted sum",
            "density_units": "persons km-2 over valid target area",
            "minimum_valid_fraction": 0.5,
            "transform": "log1p",
        },
        "aoi_stage": "worldcover_candidate",
        "is_final_scorable": False,
        "assets": assets,
        "manifest": _file_record(root, manifest),
        "cities": cities,
    }
    report = root / "artifacts" / "reports" / "worldpop.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    partial = report.with_name(f"{report.name}.part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, report)
    return {
        "assets": len(records),
        "cities": {city["city_id"]: city["candidate_nodes"] for city in cities},
        "manifest": str(manifest),
        "report": str(report),
        "total_bytes": sum(record.bytes for record in records.values()),
    }
