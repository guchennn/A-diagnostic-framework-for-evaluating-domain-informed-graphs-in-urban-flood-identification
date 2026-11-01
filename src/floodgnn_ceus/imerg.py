"""IMERG inventory, acquisition, and fixed rainfall summaries."""

from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlsplit

import earthaccess
import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file
from floodgnn_ceus.protocol import LoadedProtocol, load_protocol
from floodgnn_ceus.sources import (
    RAW_SOURCE_COLUMNS,
    RAW_SOURCE_SCHEMA,
    SourceRecord,
    build_raw_source_manifest,
    download_http,
    ensure_free_space,
    required_free_bytes,
    write_source_receipt,
)


@dataclass(frozen=True)
class ImergGranule:
    """Public CMR inventory fields for one selected half-hour granule."""

    city_id: str
    filename: str
    granule_ur: str
    url: str
    beginning: str
    ending: str
    bytes: int

    @property
    def midpoint(self) -> datetime:
        start = datetime.fromisoformat(self.beginning)
        end = datetime.fromisoformat(self.ending)
        return start + (end - start + timedelta(milliseconds=1)) / 2


def nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Match a sorted coordinate axis with lower-index tie breaking."""
    coordinates = np.asarray(axis, dtype=np.float64)
    query = np.asarray(values, dtype=np.float64)
    if (
        coordinates.ndim != 1
        or len(coordinates) < 2
        or not np.isfinite(coordinates).all()
        or not np.all(np.diff(coordinates) > 0)
        or not np.isfinite(query).all()
    ):
        raise ValueError("IMERG coordinate matching inputs are invalid")
    right = np.searchsorted(coordinates, query, side="left")
    right = np.clip(right, 1, len(coordinates) - 1)
    left = right - 1
    choose_right = np.abs(query - coordinates[right]) < np.abs(query - coordinates[left])
    return np.where(choose_right, right, left).astype(np.int32)


def _complete_sum(values: np.ndarray) -> np.ndarray:
    complete = np.isfinite(values).all(axis=0)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    result[complete] = values[:, complete].sum(axis=0)
    return result


def rainfall_features(
    precipitation_rate: np.ndarray,
    interval_midpoints: np.ndarray,
    *,
    event_start: datetime,
    event_end: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full-event, maximum 6 h, and antecedent 72 h rainfall."""
    rates = np.asarray(precipitation_rate, dtype=np.float64)
    midpoints = np.asarray(interval_midpoints, dtype=object)
    if (
        rates.ndim != 2
        or midpoints.ndim != 1
        or len(midpoints) != rates.shape[0]
        or event_start.tzinfo is None
        or event_end.tzinfo is None
        or event_end < event_start
    ):
        raise ValueError("IMERG rainfall inputs are invalid")
    if any(not isinstance(value, datetime) or value.tzinfo is None for value in midpoints):
        raise TypeError("IMERG interval midpoints must be timezone-aware datetimes")
    if any(later - earlier != timedelta(minutes=30) for earlier, later in pairwise(midpoints)):
        raise ValueError("IMERG inventory must be a complete half-hour sequence")
    antecedent_start = event_start - timedelta(hours=72)
    antecedent_mask = np.array(
        [antecedent_start <= value < event_start for value in midpoints],
        dtype=bool,
    )
    event_mask = np.array(
        [event_start <= value <= event_end for value in midpoints],
        dtype=bool,
    )
    if np.count_nonzero(antecedent_mask) != 144:
        raise ValueError("IMERG antecedent inventory must contain 144 intervals")
    event_rates = rates[event_mask]
    if len(event_rates) < 12:
        raise ValueError("IMERG event inventory must contain a complete 6 h window")
    antecedent = _complete_sum(rates[antecedent_mask] * 0.5)
    event_total = _complete_sum(event_rates * 0.5)
    maxima = np.full(rates.shape[1], np.nan, dtype=np.float64)
    for start in range(len(event_rates) - 11):
        window = _complete_sum(event_rates[start : start + 12] * 0.5)
        finite = np.isfinite(window)
        replace = finite & (~np.isfinite(maxima) | (window > maxima))
        maxima[replace] = window[replace]
    return event_total, maxima, antecedent


def validate_inventory(
    granules: list[ImergGranule],
    *,
    expected_count: int,
    first_filename: str,
    last_filename: str,
) -> None:
    """Require the frozen count, endpoints, and uninterrupted half-hour cadence."""
    if (
        len(granules) != expected_count
        or not granules
        or granules[0].filename != first_filename
        or granules[-1].filename != last_filename
        or any(not item.filename.endswith(".V07B.HDF5") for item in granules)
        or any(
            later.midpoint - earlier.midpoint != timedelta(minutes=30)
            for earlier, later in pairwise(granules)
        )
    ):
        raise ValueError("IMERG inventory differs from the frozen half-hour sequence")


def _cmr_granules(city_id: str, start: str, end: str) -> list[ImergGranule]:
    query = earthaccess.DataGranules()
    # The configured local proxy terminates CMR TLS; direct CMR access is public and credential-free.
    query.session.trust_env = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        results = query.short_name("GPM_3IMERGHH").version("07").temporal(start, end).get()
    granules = []
    for result in results:
        umm = result.get("umm", {})
        granule_ur = umm.get("GranuleUR")
        temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
        links = [url for url in result.data_links(access="external") if url.endswith(".HDF5")]
        if (
            not isinstance(granule_ur, str)
            or len(links) != 1
            or not isinstance(temporal.get("BeginningDateTime"), str)
            or not isinstance(temporal.get("EndingDateTime"), str)
        ):
            raise ValueError("CMR returned incomplete IMERG granule metadata")
        filename = granule_ur.split(":", 1)[-1]
        parsed = urlsplit(links[0])
        if parsed.scheme != "https" or parsed.query or Path(parsed.path).name != filename:
            raise ValueError("CMR returned an invalid IMERG data link")
        byte_count = round(float(result["size"]) * 1024 * 1024)
        if byte_count <= 0:
            raise ValueError("CMR returned an invalid IMERG granule size")
        granules.append(
            ImergGranule(
                city_id=city_id,
                filename=filename,
                granule_ur=granule_ur,
                url=links[0],
                beginning=temporal["BeginningDateTime"],
                ending=temporal["EndingDateTime"],
                bytes=byte_count,
            )
        )
    return sorted(granules, key=lambda item: item.beginning)


def _write_inventory(rows: list[ImergGranule], destination: Path) -> Path:
    schema = pa.schema(
        [
            pa.field("city_id", pa.string(), nullable=False),
            pa.field("filename", pa.string(), nullable=False),
            pa.field("granule_ur", pa.string(), nullable=False),
            pa.field("url", pa.string(), nullable=False),
            pa.field("beginning", pa.string(), nullable=False),
            pa.field("ending", pa.string(), nullable=False),
            pa.field("bytes", pa.int64(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist([asdict(row) for row in rows], schema=schema)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    pq.write_table(table, partial, compression="zstd", use_dictionary=False, write_statistics=False)
    os.replace(partial, destination)
    return destination


def prepare_imerg_inventory(root: Path) -> tuple[list[ImergGranule], Path, Path]:
    """Query and freeze the two public CMR inventories without authenticating."""
    protocol = load_protocol(root / "configs")
    granules: list[ImergGranule] = []
    city_reports = []
    for city_id in ("zhengzhou", "beijing"):
        city = protocol.cities[city_id]
        coverage = city["imerg"]["inventory_coverage"]
        selected = _cmr_granules(city_id, coverage["start"], coverage["end"])
        validate_inventory(
            selected,
            expected_count=int(city["imerg"]["expected_granules"]),
            first_filename=str(coverage["first_filename"]),
            last_filename=str(coverage["last_filename"]),
        )
        granules.extend(selected)
        city_reports.append(
            {
                "city_id": city_id,
                "granules": len(selected),
                "bytes": sum(item.bytes for item in selected),
                "coverage": {"start": coverage["start"], "end": coverage["end"]},
                "first_filename": selected[0].filename,
                "last_filename": selected[-1].filename,
            }
        )
    manifest = _write_inventory(
        granules,
        root / "artifacts" / "manifests" / "imerg_inventory.parquet",
    )
    planned_bytes = sum(item.bytes for item in granules)
    payload = {
        "schema_version": 1,
        "source_id": "imerg_v07b",
        "short_name": protocol.sources["imerg_v07b"]["short_name"],
        "cmr_version": protocol.sources["imerg_v07b"]["version"],
        "file_version": protocol.sources["imerg_v07b"]["file_version"],
        "status": "inventory-validated-download-pending",
        "authentication": {"strategy": "interactive", "persist": False},
        "planned_bytes": planned_bytes,
        "required_free_bytes": required_free_bytes(planned_bytes),
        "manifest": {
            "path": manifest.relative_to(root).as_posix(),
            "bytes": manifest.stat().st_size,
            "sha256": sha256_file(manifest),
        },
        "cities": city_reports,
    }
    report = root / "artifacts" / "reports" / "imerg_inventory.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    partial = report.with_name(f"{report.name}.part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, report)
    return granules, manifest, report


def _raw_path(raw_root: Path, granule: ImergGranule) -> Path:
    parts = Path(urlsplit(granule.url).path).parts
    try:
        product_index = parts.index("GPM_3IMERGHH.07")
    except ValueError:
        raise ValueError("IMERG data link lacks the frozen product path") from None
    relative = parts[product_index:]
    if len(relative) != 4 or relative[-1] != granule.filename:
        raise ValueError("IMERG data link has an invalid granule path")
    return raw_root / "imerg_v07b" / Path(*relative)


def _existing_record(
    destination: Path,
    raw_root: Path,
    granule: ImergGranule,
) -> SourceRecord | None:
    receipt = destination.with_name(f"{destination.name}.source.json")
    if not receipt.exists():
        return None
    try:
        record = SourceRecord(**json.loads(receipt.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("existing IMERG source receipt is invalid") from None
    if (
        record.source_id != "imerg_v07b"
        or record.asset_key != granule.granule_ur
        or record.relative_path != destination.relative_to(raw_root).as_posix()
        or not destination.is_file()
        or destination.stat().st_size != record.bytes
        or sha256_file(destination) != record.sha256
    ):
        raise ValueError("existing IMERG source receipt does not match its granule")
    return record


def _source_record(
    protocol: LoadedProtocol,
    granule: ImergGranule,
    destination: Path,
    raw_root: Path,
    *,
    sha256: str,
    retrieved_at: str,
) -> SourceRecord:
    source = protocol.sources["imerg_v07b"]
    return SourceRecord(
        source_id="imerg_v07b",
        city_id=granule.city_id,
        asset_key=granule.granule_ur,
        provider=str(source["provider"]),
        version_or_release=f"{source['path_product']}/{source['file_version']}",
        source_url_or_doi=str(source["doi"]),
        license=str(source["license"]),
        privacy_status=str(source["privacy_status"]),
        reference_date_or_period=f"{granule.beginning}/{granule.ending}",
        relative_path=destination.relative_to(raw_root).as_posix(),
        retrieved_at=retrieved_at,
        bytes=granule.bytes,
        sha256=sha256,
        request_sha256=sha256_bytes(
            canonical_json_bytes(
                {
                    "cmr_concept_id": source["cmr_concept_id"],
                    "granule_ur": granule.granule_ur,
                    "method": "GET",
                }
            )
        ),
    )


def _download_one(
    protocol: LoadedProtocol,
    raw_root: Path,
    auth: earthaccess.Auth,
    granule: ImergGranule,
) -> SourceRecord:
    destination = _raw_path(raw_root, granule)
    existing = _existing_record(destination, raw_root, granule)
    if existing is not None:
        return existing
    if destination.exists():
        if destination.stat().st_size != granule.bytes:
            raise ValueError("existing IMERG granule has the wrong size")
        return _source_record(
            protocol,
            granule,
            destination,
            raw_root,
            sha256=sha256_file(destination),
            retrieved_at=datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        )
    session = auth.get_session()
    session.trust_env = False
    download = download_http(
        session,
        granule.url,
        destination,
        asset_identity=f"imerg:{granule.granule_ur}",
        expected_size=granule.bytes,
        timeout=(20, 300),
    )
    return _source_record(
        protocol,
        granule,
        destination,
        raw_root,
        sha256=download.sha256,
        retrieved_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )


def _write_source_manifest(records: list[SourceRecord], destination: Path) -> Path:
    rows = [
        {name: getattr(record, name) for name in RAW_SOURCE_COLUMNS}
        for record in sorted(
            records, key=lambda item: (item.city_id, item.reference_date_or_period)
        )
    ]
    table = pa.Table.from_pylist(rows, schema=RAW_SOURCE_SCHEMA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    pq.write_table(table, partial, compression="zstd", use_dictionary=False, write_statistics=False)
    os.replace(partial, destination)
    return destination


def download_imerg(
    root: Path,
    granules: list[ImergGranule],
    auth: earthaccess.Auth,
    *,
    workers: int = 8,
) -> tuple[list[SourceRecord], Path]:
    """Download all inventoried granules with in-memory Earthdata authentication."""
    if workers <= 0:
        raise ValueError("IMERG worker count must be positive")
    protocol = load_protocol(root / "configs")
    raw_root = root / "data" / "raw"
    ensure_free_space(raw_root, sum(item.bytes for item in granules))
    records: list[SourceRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, protocol, raw_root, auth, granule): granule
            for granule in granules
        }
        for future in as_completed(futures):
            record = future.result()
            write_source_receipt(record, raw_root, protocol)
            records.append(record)
    records.sort(key=lambda item: (item.city_id, item.reference_date_or_period))
    manifest = _write_source_manifest(
        records,
        root / "artifacts" / "manifests" / "imerg_granule_manifest.parquet",
    )
    build_raw_source_manifest(raw_root, root / "artifacts", protocol)
    return records, manifest


def _axes(source_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(
        source_path,
        group="Grid",
        engine="h5netcdf",
        decode_times=False,
    ) as dataset:
        latitude = np.asarray(dataset["lat"].values, dtype=np.float64)
        longitude = np.asarray(dataset["lon"].values, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("IMERG coordinate axes are invalid")
    return latitude, longitude


def _native_mapping(
    nodes: gpd.GeoDataFrame,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = gpd.GeoSeries(
        gpd.points_from_xy(nodes["center_x"], nodes["center_y"]),
        crs=nodes.crs,
    ).to_crs(4326)
    lat_index = nearest_axis_indices(latitude, centers.y.to_numpy())
    lon_index = nearest_axis_indices(longitude, centers.x.to_numpy())
    pairs = np.column_stack((lat_index, lon_index))
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return lat_index, lon_index, unique, inverse


def _read_rates(
    paths: list[Path],
    unique_pairs: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    min_lat, min_lon = unique_pairs.min(axis=0)
    max_lat, max_lon = unique_pairs.max(axis=0)
    local_lat = unique_pairs[:, 0] - min_lat
    local_lon = unique_pairs[:, 1] - min_lon
    rates = np.empty((len(paths), len(unique_pairs)), dtype=np.float32)
    for index, path in enumerate(paths):
        with xr.open_dataset(
            path,
            group="Grid",
            engine="h5netcdf",
            decode_times=False,
        ) as dataset:
            if not np.array_equal(dataset["lat"].values, latitude) or not np.array_equal(
                dataset["lon"].values, longitude
            ):
                raise ValueError("IMERG coordinate axes changed within the inventory")
            precipitation = dataset["precipitation"]
            if "time" in precipitation.dims:
                precipitation = precipitation.isel(time=0)
            window = np.asarray(
                precipitation.isel(
                    lat=slice(int(min_lat), int(max_lat) + 1),
                    lon=slice(int(min_lon), int(max_lon) + 1),
                )
                .transpose("lat", "lon")
                .values,
                dtype=np.float32,
            )
        values = window[local_lat, local_lon]
        values[values < 0] = np.nan
        rates[index] = values
    return rates


def _write_geoparquet(frame: gpd.GeoDataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    frame.to_parquet(partial, index=False, compression="zstd", row_group_size=65_536)
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
    granules: list[ImergGranule],
    records: dict[str, SourceRecord],
    inventory_sha256: str,
    protocol: LoadedProtocol,
) -> dict[str, object]:
    city_root = root / "data" / "interim" / city_id
    nodes = gpd.read_parquet(city_root / "candidate_urban_domain.parquet")
    nodes = nodes.sort_values(["row", "column"], kind="stable").reset_index(drop=True)
    paths = [root / "data" / "raw" / records[item.granule_ur].relative_path for item in granules]
    latitude, longitude = _axes(paths[0])
    lat_index, lon_index, unique_pairs, inverse = _native_mapping(nodes, latitude, longitude)
    rates = _read_rates(paths, unique_pairs, latitude, longitude)
    city = protocol.cities[city_id]
    event_start = datetime.fromisoformat(str(city["event_window"]["utc_start"]))
    event_end = datetime.fromisoformat(str(city["event_window"]["utc_end"]))
    event_total, max_six_hours, antecedent = rainfall_features(
        rates,
        np.asarray([item.midpoint for item in granules], dtype=object),
        event_start=event_start,
        event_end=event_end,
    )
    native = gpd.GeoDataFrame(
        {
            "city_id": city_id,
            "imerg_lat_index": unique_pairs[:, 0],
            "imerg_lon_index": unique_pairs[:, 1],
            "imerg_lat": latitude[unique_pairs[:, 0]],
            "imerg_lon": longitude[unique_pairs[:, 1]],
            "event_rainfall_total": event_total,
            "max_6h_rainfall": max_six_hours,
            "antecedent_72h_rainfall": antecedent,
        },
        geometry=gpd.points_from_xy(longitude[unique_pairs[:, 1]], latitude[unique_pairs[:, 0]]),
        crs=4326,
    )
    native_path = _write_geoparquet(native, city_root / "imerg_native_features.parquet")
    output = nodes[
        ["city_id", "grid_id", "row", "column", "center_x", "center_y", "geometry"]
    ].copy()
    output["event_rainfall_total"] = event_total[inverse]
    output["max_6h_rainfall"] = max_six_hours[inverse]
    output["antecedent_72h_rainfall"] = antecedent[inverse]
    output["imerg_lat_index"] = lat_index
    output["imerg_lon_index"] = lon_index
    output["imerg_lat"] = latitude[lat_index]
    output["imerg_lon"] = longitude[lon_index]
    output["imerg_inventory_sha256"] = inventory_sha256
    output["feature_stage"] = "worldcover_candidate"
    output["is_final_scorable"] = False
    feature_path = _write_geoparquet(output, city_root / "imerg_features.parquet")
    feature_names = (
        "event_rainfall_total",
        "max_6h_rainfall",
        "antecedent_72h_rainfall",
    )
    return {
        "city_id": city_id,
        "granules": len(granules),
        "granule_bytes": sum(item.bytes for item in granules),
        "independent_native_pixels": len(unique_pairs),
        "candidate_nodes": len(output),
        "feature_missing": {name: int(output[name].isna().sum()) for name in feature_names},
        "feature_ranges_mm": {
            name: [float(output[name].min()), float(output[name].max())] for name in feature_names
        },
        "outputs": [_file_record(root, native_path), _file_record(root, feature_path)],
    }


def prepare_imerg(
    root: Path,
    auth: earthaccess.Auth,
    *,
    workers: int = 8,
) -> dict[str, object]:
    """Inventory, download, and prepare all fixed IMERG rainfall features."""
    protocol = load_protocol(root / "configs")
    granules, inventory, inventory_report = prepare_imerg_inventory(root)
    records, manifest = download_imerg(root, granules, auth, workers=workers)
    by_ur = {record.asset_key: record for record in records}
    inventory_sha256 = sha256_file(inventory)
    city_reports = [
        _process_city(
            root,
            city_id,
            [item for item in granules if item.city_id == city_id],
            by_ur,
            inventory_sha256,
            protocol,
        )
        for city_id in ("zhengzhou", "beijing")
    ]
    payload = {
        "schema_version": 1,
        "source_id": "imerg_v07b",
        "doi": protocol.sources["imerg_v07b"]["doi"],
        "license": protocol.sources["imerg_v07b"]["license"],
        "privacy_status": protocol.sources["imerg_v07b"]["privacy_status"],
        "authentication": {"strategy": "interactive", "persist": False},
        "raw_file_count": len(records),
        "raw_total_bytes": sum(record.bytes for record in records),
        "inventory": _file_record(root, inventory),
        "inventory_report": _file_record(root, inventory_report),
        "granule_manifest": _file_record(root, manifest),
        "processing": {
            "variable": protocol.sources["imerg_v07b"]["variable"]["path"],
            "source_units": protocol.sources["imerg_v07b"]["variable"]["units"],
            "interval_amount": "rate multiplied by 0.5 h",
            "spatial_mapping": "nearest native center; lower latitude then longitude index on ties",
            "event_endpoint_rule": "interval midpoint in closed event interval",
            "antecedent_endpoint_rule": "[event_start - 72 h, event_start)",
            "missing_rule": "no fill value is treated as zero rainfall",
        },
        "task_definition": "retrospective full-event flood-report hotspot ranking",
        "aoi_stage": "worldcover_candidate",
        "is_final_scorable": False,
        "cities": city_reports,
    }
    report = root / "artifacts" / "reports" / "imerg.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    partial = report.with_name(f"{report.name}.part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, report)
    return {
        "raw_file_count": len(records),
        "total_bytes": sum(record.bytes for record in records),
        "manifest": str(manifest),
        "report": str(report),
        "cities": {city["city_id"]: city["candidate_nodes"] for city in city_reports},
    }
