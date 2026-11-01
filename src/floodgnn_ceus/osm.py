"""Historical OSM acquisition and fixed 200 m line features."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import box, shape
from shapely.strtree import STRtree

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file
from floodgnn_ceus.protocol import load_protocol
from floodgnn_ceus.sources import (
    DownloadError,
    SourceRecord,
    build_raw_source_manifest,
    write_source_receipt,
)


def partition_bounds(
    bounds: tuple[float, float, float, float],
    *,
    degrees: float,
) -> tuple[tuple[float, float, float, float], ...]:
    """Cover bounds with fixed degree-aligned boxes in stable order."""
    min_x, min_y, max_x, max_y = bounds
    if degrees <= 0 or max_x <= min_x or max_y <= min_y:
        raise ValueError("OSM partition bounds are invalid")
    first_x = math.floor(min_x / degrees) * degrees
    first_y = math.floor(min_y / degrees) * degrees
    column_count = math.ceil((max_x - first_x) / degrees)
    row_count = math.ceil((max_y - first_y) / degrees)
    return tuple(
        (
            round(first_x + column * degrees, 10),
            round(first_y + row * degrees, 10),
            round(first_x + (column + 1) * degrees, 10),
            round(first_y + (row + 1) * degrees, 10),
        )
        for row in range(row_count)
        for column in range(column_count)
    )


def validate_feature_collection(payload: object) -> str:
    """Validate one complete ohsome line-extraction response."""
    if not isinstance(payload, Mapping):
        raise TypeError("ohsome response must be an object")
    api_version = payload.get("apiVersion")
    features = payload.get("features")
    if (
        payload.get("type") != "FeatureCollection"
        or not isinstance(api_version, str)
        or not api_version
        or not isinstance(features, list)
    ):
        raise ValueError("ohsome response is not a feature collection")
    for feature in features:
        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
            raise ValueError("ohsome response contains an invalid feature")
        geometry = feature.get("geometry")
        properties = feature.get("properties")
        if (
            not isinstance(geometry, Mapping)
            or geometry.get("type") not in {"LineString", "MultiLineString"}
            or not isinstance(properties, Mapping)
            or not isinstance(properties.get("@osmId"), str)
            or type(properties.get("@version")) is not int
        ):
            raise ValueError("ohsome response requires versioned line features")
        try:
            parsed = shape(geometry)
        except (TypeError, ValueError):
            raise ValueError("ohsome response contains invalid line geometry") from None
        if parsed.is_empty or not parsed.is_valid:
            raise ValueError("ohsome response contains invalid line geometry")
    return api_version


def _snapshot_features(payload: object) -> tuple[str, list[Mapping[str, object]]]:
    if not isinstance(payload, Mapping):
        raise TypeError("ohsome snapshot response must be an object")
    api_version = payload.get("apiVersion")
    features = payload.get("features")
    if (
        payload.get("type") != "FeatureCollection"
        or not isinstance(api_version, str)
        or not api_version
        or not isinstance(features, list)
    ):
        raise ValueError("ohsome snapshot is not a feature collection")
    for feature in features:
        properties = feature.get("properties") if isinstance(feature, Mapping) else None
        if (
            not isinstance(feature, Mapping)
            or feature.get("type") != "Feature"
            or not isinstance(properties, Mapping)
            or not isinstance(properties.get("@osmId"), str)
            or type(properties.get("@version")) is not int
        ):
            raise ValueError("ohsome snapshot requires versioned features")
    return api_version, features


def match_snapshot_geometry(
    snapshot: object,
    history: object,
) -> list[Mapping[str, object]]:
    """Attach full-history geometry only when both OSM object sets match exactly."""
    snapshot_version, snapshot_features = _snapshot_features(snapshot)
    history_version = validate_feature_collection(history)
    if snapshot_version != history_version:
        raise ValueError("ohsome snapshot and history API versions differ")
    history_features = history["features"]
    snapshots: dict[tuple[str, int], Mapping[str, object]] = {}
    histories: dict[tuple[str, int], Mapping[str, object]] = {}
    for destination, features in (
        (snapshots, snapshot_features),
        (histories, history_features),
    ):
        for feature in features:
            properties = feature["properties"]
            key = (properties["@osmId"], properties["@version"])
            if key in destination:
                raise ValueError("ohsome response contains duplicate object versions")
            destination[key] = feature
    if snapshots.keys() != histories.keys():
        raise ValueError("ohsome snapshot and history object sets differ")
    return [
        {
            "type": "Feature",
            "geometry": histories[key]["geometry"],
            "properties": dict(snapshots[key]["properties"]),
        }
        for key in sorted(snapshots)
    ]


def deduplicate_features(features: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """Deduplicate un-clipped chunk overlap by the frozen OSM key."""
    unique: dict[tuple[str, int], tuple[bytes, Mapping[str, object]]] = {}
    for feature in features:
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            raise TypeError("OSM feature properties are invalid")
        osm_id = properties.get("@osmId")
        version = properties.get("@version")
        if not isinstance(osm_id, str) or type(version) is not int:
            raise ValueError("OSM feature deduplication key is invalid")
        key = (osm_id, version)
        encoded = canonical_json_bytes(dict(feature))
        previous = unique.get(key)
        if previous is not None and previous[0] != encoded:
            raise ValueError("conflicting duplicate OSM feature")
        unique[key] = (encoded, feature)
    return [unique[key][1] for key in sorted(unique)]


def _length_by_cell(cells: np.ndarray, lines: np.ndarray) -> np.ndarray:
    lengths = np.zeros(len(cells), dtype=np.float64)
    if not len(lines):
        return lengths
    pairs = STRtree(lines).query(cells, predicate="intersects")
    if pairs.shape[1]:
        clipped = shapely.intersection(cells[pairs[0]], lines[pairs[1]])
        lengths = np.bincount(
            pairs[0],
            weights=np.asarray(shapely.length(clipped), dtype=np.float64),
            minlength=len(cells),
        )
    return lengths


def osm_grid_features(
    grid: gpd.GeoDataFrame,
    roads: np.ndarray,
    waterways: np.ndarray,
    *,
    maximum_distance_m: float,
) -> pd.DataFrame:
    """Compute the three fixed OSM features on projected grid cells."""
    if grid.empty or grid.crs is None or not grid.crs.is_projected or maximum_distance_m <= 0:
        raise ValueError("OSM grid feature inputs are invalid")
    cells = np.asarray(grid.geometry.array, dtype=object)
    roads = np.asarray(roads, dtype=object)
    waterways = np.asarray(waterways, dtype=object)
    cell_areas = np.asarray(shapely.area(cells), dtype=np.float64)
    if np.any(cell_areas <= 0):
        raise ValueError("OSM grid cells must have positive area")
    road_density = _length_by_cell(cells, roads) * 1000 / cell_areas
    waterway_density = _length_by_cell(cells, waterways) * 1000 / cell_areas
    if len(waterways):
        points = shapely.points(grid["center_x"].to_numpy(), grid["center_y"].to_numpy())
        nearest = STRtree(waterways).nearest(points)
        distance = np.minimum(
            np.asarray(shapely.distance(points, waterways[nearest]), dtype=np.float64),
            maximum_distance_m,
        )
    else:
        distance = np.full(len(grid), maximum_distance_m, dtype=np.float64)
    return pd.DataFrame(
        {
            "grid_id": grid["grid_id"].to_numpy(),
            "road_length_density": road_density.astype(np.float32),
            "waterway_length_density": waterway_density.astype(np.float32),
            "distance_to_nearest_waterway": distance.astype(np.float32),
        }
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _paired_endpoints(endpoint: str) -> tuple[str, str]:
    suffix = "/elements/geometry"
    if not endpoint.endswith(suffix):
        raise ValueError("ohsome endpoint must end with /elements/geometry")
    base = endpoint.removesuffix(suffix)
    return f"{base}/elements/bbox", f"{base}/elementsFullHistory/geometry"


def _history_interval(timestamp: str) -> str:
    try:
        start = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        raise ValueError("ohsome snapshot timestamp must use UTC seconds") from None
    end = (start + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{timestamp},{end}"


def _metadata(session: requests.Session, endpoint: str) -> tuple[str, str]:
    metadata_url = endpoint.rsplit("/elements/geometry", 1)[0] + "/metadata"
    try:
        response = session.get(metadata_url, timeout=(10, 60))
    except requests.RequestException:
        raise DownloadError("ohsome metadata request failed") from None
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, requests.JSONDecodeError):
        raise DownloadError("ohsome metadata response is invalid") from None
    api_version = payload.get("apiVersion")
    sequence = payload.get("extractRegion", {}).get("replicationSequenceNumber")
    if not isinstance(api_version, str) or type(sequence) is not int:
        raise DownloadError("ohsome metadata lacks version or replication sequence")
    return api_version, str(sequence)


def _atomic_response(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DownloadError("existing ohsome response differs from the verified response")
        return path
    partial = path.with_name(f"{path.name}.part")
    partial.write_bytes(payload)
    try:
        os.link(partial, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise DownloadError(
                "existing ohsome response differs from the verified response"
            ) from None
    finally:
        partial.unlink(missing_ok=True)
    return path


def _existing_record(path: Path, raw_root: Path, asset_key: str) -> SourceRecord | None:
    receipt = path.with_name(f"{path.name}.source.json")
    if not receipt.exists():
        return None
    try:
        record = SourceRecord(**json.loads(receipt.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("existing ohsome source receipt is invalid") from None
    if (
        record.source_id != "ohsome_v1"
        or record.asset_key != asset_key
        or record.relative_path != path.relative_to(raw_root).as_posix()
        or not path.is_file()
        or path.stat().st_size != record.bytes
        or sha256_file(path) != record.sha256
    ):
        raise ValueError("existing ohsome source receipt does not match its response")
    return record


def _request_one(
    session: requests.Session,
    endpoint: str,
    form: dict[str, str],
    *,
    response_kind: str,
) -> tuple[dict[str, object], bytes]:
    try:
        response = session.post(endpoint, data=form, timeout=(20, 600))
    except requests.RequestException:
        raise DownloadError("ohsome geometry request failed") from None
    if response.status_code != 200:
        raise DownloadError(f"ohsome geometry request returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except requests.JSONDecodeError:
        raise DownloadError("ohsome geometry response is not complete JSON") from None
    if response_kind == "bbox":
        _snapshot_features(payload)
    elif response_kind == "full-history":
        validate_feature_collection(payload)
    else:
        raise ValueError("unknown ohsome response kind")
    return payload, canonical_json_bytes(payload)


def _query_boxes(
    aoi: gpd.GeoDataFrame, context_buffer_m: int, degrees: float
) -> tuple[tuple[float, ...], ...]:
    context = (
        gpd.GeoSeries(
            [aoi.geometry.iloc[0].buffer(context_buffer_m)],
            crs=aoi.crs,
        )
        .to_crs(4326)
        .iloc[0]
    )
    return tuple(
        bounds
        for bounds in partition_bounds(context.bounds, degrees=degrees)
        if context.intersects(box(*bounds)) and not context.touches(box(*bounds))
    )


def fetch_osm(
    root: Path,
    session: requests.Session,
) -> tuple[
    list[SourceRecord],
    dict[str, dict[str, list[tuple[Path, Path]]]],
    Path,
]:
    """Fetch fixed historic road and waterway snapshots in 0.25-degree chunks."""
    protocol = load_protocol(root / "configs")
    source = protocol.sources["ohsome_v1"]
    endpoint = str(source["endpoint"])
    snapshot_endpoint, history_endpoint = _paired_endpoints(endpoint)
    metadata_version, replication_sequence = _metadata(session, endpoint)
    raw_root = root / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    context_buffer = int(protocol.protocol["terrain"]["context_buffer_m"])
    degrees = float(source["spatial_partition_degrees"])
    records: list[SourceRecord] = []
    paths: dict[str, dict[str, list[tuple[Path, Path]]]] = {}
    for city_id in ("zhengzhou", "beijing"):
        city_root = root / "data" / "interim" / city_id
        aoi = gpd.read_parquet(city_root / "admin_aoi.parquet")
        boxes = _query_boxes(aoi, context_buffer, degrees)
        timestamp = str(protocol.cities[city_id]["osm_snapshot"]["timestamp"])
        timestamp_key = timestamp.replace("-", "").replace(":", "")
        paths[city_id] = {}
        for layer in ("roads", "waterways"):
            paths[city_id][layer] = []
            for index, bounds in enumerate(boxes):
                pair: list[Path] = []
                for response_kind, request_endpoint, request_time in (
                    ("bbox", snapshot_endpoint, timestamp),
                    ("full-history", history_endpoint, _history_interval(timestamp)),
                ):
                    asset_key = f"{city_id}:{layer}:{index:04d}:{response_kind}"
                    destination = (
                        raw_root
                        / "ohsome_v1"
                        / metadata_version
                        / city_id
                        / timestamp_key
                        / layer
                        / f"part-{index:04d}.{response_kind}.geojson"
                    )
                    form = {
                        "bboxes": ",".join(f"{value:.10g}" for value in bounds),
                        "time": request_time,
                        "filter": str(source["filters"][layer]),
                        "showMetadata": "yes",
                        "properties": ",".join(source["properties"]),
                        "clipGeometry": "false",
                    }
                    request_sha256 = sha256_bytes(
                        canonical_json_bytes(
                            {"method": "POST", "url": request_endpoint, "form": form}
                        )
                    )
                    existing = _existing_record(destination, raw_root, asset_key)
                    if existing is None:
                        if destination.exists():
                            payload = json.loads(destination.read_text(encoding="utf-8"))
                        else:
                            payload, encoded = _request_one(
                                session,
                                request_endpoint,
                                form,
                                response_kind=response_kind,
                            )
                            _atomic_response(destination, encoded)
                        if response_kind == "bbox":
                            api_version = _snapshot_features(payload)[0]
                        else:
                            api_version = validate_feature_collection(payload)
                        if api_version != metadata_version:
                            raise DownloadError("ohsome response API version differs from metadata")
                        record = SourceRecord(
                            source_id="ohsome_v1",
                            city_id=city_id,
                            asset_key=asset_key,
                            provider=str(source["provider"]),
                            version_or_release=api_version,
                            source_url_or_doi=endpoint,
                            license=str(source["license"]),
                            privacy_status=str(source["privacy_status"]),
                            reference_date_or_period=request_time,
                            relative_path=destination.relative_to(raw_root).as_posix(),
                            retrieved_at=_utc_now(),
                            bytes=destination.stat().st_size,
                            sha256=sha256_file(destination),
                            request_sha256=request_sha256,
                            api_version=api_version,
                            replication_sequence=replication_sequence,
                        )
                        write_source_receipt(record, raw_root, protocol)
                    else:
                        record = existing
                        if record.request_sha256 != request_sha256:
                            raise ValueError(
                                "existing ohsome request hash differs from the frozen query"
                            )
                    records.append(record)
                    pair.append(destination)
                snapshot = json.loads(pair[0].read_text(encoding="utf-8"))
                history = json.loads(pair[1].read_text(encoding="utf-8"))
                match_snapshot_geometry(snapshot, history)
                paths[city_id][layer].append((pair[0], pair[1]))
    manifest = build_raw_source_manifest(raw_root, root / "artifacts", protocol)
    return records, paths, manifest


def _write_geoparquet(frame: gpd.GeoDataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    frame.to_parquet(partial, index=False, compression="zstd", row_group_size=65_536)
    os.replace(partial, destination)
    return destination


def _layer_frame(
    paths: Sequence[tuple[Path, Path]], *, crs: object, context: object
) -> tuple[gpd.GeoDataFrame, int]:
    raw_features: list[Mapping[str, object]] = []
    for snapshot_path, history_path in paths:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        raw_features.extend(match_snapshot_geometry(snapshot, history))
    features = deduplicate_features(raw_features)
    osm_ids = [feature["properties"]["@osmId"] for feature in features]
    versions = [feature["properties"]["@version"] for feature in features]
    geometries = gpd.GeoSeries(
        [shape(feature["geometry"]) for feature in features],
        crs=4326,
    ).to_crs(crs)
    clipped = shapely.intersection(np.asarray(geometries.array, dtype=object), context)
    keep = ~shapely.is_empty(clipped) & shapely.is_valid(clipped)
    frame = gpd.GeoDataFrame(
        {
            "osm_id": np.asarray(osm_ids, dtype=object)[keep],
            "version": np.asarray(versions, dtype=np.int64)[keep],
        },
        geometry=clipped[keep],
        crs=crs,
    ).sort_values(["osm_id", "version"], kind="stable")
    return frame.reset_index(drop=True), len(raw_features)


def _file_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare_osm(root: Path, session: requests.Session) -> dict[str, object]:
    """Acquire and prepare the three fixed OSM features for candidate nodes."""
    protocol = load_protocol(root / "configs")
    records, raw_paths, manifest = fetch_osm(root, session)
    snapshot_endpoint, history_endpoint = _paired_endpoints(
        str(protocol.sources["ohsome_v1"]["endpoint"])
    )
    maximum_distance = int(protocol.protocol["terrain"]["context_buffer_m"])
    city_reports = []
    for city_id in ("zhengzhou", "beijing"):
        city_root = root / "data" / "interim" / city_id
        aoi = gpd.read_parquet(city_root / "admin_aoi.parquet")
        context = aoi.geometry.iloc[0].buffer(maximum_distance)
        layers = {}
        frames = {}
        for layer in ("roads", "waterways"):
            frame, raw_count = _layer_frame(
                raw_paths[city_id][layer],
                crs=aoi.crs,
                context=context,
            )
            path = _write_geoparquet(frame, city_root / f"osm_{layer}.parquet")
            frames[layer] = frame
            layers[layer] = {
                "chunks": len(raw_paths[city_id][layer]),
                "raw_features_with_chunk_overlap": raw_count,
                "deduplicated_features": len(frame),
                "output": _file_record(root, path),
            }
        nodes = gpd.read_parquet(city_root / "candidate_urban_domain.parquet")
        nodes = nodes.sort_values(["row", "column"], kind="stable").reset_index(drop=True)
        values = osm_grid_features(
            nodes,
            np.asarray(frames["roads"].geometry.array, dtype=object),
            np.asarray(frames["waterways"].geometry.array, dtype=object),
            maximum_distance_m=maximum_distance,
        )
        output = nodes[
            ["city_id", "grid_id", "row", "column", "center_x", "center_y", "geometry"]
        ].copy()
        for name in values.columns[1:]:
            output[name] = values[name].to_numpy()
        output["feature_stage"] = "worldcover_candidate"
        output["is_final_scorable"] = False
        feature_path = _write_geoparquet(output, city_root / "osm_features.parquet")
        city_reports.append(
            {
                "city_id": city_id,
                "candidate_nodes": len(output),
                "layers": layers,
                "feature_ranges": {
                    name: [float(output[name].min()), float(output[name].max())]
                    for name in values.columns[1:]
                },
                "features": _file_record(root, feature_path),
            }
        )
    payload = {
        "schema_version": 1,
        "source_id": "ohsome_v1",
        "license": protocol.sources["ohsome_v1"]["license"],
        "privacy_status": protocol.sources["ohsome_v1"]["privacy_status"],
        "snapshot_contract": "city event UTC end",
        "request_contract": {
            "snapshot_endpoint": snapshot_endpoint,
            "history_geometry_endpoint": history_endpoint,
            "history_interval_seconds": 1,
            "pairing_key": ["@osmId", "@version"],
            "pairing_policy": "exact-object-set",
        },
        "partition_degrees": protocol.sources["ohsome_v1"]["spatial_partition_degrees"],
        "raw_file_count": len(records),
        "raw_total_bytes": sum(record.bytes for record in records),
        "api_versions": sorted({record.api_version for record in records}),
        "replication_sequences": sorted({record.replication_sequence for record in records}),
        "manifest": _file_record(root, manifest),
        "aoi_stage": "worldcover_candidate",
        "is_final_scorable": False,
        "cities": city_reports,
    }
    report = root / "artifacts" / "reports" / "osm.json"
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
