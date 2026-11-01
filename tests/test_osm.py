from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from floodgnn_ceus.osm import (
    _history_interval,
    _paired_endpoints,
    deduplicate_features,
    match_snapshot_geometry,
    osm_grid_features,
    partition_bounds,
    validate_feature_collection,
)


def test_ohsome_snapshot_and_history_requests_are_one_second_apart() -> None:
    endpoint = "https://api.ohsome.org/v1/elements/geometry"

    assert _paired_endpoints(endpoint) == (
        "https://api.ohsome.org/v1/elements/bbox",
        "https://api.ohsome.org/v1/elementsFullHistory/geometry",
    )
    assert _history_interval("2021-08-04T08:58:00Z") == (
        "2021-08-04T08:58:00Z,2021-08-04T08:58:01Z"
    )


def test_partition_bounds_are_fixed_aligned_and_row_major() -> None:
    partitions = partition_bounds((113.10, 34.10, 113.60, 34.40), degrees=0.25)

    assert partitions == (
        (113.0, 34.0, 113.25, 34.25),
        (113.25, 34.0, 113.5, 34.25),
        (113.5, 34.0, 113.75, 34.25),
        (113.0, 34.25, 113.25, 34.5),
        (113.25, 34.25, 113.5, 34.5),
        (113.5, 34.25, 113.75, 34.5),
    )


def test_feature_collection_requires_versioned_line_features() -> None:
    payload = {
        "type": "FeatureCollection",
        "apiVersion": "1.10.4",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 0]]},
                "properties": {"@osmId": "way/1", "@version": 2, "waterway": "stream"},
            }
        ],
    }

    assert validate_feature_collection(payload) == "1.10.4"
    del payload["features"][0]["properties"]["@version"]
    with pytest.raises(ValueError, match="versioned line"):
        validate_feature_collection(payload)


def test_deduplication_fails_if_same_osm_version_has_different_content() -> None:
    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 0]]},
        "properties": {"@osmId": "way/1", "@version": 2},
    }
    duplicate = {**feature, "properties": dict(feature["properties"])}

    assert deduplicate_features([feature, duplicate]) == [feature]
    duplicate["geometry"] = {"type": "LineString", "coordinates": [[0, 0], [2, 0]]}
    with pytest.raises(ValueError, match="conflicting duplicate"):
        deduplicate_features([feature, duplicate])


def test_snapshot_objects_receive_exact_full_history_geometry() -> None:
    snapshot = {
        "type": "FeatureCollection",
        "apiVersion": "1.10.4",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "properties": {"@osmId": "way/1", "@version": 2, "highway": "primary"},
            }
        ],
    }
    history = {
        "type": "FeatureCollection",
        "apiVersion": "1.10.4",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 0]]},
                "properties": {
                    "@osmId": "way/1",
                    "@version": 2,
                    "@validFrom": "2021-01-01T00:00:00Z",
                },
            }
        ],
    }

    assert match_snapshot_geometry(snapshot, history) == [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 0]]},
            "properties": {"@osmId": "way/1", "@version": 2, "highway": "primary"},
        }
    ]


def test_snapshot_and_history_object_sets_must_match() -> None:
    snapshot = {
        "type": "FeatureCollection",
        "apiVersion": "1.10.4",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "properties": {"@osmId": "way/1", "@version": 2},
            }
        ],
    }
    history = {"type": "FeatureCollection", "apiVersion": "1.10.4", "features": []}

    with pytest.raises(ValueError, match="object sets differ"):
        match_snapshot_geometry(snapshot, history)


def test_osm_features_use_clipped_length_and_center_distance() -> None:
    grid = gpd.GeoDataFrame(
        {
            "grid_id": ["a", "b"],
            "center_x": [100.0, 300.0],
            "center_y": [100.0, 100.0],
        },
        geometry=[box(0, 0, 200, 200), box(200, 0, 400, 200)],
        crs="EPSG:3857",
    )
    roads = np.array([LineString([(0, 100), (300, 100)])], dtype=object)
    waterways = np.array([LineString([(0, 0), (400, 0)])], dtype=object)

    result = osm_grid_features(grid, roads, waterways, maximum_distance_m=10_000)

    np.testing.assert_allclose(result["road_length_density"], [5.0, 2.5])
    np.testing.assert_allclose(result["waterway_length_density"], [5.0, 5.0])
    np.testing.assert_allclose(result["distance_to_nearest_waterway"], [100.0, 100.0])
    assert list(result) == [
        "grid_id",
        "road_length_density",
        "waterway_length_density",
        "distance_to_nearest_waterway",
    ]
