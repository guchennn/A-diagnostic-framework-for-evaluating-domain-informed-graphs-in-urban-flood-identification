from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from floodgnn_ceus import features
from floodgnn_ceus.features import (
    aggregate_class_fractions,
    candidate_node_mask,
    retain_large_built_components,
    select_worldcover_tiles,
    write_worldcover_manifest,
)
from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.sources import RAW_SOURCE_COLUMNS, SourceRecord


def test_tile_selection_uses_each_projected_aoi_plus_context() -> None:
    tiles = gpd.GeoDataFrame(
        {
            "ll_tile": ["N00E000", "N00E003", "N03E000"],
            "geometry": [
                Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
                Polygon([(3, 0), (6, 0), (6, 3), (3, 3)]),
                Polygon([(0, 3), (3, 3), (3, 6), (0, 6)]),
            ],
        },
        crs="EPSG:4326",
    )
    aoi = gpd.GeoDataFrame(
        {"city_id": ["test"], "geometry": [Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])]},
        crs="EPSG:4326",
    ).to_crs(32631)

    selected = select_worldcover_tiles(tiles, {"test": aoi}, context_buffer_m=10_000)

    assert selected == {"test": ("N00E000",)}


def test_class_fractions_use_only_valid_pixels() -> None:
    classes = np.array(
        [
            [50, 40, 80, 0],
            [10, 20, 80, 80],
            [0, 0, 50, 95],
            [0, 0, 100, 80],
        ],
        dtype=np.uint8,
    )

    built, water, vegetation, cropland, valid = aggregate_class_fractions(classes, factor=2)

    np.testing.assert_allclose(built, [[0.25, 0.0], [np.nan, 0.25]], equal_nan=True)
    np.testing.assert_allclose(water, [[0.0, 1.0], [np.nan, 0.25]], equal_nan=True)
    np.testing.assert_allclose(vegetation, [[0.5, 0.0], [np.nan, 0.5]], equal_nan=True)
    np.testing.assert_allclose(cropland, [[0.25, 0.0], [np.nan, 0.0]], equal_nan=True)
    np.testing.assert_array_equal(valid, [[4, 3], [0, 4]])


def test_distance_uses_class_pixel_boundaries_and_caps_missing_searches() -> None:
    classes = np.array([[80, 0], [0, 0]], dtype=np.uint8)
    transform = from_origin(0, 20, 10, 10)

    distance = features.distance_to_class_boundary(
        classes,
        transform=transform,
        class_value=80,
        x=np.array([5.0, 15.0, 35.0]),
        y=np.array([15.0, 15.0, 15.0]),
        maximum_distance_m=20,
    )
    missing = features.distance_to_class_boundary(
        np.zeros((2, 2), dtype=np.uint8),
        transform=transform,
        class_value=80,
        x=np.array([5.0]),
        y=np.array([15.0]),
        maximum_distance_m=20,
    )

    np.testing.assert_allclose(distance, [5.0, 5.0, 20.0])
    np.testing.assert_allclose(missing, [20.0])


def test_large_components_use_eight_neighbor_connectivity() -> None:
    classes = np.array(
        [
            [50, 0, 0, 50],
            [0, 50, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 50],
        ],
        dtype=np.uint8,
    )

    retained, component_count, retained_count = retain_large_built_components(
        classes,
        built_class=50,
        pixel_size_m=10,
        minimum_component_km2=0.0002,
    )

    np.testing.assert_array_equal(
        retained,
        np.array(
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ),
    )
    assert component_count == 3
    assert retained_count == 1


def test_candidate_nodes_are_within_buffer_and_not_mostly_water() -> None:
    distance_to_built = np.array([2_000.0, 1_500.0, 2_000.1])
    water = np.array([0.5, 0.5001, 0.0], dtype=np.float32)

    keep = candidate_node_mask(
        distance_to_built,
        water,
        buffer_m=2_000,
        exclude_water_fraction_above=0.5,
    )

    np.testing.assert_array_equal(keep, [True, False, False])


def test_worldcover_manifest_is_replaceable_and_reproducible(tmp_path: Path) -> None:
    def record(tile_id: str) -> SourceRecord:
        return SourceRecord(
            source_id="worldcover_2021_v200",
            city_id="all",
            asset_key=tile_id,
            provider="ESA WorldCover",
            version_or_release="v200",
            source_url_or_doi="https://example.test/{ll_tile}.tif",
            license="CC-BY-4.0",
            privacy_status="approved",
            reference_date_or_period="2021",
            relative_path=f"worldcover/{tile_id}.tif",
            retrieved_at="2026-08-08T00:00:00Z",
            bytes=10,
            sha256="a" * 64,
            request_sha256="b" * 64,
        )

    destination = tmp_path / "worldcover_tile_manifest.parquet"
    write_worldcover_manifest([record("N00E000")], destination)
    write_worldcover_manifest([record("N00E000"), record("N00E003")], destination)
    first_hash = sha256_file(destination)
    write_worldcover_manifest([record("N00E000"), record("N00E003")], destination)

    assert sha256_file(destination) == first_hash
    table = pq.read_table(destination)
    assert table.column_names == list(RAW_SOURCE_COLUMNS)
    assert table.num_rows == 2
