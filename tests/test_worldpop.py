from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from floodgnn_ceus.worldpop import population_feature, reproject_population_counts


def test_population_feature_uses_valid_area_and_fifty_percent_gate() -> None:
    population = np.array([[4.0, 2.0], [0.0, 8.0]])
    coverage = np.array([[1.0, 0.5], [0.49, 0.0]])

    feature = population_feature(
        population,
        coverage,
        cell_area_km2=0.04,
        minimum_coverage=0.5,
    )

    np.testing.assert_allclose(feature[0], np.log1p([100.0, 100.0]))
    assert np.isnan(feature[1]).all()


def test_population_counts_are_area_weighted_onto_aligned_cells(tmp_path: Path) -> None:
    source_path = tmp_path / "population.tif"
    values = np.ones((4, 4), dtype=np.float32)
    values[0, 0] = -99999
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 400, 100, 100),
        nodata=-99999,
    ) as target:
        target.write(values, 1)

    population, coverage, transform = reproject_population_counts(
        source_path,
        dst_crs="EPSG:3857",
        bounds=(0.0, 0.0, 400.0, 400.0),
        resolution_m=200,
    )

    np.testing.assert_allclose(population, [[3.0, 4.0], [4.0, 4.0]])
    np.testing.assert_allclose(coverage, [[0.75, 1.0], [1.0, 1.0]])
    assert transform == from_origin(0, 400, 200, 200)
