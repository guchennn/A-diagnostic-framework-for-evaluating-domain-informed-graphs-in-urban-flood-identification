from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import box

from floodgnn_ceus import features
from floodgnn_ceus.protocol import EXPECTED_FEATURE_NAMES

CONFIG = Path(__file__).parents[1] / "configs" / "feature_schema.yaml"
SOURCE_FOR = {
    **{name: "terrain_features.parquet" for name in EXPECTED_FEATURE_NAMES[:6]},
    **{name: "imerg_features.parquet" for name in EXPECTED_FEATURE_NAMES[6:9]},
    **{name: "worldcover_fractions.parquet" for name in EXPECTED_FEATURE_NAMES[9:13]},
    **{name: "osm_features.parquet" for name in EXPECTED_FEATURE_NAMES[13:16]},
    "distance_to_WorldCover_waterbody": "worldcover_fractions.parquet",
    "log1p_population_density": "worldpop_features.parquet",
}


def feature_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_products": {},
        "features": [
            {
                "name": name,
                "source_file": SOURCE_FOR[name],
                "dtype": "float64" if index < 9 else "float32",
            }
            for index, name in enumerate(EXPECTED_FEATURE_NAMES)
        ],
    }


def final_grid() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "grid_id": ["g2", "g0", "g1"],
            "row": [2, 0, 1],
            "column": [0, 0, 0],
            "center_x": [500.0, 100.0, 300.0],
            "center_y": [100.0, 100.0, 100.0],
        },
        geometry=[box(400, 0, 600, 200), box(0, 0, 200, 200), box(200, 0, 400, 200)],
        crs=32649,
    )


def source_tables() -> dict[str, pd.DataFrame]:
    grid_ids = ["g1", "extra", "g2", "g0"]
    values = [1.0, 99.0, 2.0, 0.0]
    tables: dict[str, pd.DataFrame] = {}
    for source_file in sorted(set(SOURCE_FOR.values())):
        columns: dict[str, object] = {"grid_id": grid_ids}
        for name in EXPECTED_FEATURE_NAMES:
            if SOURCE_FOR[name] != source_file:
                continue
            dtype = np.float64 if EXPECTED_FEATURE_NAMES.index(name) < 9 else np.float32
            columns[name] = np.asarray(values, dtype=dtype)
        tables[source_file] = pd.DataFrame(columns)
    tables["worldpop_features.parquet"].loc[0, "log1p_population_density"] = np.nan
    tables["imerg_features.parquet"]["imerg_lat"] = [1.0, 2.0, 3.0, 4.0]
    tables["osm_features.parquet"]["raw_id"] = ["a", "b", "c", "d"]
    tables["worldcover_fractions.parquet"]["flood_prob"] = 1.0
    tables["worldpop_features.parquet"]["toponym"] = "private"
    return tables


def test_assembly_anchors_geometry_and_rekeys_only_allowed_features() -> None:
    anchor = final_grid()
    anchor_geometry = dict(zip(anchor["grid_id"], anchor.geometry.to_wkb(), strict=True))

    result = features.assemble_feature_table(anchor, source_tables(), feature_contract())

    assert result.columns.tolist() == ["grid_id", *EXPECTED_FEATURE_NAMES, "geometry"]
    assert result["grid_id"].tolist() == ["g0", "g1", "g2"]
    assert result.crs == anchor.crs
    assert result.geometry.to_wkb().tolist() == [
        anchor_geometry[item] for item in result["grid_id"]
    ]
    assert result["elevation_mean"].tolist() == [0.0, 1.0, 2.0]
    assert np.isnan(result.loc[1, "log1p_population_density"])
    for index, name in enumerate(EXPECTED_FEATURE_NAMES):
        assert str(result[name].dtype) == ("float64" if index < 9 else "float32")
    assert {
        "row",
        "column",
        "center_x",
        "center_y",
        "imerg_lat",
        "raw_id",
        "toponym",
        "flood_prob",
    }.isdisjoint(result.columns)


def test_assembly_rejects_duplicate_or_missing_final_keys() -> None:
    duplicate = source_tables()
    terrain = duplicate["terrain_features.parquet"]
    duplicate["terrain_features.parquet"] = pd.concat(
        [terrain, terrain.loc[terrain["grid_id"] == "g0"]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="terrain_features.parquet has duplicate grid_id"):
        features.assemble_feature_table(final_grid(), duplicate, feature_contract())

    missing = source_tables()
    worldpop = missing["worldpop_features.parquet"]
    missing["worldpop_features.parquet"] = worldpop.loc[worldpop["grid_id"] != "g1"]
    with pytest.raises(ValueError, match="worldpop_features.parquet misses final grid_id"):
        features.assemble_feature_table(final_grid(), missing, feature_contract())


def test_feature_schema_freezes_the_complete_safe_contract() -> None:
    assert CONFIG.is_file()
    schema = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    columns = schema["features"]

    assert schema["schema_version"] == 1
    assert set(schema["source_products"]) == {item["source_id"] for item in columns}
    assert all(
        set(item) == {"source_product", "native_resolution"}
        for item in schema["source_products"].values()
    )
    assert [item["name"] for item in columns] == EXPECTED_FEATURE_NAMES
    assert [item["dtype"] for item in columns] == ["float64"] * 9 + ["float32"] * 9
    required = {
        "name",
        "output_column",
        "source_id",
        "source_version",
        "source_file",
        "unit",
        "dtype",
        "nodata",
        "operator",
        "resampling",
        "temporal_rule",
        "coverage_rule",
        "log_offset",
    }
    assert all(set(item) == required for item in columns)
    assert all(item["nodata"] == "NaN" for item in columns)


def test_column_hash_preserves_stored_dtype() -> None:
    values = pd.Series([1.0, np.nan], dtype=np.float32)

    assert features._column_hash(values) != features._column_hash(values.astype(np.float64))
