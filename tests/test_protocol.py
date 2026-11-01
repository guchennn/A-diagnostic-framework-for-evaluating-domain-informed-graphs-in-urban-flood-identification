from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest
import yaml

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file
from floodgnn_ceus.protocol import effective_protocol_id, load_protocol

CONFIG_DIR = Path(__file__).parents[1] / "configs"
SOURCE_ID_ORDER = [
    "crowdflood_v1",
    "geoboundaries_chn_adm2",
    "cop_dem_glo30_dged_2024_1",
    "worldcover_2021_v200",
    "imerg_v07b",
    "ohsome_v1",
    "worldpop_r2024b",
]
SOURCE_IDS = set(SOURCE_ID_ORDER)
EXPECTED_BASE_ID = "FloodGNN-CEUS-ZZ-BJ-v1.0"
EXPECTED_CONFIG_SHA256 = "903220eeba99b14b1ba70c6a53fa850fa005d41e1411eb71e9a403ddae197627"
EXPECTED_CANONICAL_JSON = {
    "ensure_ascii": False,
    "sort_keys": True,
    "separators": [",", ":"],
    "allow_nan": False,
    "encoding": "UTF-8",
    "trailing_newline": False,
}
EXPECTED_FEATURE_NAMES = [
    "elevation_mean",
    "slope_mean",
    "relative_elevation_1km",
    "local_relief_1km",
    "log_flow_accumulation",
    "twi",
    "event_rainfall_total",
    "max_6h_rainfall",
    "antecedent_72h_rainfall",
    "built_up_fraction",
    "water_fraction",
    "vegetation_fraction",
    "cropland_fraction",
    "road_length_density",
    "waterway_length_density",
    "distance_to_nearest_waterway",
    "distance_to_WorldCover_waterbody",
    "log1p_population_density",
]
EXPECTED_AOI = {
    "built_up_class": 50,
    "connectivity": 8,
    "minimum_component_km2": 1,
    "buffer_m": 2000,
    "exclude_dem_coverage_below": 0.8,
    "exclude_water_fraction_above": 0.5,
}
EXPECTED_TERRAIN = {
    "context_buffer_m": 10_000,
    "precision": "float64",
    "neighbor_order": ["E", "SE", "S", "SW", "W", "NW", "N", "NE"],
    "priority_flood": {
        "queue_key": ["filled_elevation", "row", "column"],
        "seed_contract": {
            "processing_domain_boundary": "valid_cells",
            "nodata_adjacent": "valid_cells",
        },
    },
    "flat_resolution": {
        "reference": "Barnes et al. 2014",
        "algorithm_steps": [1, 2, 3, 4, 5, 6, 7],
        "flat_mask": {"name": "FlatMask", "dtype": "integer"},
        "epsilon_dem_modification": False,
    },
}
EXPECTED_DOWNLOAD_INTEGRITY = {
    "partial_suffix": ".part",
    "resume": True,
    "finalize": "atomic",
    "existing_final_mismatch": "fail",
    "final_file_hash": "SHA-256",
    "free_space": {"planned_bytes_multiplier": 2, "extra_bytes": 21_474_836_480},
    "planned_bytes_definition": "selected_asset_content_bytes_sum",
    "source_manifest_runtime_fields": ["bytes", "sha256"],
}


def test_canonical_json_bytes_has_stable_utf8_encoding() -> None:
    value = {"z": "汉字", "a": [1, True, None], "nested": {"b": 2, "a": 1}}

    result = canonical_json_bytes(value)

    assert result == (b'{"a":[1,true,null],"nested":{"a":1,"b":2},"z":"\xe6\xb1\x89\xe5\xad\x97"}')
    assert not result.endswith(b"\n")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_bytes_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"value": value})


@pytest.mark.parametrize("value", [{1: "integer key"}, (1, 2), {"value": {1, 2}}])
def test_canonical_json_bytes_rejects_non_json_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_json_bytes(value)


def test_sha256_helpers_hash_bytes_and_files(tmp_path: Path) -> None:
    content = b"FloodGNN\x00CEUS"
    path = tmp_path / "input.bin"
    path.write_bytes(content)

    assert (
        sha256_bytes(content) == "a7a37bde5475204d556ffe0ffcc86bb9e0bd2a1ab822898eab98fddc891522d6"
    )
    assert sha256_file(path) == sha256_bytes(content)


def test_effective_protocol_id_uses_canonical_lock_hash() -> None:
    assert effective_protocol_id(EXPECTED_BASE_ID, {"b": 2, "a": 1}) == (
        "FloodGNN-CEUS-ZZ-BJ-v1.0-lock-43258cff783f"
    )


def test_effective_protocol_id_rejects_other_base_ids() -> None:
    with pytest.raises(ValueError, match="base_id"):
        effective_protocol_id("FloodGNN", {"b": 2, "a": 1})


def test_protocol_freezes_spatial_and_e0_invariants() -> None:
    loaded = load_protocol(CONFIG_DIR)
    protocol = loaded.protocol

    assert protocol["base_id"] == EXPECTED_BASE_ID
    assert protocol["phase"] == "E0"
    assert protocol["performance_blind"] is True
    assert protocol["terminal_statuses"] == ["success", "failed", "not-estimable"]
    assert protocol["privacy_statuses"] == ["approved", "processing-only", "prohibited"]
    assert protocol["cities"] == ["zhengzhou", "beijing"]
    assert protocol["source_ids"] == SOURCE_ID_ORDER
    assert protocol["canonical_json"] == EXPECTED_CANONICAL_JSON

    grid = protocol["grid"]
    assert grid["resolution_m"] == 200
    assert grid["origin_x"] == "floor(minx/200)*200"
    assert grid["origin_y"] == "ceil(maxy/200)*200"
    assert grid["row_direction"] == "south"
    assert grid["column_direction"] == "east"
    assert grid["index_start"] == 0

    report = protocol["report_assignment"]
    assert report["requires"] == ["admin_covers", "aoi_covers", "final_scorable_grid"]
    assert report["map_to"] == "final_scorable_grid"
    assert report["otherwise"] == "outside_AOI"

    assert protocol["aoi"] == EXPECTED_AOI
    assert protocol["terrain"] == EXPECTED_TERRAIN
    assert protocol["features"]["count"] == 18
    assert protocol["features"]["names"] == EXPECTED_FEATURE_NAMES
    assert protocol["download_integrity"] == EXPECTED_DOWNLOAD_INTEGRITY
    assert set(protocol["e0"]["forbidden_performance_outputs"]) >= {
        "predictions",
        "metrics",
        "model_selection",
    }


def test_city_configs_freeze_event_inventories() -> None:
    cities = load_protocol(CONFIG_DIR).cities
    zhengzhou = cities["zhengzhou"]
    beijing = cities["beijing"]

    assert zhengzhou["city_event"] == "Zhengzhou City_2021_7_19"
    assert zhengzhou["timezone"] == "Asia/Shanghai"
    assert zhengzhou["event_window"] == {
        "local_start": "2021-07-19T19:03:00+08:00",
        "local_end": "2021-08-04T16:58:00+08:00",
        "utc_start": "2021-07-19T11:03:00Z",
        "utc_end": "2021-08-04T08:58:00Z",
    }
    assert zhengzhou["imerg"] == {
        "inventory_start": "2021-07-16T11:03:00Z",
        "expected_granules": 908,
        "inventory_coverage": {
            "start": "2021-07-16T11:00:00Z",
            "end": "2021-08-04T08:59:59.999Z",
            "first_filename": "3B-HHR.MS.MRG.3IMERG.20210716-S110000-E112959.0660.V07B.HDF5",
            "last_filename": "3B-HHR.MS.MRG.3IMERG.20210804-S083000-E085959.0510.V07B.HDF5",
        },
    }
    assert zhengzhou["crs_epsg"] == 32649
    assert zhengzhou["geoboundaries"] == {
        "shapeName": "Zhengzhoushi",
        "shapeID": "17275852B14866353147502",
    }
    assert zhengzhou["worldpop_year"] == 2021
    assert zhengzhou["osm_snapshot"] == {
        "policy": "event-end",
        "timestamp": "2021-08-04T08:58:00Z",
    }

    assert beijing["city_event"] == "Beijing_2023_7_30"
    assert beijing["timezone"] == "Asia/Shanghai"
    assert beijing["event_window"] == {
        "local_start": "2023-07-30T14:50:00+08:00",
        "local_end": "2023-08-11T21:42:00+08:00",
        "utc_start": "2023-07-30T06:50:00Z",
        "utc_end": "2023-08-11T13:42:00Z",
    }
    assert beijing["imerg"] == {
        "inventory_start": "2023-07-27T06:50:00Z",
        "expected_granules": 735,
        "inventory_coverage": {
            "start": "2023-07-27T06:30:00Z",
            "end": "2023-08-11T13:59:59.999Z",
            "first_filename": "3B-HHR.MS.MRG.3IMERG.20230727-S063000-E065959.0390.V07B.HDF5",
            "last_filename": "3B-HHR.MS.MRG.3IMERG.20230811-S133000-E135959.0810.V07B.HDF5",
        },
    }
    assert beijing["crs_epsg"] == 32650
    assert beijing["geoboundaries"] == {
        "shapeName": "Beijingshi",
        "shapeID": "17275852B40522342172572",
    }
    assert beijing["worldpop_year"] == 2023
    assert beijing["osm_snapshot"] == {
        "policy": "event-end",
        "timestamp": "2023-08-11T13:42:00Z",
    }


def test_source_configs_freeze_first_party_locks() -> None:
    sources = load_protocol(CONFIG_DIR).sources

    assert set(sources) == SOURCE_IDS
    crowd = sources["crowdflood_v1"]
    assert crowd["record_id"] == 19921295
    assert crowd["concept_id"] == 19921294
    assert crowd["record_doi"] == "10.5281/zenodo.19921295"
    assert crowd["concept_doi"] == "10.5281/zenodo.19921294"
    assert crowd["filename"] == "CrowdFlood dataset.zip"
    assert crowd["url"] == (
        "https://zenodo.org/api/records/19921295/files/CrowdFlood%20dataset.zip/content"
    )
    assert crowd["bytes"] == 16_290_257
    assert crowd["md5"] == "53b7389ef3495f172d73d1dc54180518"
    assert crowd["license"] == "CC-BY-4.0"
    assert crowd["privacy_status"] == "processing-only"

    boundaries = sources["geoboundaries_chn_adm2"]
    commit = "9469f09592ced973a3448cf66b6100b741b64c0d"
    assert boundaries["commit"] == commit
    assert boundaries["url"] == (
        "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/"
        f"{commit}/releaseData/gbOpen/CHN/ADM2/geoBoundaries-CHN-ADM2.geojson"
    )
    assert boundaries["bytes"] == 7_618_692
    assert boundaries["sha256"] == (
        "2b68d8a808742fc6d7acd769584db960d8fc2c25b9f1d20e3e98c72e9f1c4d34"
    )
    assert boundaries["feature_count"] == 2391
    assert boundaries["crs"] == "OGC:CRS84"
    assert boundaries["selectors"] == {
        "zhengzhou": {"shapeName": "Zhengzhoushi", "shapeID": "17275852B14866353147502"},
        "beijing": {"shapeName": "Beijingshi", "shapeID": "17275852B40522342172572"},
    }
    assert boundaries["license"] == "PDDL-1.0"
    assert boundaries["attribution_license"] == "CC-BY-4.0"

    dem = sources["cop_dem_glo30_dged_2024_1"]
    assert dem["odata_api"] == "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    assert dem["collection"] == "CCM"
    assert dem["dataset"] == "COP-DEM_GLO-30-DGED/2024_1"
    assert dem["native_product"] == "SAR_DGE_30"
    assert dem["product_type"] == "SAR_DGE_30_A4AD"
    assert dem["doi"] == "10.5270/ESA-c5d3d65"
    assert dem["format"] == "GeoTIFF"
    assert dem["horizontal_crs"] == "EPSG:4326"
    assert dem["vertical_crs"] == "EPSG:3855"
    assert dem["authentication"] == {
        "access": "registered-interactive",
        "storage": "in-memory",
    }
    assert dem["version"] == "2024_1"
    assert dem["download_api"] == "https://download.dataspace.copernicus.eu/odata/v1"
    assert set(dem["tiles"]) == {
        "N34E113",
        "N35E113",
        "N39E115",
        "N39E116",
        "N40E115",
        "N40E116",
    }
    assert {tile["bytes"] for tile in dem["tiles"].values()} == {51_897_972}
    assert dem["license_url"] == (
        "https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/"
        "copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf"
    )

    cover = sources["worldcover_2021_v200"]
    assert cover["url_template"] == (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
        "ESA_WorldCover_10m_2021_v200_{ll_tile}_Map.tif"
    )
    assert cover["s3_url_template"] == (
        "s3://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_{ll_tile}_Map.tif"
    )
    assert cover["grid_fgb_url"] == (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/esa_worldcover_grid.fgb"
    )
    assert cover["grid_geojson_url"] == (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/esa_worldcover_grid.geojson"
    )
    assert "Map_Index" not in str(cover)
    assert "kml" not in str(cover).lower()
    assert cover["tile_count"] == 2651
    assert cover["classes"]["built_up"] == 50

    imerg = sources["imerg_v07b"]
    assert imerg["short_name"] == "GPM_3IMERGHH"
    assert imerg["version"] == "07"
    assert imerg["path_product"] == "GPM_3IMERGHH.07"
    assert imerg["file_version"] == "V07B"
    assert imerg["cmr_concept_id"] == "C2723754847-GES_DISC"
    assert imerg["doi"] == "10.5067/GPM/IMERG/3B-HH/07"
    assert imerg["authentication"] == {"client": "earthaccess", "persist": False}
    assert imerg["retrieval"] == {"granules": "full", "server_subset": False}
    assert imerg["variable"] == {"path": "Grid/precipitation", "units": "mm h-1"}
    assert imerg["interval_minutes"] == 30
    assert imerg["license"] == "NASA Earth Science Data and Information Policy"
    assert imerg["license_url"] == "https://gpm.nasa.gov/data/policy"

    osm = sources["ohsome_v1"]
    assert osm["endpoint"] == "https://api.ohsome.org/v1/elements/geometry"
    assert osm["method"] == "POST"
    assert osm["timestamp_format"] == "UTC Z"
    assert osm["properties"] == ["tags", "metadata"]
    assert osm["clipGeometry"] is False
    assert osm["spatial_partition_degrees"] == 0.25
    assert osm["filters"]["roads"] == (
        "geometry:line and highway=* and not (highway in (construction, proposed, abandoned, "
        "razed, platform, raceway, elevator))"
    )
    assert osm["filters"]["waterways"] == (
        "geometry:line and waterway in (river, stream, canal, drain, ditch)"
    )
    assert "building" not in str(osm).lower()
    assert osm["deduplicate_by"] == ["@osmId", "@version"]

    population = sources["worldpop_r2024b"]
    assert population["release"] == "R2024B"
    assert population["version"] == "v1"
    assert population["dataset"] == "population_count"
    assert population["country"] == "CHN"
    assert population["constrained"] is False
    assert population["resolution_m"] == 100
    assert population["resolution_arc_seconds"] == 3
    assert population["units"] == "persons_per_grid_cell"
    assert population["status"] == "beta"
    assert population["assets"]["2021"]["url"] == (
        "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/2021/CHN/v1/"
        "100m/unconstrained/chn_pop_2021_UC_100m_R2024B_v1.tif"
    )
    assert population["assets"]["2021"]["bytes"] == 2_092_461_699
    assert population["assets"]["2021"]["observed_etag"] == '"7cb86e83-62dc566df9580"'
    assert population["assets"]["2021"]["observed_last_modified"] == ("2025-02-10T08:21:58Z")
    assert population["assets"]["2023"]["url"].endswith(
        "/2023/CHN/v1/100m/unconstrained/chn_pop_2023_UC_100m_R2024B_v1.tif"
    )
    assert population["assets"]["2023"]["bytes"] == 2_138_386_657
    assert population["assets"]["2023"]["observed_etag"] == '"7f7530e1-62e079e8a4389"'
    assert population["assets"]["2023"]["observed_last_modified"] == ("2025-02-13T15:21:59Z")
    assert population["etag_is_checksum"] is False
    assert population["final_sha256_requirement"] == "download-time"
    assert population["asset_license_status"] == "unresolved_per_asset"
    assert population["general_license"] == "CC-BY-4.0"
    assert "ODbL" in population["license_caveat"]
    assert "R2025A" not in str(population)

    assert {source["privacy_status"] for source in sources.values()} == {
        "approved",
        "processing-only",
    }


def _copy_configs(tmp_path: Path) -> Path:
    destination = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, destination)
    return destination


def _edit_yaml(path: Path, edit: object) -> None:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    edit(value)  # type: ignore[operator]
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _set_nested(value: dict[str, object], path: tuple[str, ...], replacement: object) -> None:
    current = value
    for key in path[:-1]:
        child = current[key]
        assert isinstance(child, dict)
        current = child
    current[path[-1]] = replacement


def test_load_protocol_hash_is_deterministic() -> None:
    first = load_protocol(CONFIG_DIR)
    second = load_protocol(CONFIG_DIR)

    assert first.input_sha256 == second.input_sha256
    assert first.input_sha256 == EXPECTED_CONFIG_SHA256
    assert len(first.input_sha256) == 64
    assert set(first.input_sha256) <= set("0123456789abcdef")


def test_load_protocol_rejects_bad_privacy_status(tmp_path: Path) -> None:
    config_dir = _copy_configs(tmp_path)
    sources_path = config_dir / "sources.yaml"
    _edit_yaml(
        sources_path,
        lambda value: value["sources"]["crowdflood_v1"].update(privacy_status="private"),
    )

    with pytest.raises(ValueError, match="privacy_status"):
        load_protocol(config_dir)


@pytest.mark.parametrize(
    "field_name",
    [
        "secret",
        "token",
        "password",
        "totp",
        "cookie",
        "credential",
        "credentials",
        "api_key",
        "authorization",
        "private_key",
        "access_key",
        "client_secret",
        "refresh_token",
        "access_token",
        "apiKey",
        "apikey",
        "x-api-key",
        "privateKey",
        "accessKey",
        "clientSecret",
        "refreshToken",
        "accessToken",
        "pwd",
        "passwd",
        "auth",
        "bearer",
        "sessionId",
        "sessionKey",
    ],
)
def test_load_protocol_rejects_secret_values(tmp_path: Path, field_name: str) -> None:
    config_dir = _copy_configs(tmp_path)
    sources_path = config_dir / "sources.yaml"
    _edit_yaml(sources_path, lambda value: value.update({field_name: "actual-secret-value"}))

    with pytest.raises(ValueError, match="secret|token|credential|sensitive"):
        load_protocol(config_dir)


@pytest.mark.parametrize(
    "field_name",
    ["pwd", "passwd", "auth", "bearer", "sessionId", "sessionKey"],
)
def test_load_protocol_rejects_nested_sensitive_alias(
    tmp_path: Path,
    field_name: str,
) -> None:
    config_dir = _copy_configs(tmp_path)
    sources_path = config_dir / "sources.yaml"
    _edit_yaml(
        sources_path,
        lambda value: value["sources"]["crowdflood_v1"].update(
            metadata={field_name: "actual-secret-value"}
        ),
    )

    with pytest.raises(ValueError, match="sensitive"):
        load_protocol(config_dir)


def test_valid_metadata_keys_are_not_sensitive() -> None:
    sources = load_protocol(CONFIG_DIR).sources

    assert sources["cop_dem_glo30_dged_2024_1"]["product_type"] == "SAR_DGE_30_A4AD"
    assert sources["ohsome_v1"]["timestamp_format"] == "UTC Z"
    assert sources["cop_dem_glo30_dged_2024_1"]["authentication"] == {
        "access": "registered-interactive",
        "storage": "in-memory",
    }


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("canonical_json", {**EXPECTED_CANONICAL_JSON, "allow_nan": True}),
        ("features", {"count": 18, "names": list(reversed(EXPECTED_FEATURE_NAMES))}),
        ("aoi", {**EXPECTED_AOI, "exclude_dem_coverage_below": 0.7}),
        ("terrain", {**EXPECTED_TERRAIN, "precision": "float32"}),
        ("download_integrity", {**EXPECTED_DOWNLOAD_INTEGRITY, "resume": False}),
    ],
)
def test_load_protocol_rejects_changed_frozen_contract(
    tmp_path: Path,
    key: str,
    replacement: object,
) -> None:
    config_dir = _copy_configs(tmp_path)
    protocol_path = config_dir / "protocol.yaml"
    _edit_yaml(protocol_path, lambda value: value.update({key: replacement}))

    with pytest.raises(ValueError, match=key):
        load_protocol(config_dir)


def test_load_protocol_rejects_city_mismatch(tmp_path: Path) -> None:
    config_dir = _copy_configs(tmp_path)
    protocol_path = config_dir / "protocol.yaml"
    _edit_yaml(protocol_path, lambda value: value.update(cities=["zhengzhou", "shanghai"]))

    with pytest.raises(ValueError, match="cities"):
        load_protocol(config_dir)


def test_load_protocol_rejects_extra_city_file(tmp_path: Path) -> None:
    config_dir = _copy_configs(tmp_path)
    shutil.copy(config_dir / "cities" / "beijing.yaml", config_dir / "cities" / "shanghai.yaml")

    with pytest.raises(ValueError, match="cities"):
        load_protocol(config_dir)


def test_load_protocol_rejects_duplicate_top_level_yaml_key(tmp_path: Path) -> None:
    config_dir = _copy_configs(tmp_path)
    protocol_path = config_dir / "protocol.yaml"
    text = protocol_path.read_text(encoding="utf-8")
    protocol_path.write_text(f"{text}\nphase: E0\n", encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        load_protocol(config_dir)


def test_load_protocol_rejects_duplicate_nested_yaml_key(tmp_path: Path) -> None:
    config_dir = _copy_configs(tmp_path)
    sources_path = config_dir / "sources.yaml"
    text = sources_path.read_text(encoding="utf-8")
    duplicate = "    privacy_status: processing-only\n    privacy_status: processing-only\n"
    text = text.replace("    privacy_status: processing-only\n", duplicate, 1)
    sources_path.write_text(text, encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        load_protocol(config_dir)


def test_protocol_loader_does_not_modify_global_safe_loader() -> None:
    assert yaml.safe_load("key: first\nkey: second\n") == {"key": "second"}


def test_load_protocol_rejects_extra_sources_root_key(tmp_path: Path) -> None:
    config_dir = _copy_configs(tmp_path)
    sources_path = config_dir / "sources.yaml"
    _edit_yaml(sources_path, lambda value: value.update(metadata={}))

    with pytest.raises(ValueError, match="sources.yaml root"):
        load_protocol(config_dir)


def test_load_protocol_rejects_reordered_source_ids(tmp_path: Path) -> None:
    config_dir = _copy_configs(tmp_path)
    protocol_path = config_dir / "protocol.yaml"
    _edit_yaml(
        protocol_path,
        lambda value: value.update(source_ids=list(reversed(value["source_ids"]))),
    )

    with pytest.raises(ValueError, match="source IDs"):
        load_protocol(config_dir)


def test_load_protocol_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    config_dir = _copy_configs(tmp_path)
    protocol_path = config_dir / "protocol.yaml"
    _edit_yaml(
        protocol_path,
        lambda value: value.update(source_ids=[*value["source_ids"], value["source_ids"][0]]),
    )

    with pytest.raises(ValueError, match="source IDs"):
        load_protocol(config_dir)


@pytest.mark.parametrize(
    ("relative_path", "key_path", "replacement"),
    [
        (
            "cities/beijing.yaml",
            ("event_window", "utc_end"),
            "2023-08-11T13:43:00Z",
        ),
        (
            "sources.yaml",
            ("sources", "worldcover_2021_v200", "url_template"),
            "https://invalid.example/worldcover.tif",
        ),
        (
            "cities/zhengzhou.yaml",
            ("geoboundaries", "shapeID"),
            "wrong-shape-id",
        ),
        (
            "cities/beijing.yaml",
            ("imerg", "expected_granules"),
            734,
        ),
    ],
)
def test_load_protocol_rejects_frozen_config_mutation(
    tmp_path: Path,
    relative_path: str,
    key_path: tuple[str, ...],
    replacement: object,
) -> None:
    config_dir = _copy_configs(tmp_path)
    config_path = config_dir / relative_path
    _edit_yaml(config_path, lambda value: _set_nested(value, key_path, replacement))

    with pytest.raises(ValueError, match="frozen config digest mismatch"):
        load_protocol(config_dir)
