"""Load and validate the frozen FloodGNN CEUS protocol inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes

EXPECTED_BASE_ID = "FloodGNN-CEUS-ZZ-BJ-v1.0"
EXPECTED_CITIES = ("zhengzhou", "beijing")
EXPECTED_SOURCES = (
    "crowdflood_v1",
    "geoboundaries_chn_adm2",
    "cop_dem_glo30_dged_2024_1",
    "worldcover_2021_v200",
    "imerg_v07b",
    "ohsome_v1",
    "worldpop_r2024b",
)
EXPECTED_CONFIG_SHA256 = "903220eeba99b14b1ba70c6a53fa850fa005d41e1411eb71e9a403ddae197627"
ALLOWED_PRIVACY_STATUSES = {"approved", "processing-only", "prohibited"}
SENSITIVE_KEY_FRAGMENTS = {
    "secret",
    "token",
    "password",
    "totp",
    "cookie",
    "credential",
    "apikey",
    "authorization",
    "privatekey",
    "accesskey",
}
SENSITIVE_KEY_ALIASES = {"pwd", "passwd", "auth", "bearer", "sessionid", "sessionkey"}
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


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class LoadedProtocol:
    """Validated protocol inputs and their canonical aggregate digest."""

    protocol: dict[str, Any]
    sources: dict[str, Any]
    cities: dict[str, dict[str, Any]]
    input_sha256: str


def _read_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path.name} must contain a string-keyed mapping")
    return value


def _reject_sensitive_values(value: object, location: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            is_sensitive = normalized_key in SENSITIVE_KEY_ALIASES or any(
                part in normalized_key for part in SENSITIVE_KEY_FRAGMENTS
            )
            if is_sensitive and item is not None and item != "":
                raise ValueError(f"sensitive value is forbidden at {location}.{key}")
            _reject_sensitive_values(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_values(item, f"{location}[{index}]")


def _validate_protocol(
    protocol: dict[str, Any],
    sources: dict[str, Any],
    cities: dict[str, dict[str, Any]],
) -> None:
    if protocol.get("base_id") != EXPECTED_BASE_ID:
        raise ValueError("invalid protocol base_id")
    if protocol.get("cities") != list(EXPECTED_CITIES) or set(cities) != set(EXPECTED_CITIES):
        raise ValueError("protocol cities must match exactly zhengzhou and beijing")
    if protocol.get("source_ids") != list(EXPECTED_SOURCES):
        raise ValueError("protocol source IDs must match the frozen order without duplicates")
    if set(sources) != set(EXPECTED_SOURCES):
        raise ValueError("protocol source IDs are incomplete")
    exact_contracts = {
        "canonical_json": EXPECTED_CANONICAL_JSON,
        "features": {"count": 18, "names": EXPECTED_FEATURE_NAMES},
        "aoi": EXPECTED_AOI,
        "terrain": EXPECTED_TERRAIN,
        "download_integrity": EXPECTED_DOWNLOAD_INTEGRITY,
    }
    for key, expected in exact_contracts.items():
        if protocol.get(key) != expected:
            raise ValueError(f"invalid {key} contract")
    grid = protocol.get("grid", {})
    if (
        grid.get("resolution_m") != 200
        or grid.get("row_direction") != "south"
        or grid.get("column_direction") != "east"
    ):
        raise ValueError("invalid grid definition")
    expected_crs = {"zhengzhou": 32649, "beijing": 32650}
    for city_id, crs_epsg in expected_crs.items():
        city = cities[city_id]
        if city.get("city_id") != city_id or city.get("crs_epsg") != crs_epsg:
            raise ValueError(f"invalid city definition for {city_id}")
    for source_id, source in sources.items():
        status = source.get("privacy_status")
        if status not in ALLOWED_PRIVACY_STATUSES:
            raise ValueError(f"invalid privacy_status for {source_id}: {status}")


def load_protocol(config_dir: str | Path) -> LoadedProtocol:
    """Load, validate, and hash the complete protocol configuration."""
    directory = Path(config_dir)
    protocol = _read_mapping(directory / "protocol.yaml")
    sources_document = _read_mapping(directory / "sources.yaml")
    _reject_sensitive_values(protocol)
    _reject_sensitive_values(sources_document)
    if set(sources_document) != {"sources"}:
        raise ValueError("sources.yaml root must contain exactly sources")
    sources = sources_document["sources"]
    if not isinstance(sources, dict):
        raise TypeError("sources.yaml must contain a sources mapping")
    city_dir = directory / "cities"
    city_ids = {path.stem for path in city_dir.glob("*.yaml")}
    if city_ids != set(EXPECTED_CITIES):
        raise ValueError("config cities must be exactly zhengzhou and beijing")
    cities = {city_id: _read_mapping(city_dir / f"{city_id}.yaml") for city_id in EXPECTED_CITIES}
    inputs = {"protocol": protocol, "sources": sources, "cities": cities}
    _reject_sensitive_values(cities)
    _validate_protocol(protocol, sources, cities)
    input_sha256 = sha256_bytes(canonical_json_bytes(inputs))
    if input_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError("frozen config digest mismatch")
    return LoadedProtocol(protocol, sources, cities, input_sha256)


def effective_protocol_id(base_id: str, lock_inputs: object) -> str:
    """Bind a base protocol identifier to canonical lock inputs."""
    if base_id != EXPECTED_BASE_ID:
        raise ValueError("invalid protocol base_id")
    digest = sha256_bytes(canonical_json_bytes(lock_inputs))
    return f"{base_id}-lock-{digest[:12]}"
