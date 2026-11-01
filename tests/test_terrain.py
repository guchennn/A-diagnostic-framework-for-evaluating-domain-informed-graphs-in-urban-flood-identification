from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from floodgnn_ceus import terrain
from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.sources import RAW_SOURCE_COLUMNS, SourceRecord
from floodgnn_ceus.terrain import (
    CdseDownloadSession,
    apply_dem_coverage_gate,
    build_dsm_download_url,
    combine_dem_observations,
    request_cdse_access_token,
    select_glo30_tiles,
    write_dem_manifest,
)


def test_glo30_tiles_follow_each_projected_aoi_plus_ten_kilometres() -> None:
    zhengzhou = gpd.GeoDataFrame(
        geometry=[box(113.4, 34.6, 113.8, 34.95)],
        crs=4326,
    ).to_crs(32649)
    beijing = gpd.GeoDataFrame(
        geometry=[box(115.38, 39.48, 116.62, 40.15)],
        crs=4326,
    ).to_crs(32650)

    selected = select_glo30_tiles(
        {"zhengzhou": zhengzhou, "beijing": beijing},
        context_buffer_m=10_000,
    )

    assert selected == {
        "zhengzhou": ("N34E113", "N35E113"),
        "beijing": ("N39E115", "N39E116", "N40E115", "N40E116"),
    }


def test_dsm_download_url_targets_the_leaf_value() -> None:
    source = {"download_api": "https://download.example/odata/v1"}
    tile = {
        "product_id": "679359b8-91a0-40ba-a5de-40fc4c70e526",
        "product_name": "DEM1_example.DEM",
    }

    url = build_dsm_download_url(source, "N34E113", tile)

    assert url.endswith("/Nodes(Copernicus_DSM_10_N34_00_E113_00_DEM.tif)/$value")
    assert not url.endswith("/Nodes/$value")


def test_cdse_token_request_uses_only_the_supplied_secure_prompts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Response:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"access_token": "memory-only-access-token"}

        def close(self) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.data: dict[str, str] | None = None

        def post(self, url: str, *, data: dict[str, str], timeout: tuple[int, int]) -> Response:
            assert url == "https://identity.example/token"
            assert timeout == (10, 60)
            self.data = data
            return Response()

    answers = iter(("researcher", "private-password", "123456"))
    session = Session()

    token = request_cdse_access_token(
        session,
        "https://identity.example/token",
        prompt=lambda _: next(answers),
    )

    assert token == "memory-only-access-token"
    assert session.data == {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": "researcher",
        "password": "private-password",
        "totp": "123456",
    }
    captured = capsys.readouterr()
    assert "private-password" not in captured.out + captured.err
    assert "memory-only-access-token" not in captured.out + captured.err


def test_cdse_token_request_omits_blank_totp() -> None:
    class Response:
        status_code = 200

        def json(self) -> dict[str, str]:
            return {"access_token": "memory-only-access-token"}

        def close(self) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.data: dict[str, str] | None = None

        def post(self, _url: str, *, data: dict[str, str], **_: object) -> Response:
            self.data = data
            return Response()

    answers = iter(("researcher", "private-password", ""))
    session = Session()

    request_cdse_access_token(
        session,
        "https://identity.example/token",
        prompt=lambda _: next(answers),
    )

    assert session.data is not None
    assert "totp" not in session.data


def test_cdse_redirect_does_not_forward_the_bearer_token() -> None:
    class Response:
        def __init__(self, status: int, url: str, location: str | None = None) -> None:
            self.status_code = status
            self.url = url
            self.headers = {} if location is None else {"Location": location}

        def close(self) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, str]]] = []

        def get(self, url: str, *, headers: dict[str, str], **_: object) -> Response:
            self.calls.append((url, headers.copy()))
            if len(self.calls) == 1:
                return Response(302, url, "https://objects.example/file?signature=private")
            return Response(200, url)

    base = Session()
    session = CdseDownloadSession(base, "memory-only-access-token")

    response = session.get(
        "https://download.dataspace.copernicus.eu/odata/v1/Products(x)/$value",
        headers={"Range": "bytes=5-"},
        stream=True,
        allow_redirects=False,
    )

    assert response.status_code == 200
    assert base.calls[0][1]["Authorization"] == "Bearer memory-only-access-token"
    assert base.calls[1][1] == {"Range": "bytes=5-"}


def test_dem_observations_are_coverage_weighted_in_float64() -> None:
    elevations = [
        np.array([[10.0, np.nan], [20.0, 40.0]], dtype=np.float64),
        np.array([[30.0, 50.0], [np.nan, 60.0]], dtype=np.float64),
    ]
    coverages = [
        np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64),
        np.array([[1.0, 0.5], [0.0, 0.5]], dtype=np.float64),
    ]

    elevation, coverage = combine_dem_observations(elevations, coverages)

    np.testing.assert_allclose(elevation, [[20.0, 50.0], [20.0, 50.0]])
    np.testing.assert_allclose(coverage, [[1.0, 0.5], [0.5, 1.0]])
    assert elevation.dtype == np.float64
    assert coverage.dtype == np.float64


def test_dem_coverage_gate_is_inclusive_at_eighty_percent() -> None:
    candidate = gpd.GeoDataFrame(
        {
            "city_id": ["test"] * 4,
            "grid_id": ["g0", "g1", "g2", "g3"],
            "row": [0, 0, 1, 1],
            "column": [0, 1, 0, 1],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)],
        crs=3857,
    )

    final = apply_dem_coverage_gate(
        candidate,
        np.array([0.799, 0.8, 1.0, np.nan], dtype=np.float64),
        threshold=0.8,
    )

    assert final["grid_id"].tolist() == ["g1", "g2"]
    assert final["dem_valid_fraction"].tolist() == [0.8, 1.0]
    assert final["grid_stage"].eq("final_scorable").all()
    assert final["aoi_stage"].eq("final_scorable").all()
    assert final["is_final_scorable"].all()


def test_dem_manifest_is_deterministic_and_uses_the_source_schema(tmp_path: Path) -> None:
    record = SourceRecord(
        source_id="cop_dem_glo30_dged_2024_1",
        city_id="all",
        asset_key="N34E113",
        provider="Copernicus Data Space Ecosystem",
        version_or_release="2024_1",
        source_url_or_doi="10.5270/ESA-c5d3d65",
        license="custom",
        privacy_status="approved",
        reference_date_or_period="2024_1",
        relative_path="cop_dem_glo30_dged_2024_1/2024_1/N34E113/dem.tif",
        retrieved_at="2026-08-08T00:00:00Z",
        bytes=4,
        sha256="a" * 64,
        upstream_checksum_algorithm=None,
        upstream_checksum=None,
        request_sha256="b" * 64,
        api_version="OData v1",
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"

    write_dem_manifest([record], first)
    write_dem_manifest([record], second)

    assert sha256_file(first) == sha256_file(second)
    assert pq.read_table(first).column_names == list(RAW_SOURCE_COLUMNS)


def test_priority_flood_fills_a_closed_sink_and_preserves_nodata() -> None:
    dem = np.array(
        [
            [5.0, 5.0, 5.0, np.nan],
            [5.0, 1.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0],
        ],
        dtype=np.float64,
    )

    filled = terrain.priority_flood(dem)

    assert filled[1, 1] == 5.0
    assert np.isnan(filled[0, 3])
    assert dem[1, 1] == 1.0
    assert filled.dtype == np.float64
    _, _, directions = terrain.resolve_d8_flats(filled)
    assert directions[1, 1] != -1


def test_flat_mask_drains_to_boundaries_without_modifying_the_filled_dem() -> None:
    filled = np.ones((3, 3), dtype=np.float64)
    original = filled.copy()

    flat_mask, labels, directions = terrain.resolve_d8_flats(filled)

    np.testing.assert_array_equal(filled, original)
    assert flat_mask.dtype == np.int64
    assert labels.dtype == np.int64
    assert labels[1, 1] > 0
    assert flat_mask[1, 1] > flat_mask[1, 2]
    assert directions[1, 1] == 0  # E is the frozen first direction on a complete tie.
    boundary_rows = [0, 0, 0, 1, 1, 2, 2, 2]
    boundary_columns = [0, 1, 2, 0, 2, 0, 1, 2]
    assert np.all(directions[boundary_rows, boundary_columns] == -2)


def test_d8_uses_east_first_when_cardinal_descents_are_tied() -> None:
    surface = np.array(
        [
            [11.0, 11.0, 11.0],
            [11.0, 10.0, 9.0],
            [11.0, 9.0, 11.0],
        ],
        dtype=np.float64,
    )

    _, _, directions = terrain.resolve_d8_flats(surface)

    assert directions[1, 1] == 0


def test_flow_accumulation_includes_each_cell_itself() -> None:
    directions = np.array([[0, 0, -2], [-3, -2, -2]], dtype=np.int16)

    accumulation = terrain.flow_accumulation(directions)

    np.testing.assert_array_equal(accumulation[0], [1, 2, 3])
    assert accumulation[1, 0] == 0
    assert accumulation.dtype == np.int64


def test_terrain_features_follow_the_frozen_operators() -> None:
    surface = np.tile(np.arange(4.0, -1.0, -1.0), (5, 1))
    directions = np.full((5, 5), 0, dtype=np.int16)
    directions[:, -1] = -2
    accumulation = terrain.flow_accumulation(directions)

    features = terrain.terrain_feature_arrays(surface, accumulation, resolution_m=200)

    assert list(features) == [
        "elevation_mean",
        "slope_mean",
        "relative_elevation_1km",
        "local_relief_1km",
        "log_flow_accumulation",
        "twi",
    ]
    assert features["elevation_mean"].dtype == np.float64
    assert features["slope_mean"][2, 2] == pytest.approx(np.degrees(np.arctan(1.0 / 200.0)))
    assert features["relative_elevation_1km"][2, 2] == pytest.approx(0.0)
    assert features["relative_elevation_1km"][0, 0] == pytest.approx(1.0)
    assert features["local_relief_1km"][2, 2] == pytest.approx(4.0)
    assert np.isnan(features["local_relief_1km"][0, 0])
    assert features["log_flow_accumulation"][2, 2] == pytest.approx(np.log1p(3 * 0.04))
    expected_twi = np.log((3 * 200.0 + 1.0) / np.tan(np.radians(0.5)))
    assert features["twi"][2, 2] == pytest.approx(expected_twi)
