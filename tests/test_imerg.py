from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from floodgnn_ceus import imerg
from floodgnn_ceus.imerg import (
    ImergGranule,
    nearest_axis_indices,
    rainfall_features,
    validate_inventory,
)


def test_nearest_axis_indices_break_ties_toward_smaller_index() -> None:
    axis = np.array([0.0, 0.1, 0.2])

    indices = nearest_axis_indices(axis, np.array([0.05, 0.16, -1.0, 1.0]))

    np.testing.assert_array_equal(indices, [0, 2, 0, 2])


def test_rainfall_features_use_midpoints_amounts_and_complete_windows() -> None:
    event_start = datetime(2021, 7, 19, 11, 3, tzinfo=UTC)
    midpoints = np.array(
        [
            event_start - timedelta(hours=72) + timedelta(minutes=15 + 30 * index)
            for index in range(157)
        ],
        dtype=object,
    )
    event_end = midpoints[-1]
    rates = np.ones((len(midpoints), 2), dtype=np.float64)
    rates[144, 1] = np.nan

    event_total, max_six_hours, antecedent = rainfall_features(
        rates,
        midpoints,
        event_start=event_start,
        event_end=event_end,
    )

    np.testing.assert_allclose(antecedent, [72.0, 72.0])
    np.testing.assert_allclose(event_total[0], 6.5)
    assert np.isnan(event_total[1])
    np.testing.assert_allclose(max_six_hours, [6.0, 6.0])


def test_inventory_requires_uninterrupted_v07b_sequence() -> None:
    first = ImergGranule(
        "test",
        "a.V07B.HDF5",
        "collection:a.V07B.HDF5",
        "https://example.test/a.V07B.HDF5",
        "2021-01-01T00:00:00.000Z",
        "2021-01-01T00:29:59.999Z",
        10,
    )
    second = ImergGranule(
        "test",
        "b.V07B.HDF5",
        "collection:b.V07B.HDF5",
        "https://example.test/b.V07B.HDF5",
        "2021-01-01T00:30:00.000Z",
        "2021-01-01T00:59:59.999Z",
        10,
    )

    validate_inventory(
        [first, second],
        expected_count=2,
        first_filename=first.filename,
        last_filename=second.filename,
    )
    broken = ImergGranule(**{**second.__dict__, "beginning": "2021-01-01T01:00:00.000Z"})
    with pytest.raises(ValueError, match="frozen half-hour"):
        validate_inventory(
            [first, broken],
            expected_count=2,
            first_filename=first.filename,
            last_filename=broken.filename,
        )


def test_hdf5_reader_selects_only_requested_native_pixels(tmp_path: Path) -> None:
    path = tmp_path / "granule.HDF5"
    dataset = xr.Dataset(
        {
            "precipitation": (
                ("time", "lat", "lon"),
                np.array([[[1.0, 2.0], [3.0, -9999.0]]], dtype=np.float32),
            )
        },
        coords={"time": [0], "lat": [0.0, 0.1], "lon": [1.0, 1.1]},
    )
    dataset["time"].attrs.update(
        {
            "units": "seconds since 1980-01-06 00:00:00 UTC",
            "calendar": "julian",
        }
    )
    dataset.to_netcdf(path, group="Grid", engine="h5netcdf")

    latitude, longitude = imerg._axes(path)
    rates = imerg._read_rates(
        [path],
        np.array([[0, 1], [1, 0], [1, 1]], dtype=np.int32),
        latitude,
        longitude,
    )

    np.testing.assert_allclose(rates[0, :2], [2.0, 3.0])
    assert np.isnan(rates[0, 2])
