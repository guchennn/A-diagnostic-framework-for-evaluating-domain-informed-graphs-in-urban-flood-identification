"""Protocol invariants for label-independent, degree/distance matched controls."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from floodgnn_ceus.topology_controls import build_control


def swappable():
    # Both diagonal assignments have the same in/outdegrees and distance counts.
    grid = pd.DataFrame(
        {"grid_id": ["a", "b", "c", "d"], "row": [0, 1, 2, 1], "column": [0, -1, 0, 1]}
    )
    edges = pd.DataFrame({"source_grid_id": ["a", "c"], "target_grid_id": ["b", "d"]})
    return grid, edges


def test_control_preserves_exact_degrees_distances_and_changes_neighbours():
    grid, edges = swappable()
    # Move away from tile boundaries without changing geometry.
    grid[["row", "column"]] += 8
    result, audit = build_control(edges, grid, seed_name="invariants", config={})
    assert audit["accepted"]
    assert audit["changed_fraction"] == 1
    assert audit["distance_counts_preserved"]
    assert Counter(result.source_grid_id) == Counter(edges.source_grid_id)
    assert Counter(result.target_grid_id) == Counter(edges.target_grid_id)
    assert set(result.itertuples(index=False, name=None)) == {("a", "d"), ("c", "b")}


def test_control_does_not_read_labels_or_other_attributes_and_is_seeded():
    grid, edges = swappable()
    grid[["row", "column"]] += 8
    a, audit = build_control(edges, grid, seed_name="stable", config={})
    grid["label"] = [1, 1, 0, 0]
    grid["rainfall"] = [np.nan, 999, -100, 0]
    b, second = build_control(
        edges.sample(frac=1, random_state=1), grid, seed_name="stable", config={}
    )
    pd.testing.assert_frame_equal(a, b)
    assert audit["control_edge_sha256"] == second["control_edge_sha256"]


def test_cross_tile_edges_stay_fixed_and_insufficient_changes_are_rejected():
    grid = pd.DataFrame({"grid_id": ["a", "b"], "row": [31, 32], "column": [2, 2]})
    edges = pd.DataFrame({"source_grid_id": ["a"], "target_grid_id": ["b"]})
    result, audit = build_control(edges, grid, seed_name="crossing", config={})
    pd.testing.assert_frame_equal(result, edges)
    assert audit["cross_tile_edges"] == 1
    assert not audit["accepted"]


def test_multiple_outgoing_input_edges_are_rejected():
    grid, edges = swappable()
    edges.loc[1, "source_grid_id"] = "a"
    with pytest.raises(ValueError, match="outdegree"):
        build_control(edges, grid, seed_name="invalid", config={})


def test_nonneighbour_input_edges_are_rejected():
    grid, edges = swappable()
    grid.loc[grid.grid_id.eq("b"), "row"] = 20
    with pytest.raises(ValueError, match="eight-neighbours"):
        build_control(edges, grid, seed_name="invalid", config={})
