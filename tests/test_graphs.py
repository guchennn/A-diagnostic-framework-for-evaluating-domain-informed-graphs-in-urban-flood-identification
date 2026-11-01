from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from floodgnn_ceus import graphs


def grid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "grid_id": ["a", "b", "c", "d", "isolated"],
            "row": [0, 0, 1, 1, 3],
            "column": [0, 1, 0, 1, 3],
        }
    )


def d8_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_grid_id": ["a", "b", "c"],
            "target_grid_id": ["b", "d", "d"],
        }
    )


def diagonal_swap() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "grid_id": ["a", "b", "c", "d"],
            "row": [0, 1, 2, 1],
            "column": [0, -1, 0, 1],
        }
    )
    edges = pd.DataFrame(
        {
            "source_grid_id": ["a", "c"],
            "target_grid_id": ["b", "d"],
            "distance_class": ["diagonal", "diagonal"],
        }
    )
    return nodes, edges


def test_build_base_edges_freezes_g1_g2_and_g4() -> None:
    result = graphs.build_base_edges(grid(), d8_edges())

    assert set(result["graph_id"]) == {"G1", "G2", "G4"}
    assert len(result.loc[result["graph_id"].eq("G1")]) == 12
    g1 = set(
        result.loc[result["graph_id"].eq("G1"), ["source_grid_id", "target_grid_id"]].itertuples(
            index=False, name=None
        )
    )
    assert all((target, source) in g1 for source, target in g1)

    g2 = result.loc[result["graph_id"].eq("G2")]
    assert list(g2[["source_grid_id", "target_grid_id"]].itertuples(index=False, name=None)) == [
        ("a", "b"),
        ("b", "d"),
        ("c", "d"),
    ]
    assert set(g2["distance_class"]) == {"rook"}

    g4 = set(
        result.loc[result["graph_id"].eq("G4"), ["source_grid_id", "target_grid_id"]].itertuples(
            index=False, name=None
        )
    )
    assert g4 == {(target, source) for source, target in d8_edges().itertuples(index=False)}
    assert not result["graph_id"].eq("G0").any()


def test_build_base_edges_rejects_multiple_g2_outgoing_edges() -> None:
    invalid = pd.concat(
        [
            d8_edges(),
            pd.DataFrame({"source_grid_id": ["a"], "target_grid_id": ["c"]}),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="outdegree"):
        graphs.build_base_edges(grid(), invalid)


def test_cut_edges_removes_cross_partition_and_buffer_incident_edges() -> None:
    edges = pd.DataFrame(
        {
            "graph_id": ["G2"] * 4,
            "source_grid_id": ["a", "a", "b", "c"],
            "target_grid_id": ["b", "c", "c", "d"],
            "distance_class": ["rook", "rook", "diagonal", "rook"],
        }
    )
    partitions = pd.DataFrame(
        {
            "grid_id": ["a", "b", "c", "d"],
            "partition": ["train", "train", "buffer", "test"],
        }
    )

    result = graphs.cut_edges(edges, partitions)

    assert result[["source_grid_id", "target_grid_id", "partition"]].to_dict("records") == [
        {"source_grid_id": "a", "target_grid_id": "b", "partition": "train"}
    ]


def test_cut_edges_rejects_duplicate_partition_keys() -> None:
    partitions = pd.DataFrame({"grid_id": ["a", "a"], "partition": ["train", "test"]})

    with pytest.raises(ValueError, match="unique"):
        graphs.cut_edges(d8_edges(), partitions)


def test_g5_is_deterministic_and_preserves_degree_and_slot_class() -> None:
    nodes, edges = diagonal_swap()

    first, first_audit = graphs.rewire_g5(
        edges,
        nodes,
        seed_name="protocol/city/0/test/G5/1",
        accepted_swap_multiplier=1,
        attempt_multiplier=20,
        overlap_max=1.0,
    )
    second, second_audit = graphs.rewire_g5(
        edges,
        nodes,
        seed_name="protocol/city/0/test/G5/1",
        accepted_swap_multiplier=1,
        attempt_multiplier=20,
        overlap_max=1.0,
    )

    assert first is not None
    assert second is not None
    pd.testing.assert_frame_equal(first, second)
    assert first_audit == second_audit
    assert first_audit["status"] == "success"
    assert first_audit["accepted_swaps"] == 2
    assert Counter(first["source_grid_id"]) == Counter(edges["source_grid_id"])
    assert Counter(first["target_grid_id"]) == Counter(edges["target_grid_id"])
    assert first.sort_values("slot_id")["distance_class"].tolist() == [
        "diagonal",
        "diagonal",
    ]


def test_g5_marks_small_or_high_overlap_graph_not_estimable() -> None:
    nodes, edges = diagonal_swap()
    one_edge, one_audit = graphs.rewire_g5(
        edges.iloc[:1],
        nodes,
        seed_name="protocol/city/0/test/G5/1",
        accepted_swap_multiplier=10,
        attempt_multiplier=200,
        overlap_max=0.10,
    )
    overlap_edges, overlap_audit = graphs.rewire_g5(
        edges,
        nodes,
        seed_name="protocol/city/0/test/G5/1",
        accepted_swap_multiplier=1,
        attempt_multiplier=20,
        overlap_max=0.10,
    )

    assert one_edge is None
    assert one_audit["status"] == "not-estimable"
    assert one_audit["failure_reason"] == "fewer-than-two-edges"
    assert overlap_edges is None
    assert overlap_audit["status"] == "not-estimable"
    assert overlap_audit["failure_reason"] == "overlap-above-limit"


def test_failed_g5_fold_discards_successful_partial_partition() -> None:
    nodes, test_edges = diagonal_swap()
    partition_edges = {"test": test_edges, "train": test_edges.iloc[:1]}

    result, audits, status = graphs.generate_g5_fold(
        partition_edges,
        nodes,
        seed_prefix="protocol/city/0",
        realization_ids=(1, 2, 3),
        accepted_swap_multiplier=1,
        attempt_multiplier=20,
        overlap_max=1.0,
    )

    assert result.empty
    assert status == "not-estimable"
    assert any(row["status"] == "success" for row in audits)
    assert any(row["failure_reason"] == "fewer-than-two-edges" for row in audits)
    assert len(audits) == 6
    assert any(row["status"] == "not-run" for row in audits)
