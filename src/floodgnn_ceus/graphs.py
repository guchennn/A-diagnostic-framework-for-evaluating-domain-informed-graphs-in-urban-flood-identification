"""Build performance-blind graph counterfactuals from the frozen D8 proxy."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file

CITY_IDS = ("zhengzhou", "beijing")
BASE_GRAPH_IDS = ("G0", "G1", "G2", "G4")
EDGE_COLUMNS = (
    "graph_id",
    "slot_id",
    "source_grid_id",
    "target_grid_id",
    "distance_class",
)


def _ordered_hash(values: Iterable[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(values)))


def _seed(seed_name: str) -> tuple[int, str]:
    digest = hashlib.sha256(seed_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big"), digest.hex()


def _distance_class(delta_row: int, delta_column: int) -> str | None:
    delta = (abs(delta_row), abs(delta_column))
    if delta in {(1, 0), (0, 1)}:
        return "rook"
    if delta == (1, 1):
        return "diagonal"
    return None


def _node_lookup(grid: pd.DataFrame) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray]:
    required = {"grid_id", "row", "column"}
    if not required.issubset(grid.columns):
        raise ValueError("grid must contain grid_id, row, and column")
    nodes = grid[["grid_id", "row", "column"]].sort_values("grid_id", kind="stable")
    if nodes.isna().any().any() or nodes["grid_id"].duplicated().any():
        raise ValueError("grid_id values must be non-null and unique")
    if nodes[["row", "column"]].duplicated().any():
        raise ValueError("grid cells must be unique")
    grid_ids = nodes["grid_id"].astype(str).tolist()
    return (
        grid_ids,
        {grid_id: index for index, grid_id in enumerate(grid_ids)},
        nodes["row"].to_numpy(dtype=np.int64),
        nodes["column"].to_numpy(dtype=np.int64),
    )


def _classify_edges(edges: pd.DataFrame, grid: pd.DataFrame) -> list[str]:
    _, positions, rows, columns = _node_lookup(grid)
    classes = []
    for source, target in edges[["source_grid_id", "target_grid_id"]].itertuples(
        index=False, name=None
    ):
        if source not in positions or target not in positions:
            raise ValueError("edge endpoint is outside the final grid")
        source_index = positions[source]
        target_index = positions[target]
        edge_class = _distance_class(
            int(rows[target_index] - rows[source_index]),
            int(columns[target_index] - columns[source_index]),
        )
        if edge_class is None:
            raise ValueError("edge is outside the frozen rook/diagonal distance classes")
        classes.append(edge_class)
    return classes


def _number_slots(edges: pd.DataFrame) -> pd.DataFrame:
    ordered = edges.sort_values(
        ["graph_id", "source_grid_id", "target_grid_id"], kind="stable", ignore_index=True
    )
    ordered["slot_id"] = ordered.groupby("graph_id", sort=False).cumcount().astype("int64")
    return ordered[list(EDGE_COLUMNS)]


def build_base_edges(grid: pd.DataFrame, d8_edges: pd.DataFrame) -> pd.DataFrame:
    """Build canonical G1, G2, and G4 edges; G0 is represented by an empty index row."""
    grid_ids, _, _, _ = _node_lookup(grid)
    coordinates = {
        (int(row), int(column)): str(grid_id)
        for grid_id, row, column in grid[["grid_id", "row", "column"]].itertuples(
            index=False, name=None
        )
    }
    g1_pairs = []
    for grid_id, row, column in grid[["grid_id", "row", "column"]].itertuples(
        index=False, name=None
    ):
        for row_offset, column_offset in ((0, 1), (1, -1), (1, 0), (1, 1)):
            target = coordinates.get((int(row) + row_offset, int(column) + column_offset))
            if target is not None:
                g1_pairs.extend(((str(grid_id), target), (target, str(grid_id))))
    g1 = pd.DataFrame(g1_pairs, columns=["source_grid_id", "target_grid_id"])
    g1.insert(0, "graph_id", "G1")
    g1["distance_class"] = _classify_edges(g1, grid)

    required = {"source_grid_id", "target_grid_id"}
    if not required.issubset(d8_edges.columns):
        raise ValueError("D8 edges miss source or target grid_id")
    g2 = d8_edges[["source_grid_id", "target_grid_id"]].copy()
    if g2.isna().any().any() or g2.duplicated().any():
        raise ValueError("G2 edges must be non-null and unique")
    if g2["source_grid_id"].duplicated().any():
        raise ValueError("G2 outdegree exceeds one")
    if set(g2["source_grid_id"]).difference(grid_ids) or set(g2["target_grid_id"]).difference(
        grid_ids
    ):
        raise ValueError("G2 endpoint is outside the final grid")
    if g2["source_grid_id"].eq(g2["target_grid_id"]).any():
        raise ValueError("G2 contains a self-loop")
    g2.insert(0, "graph_id", "G2")
    g2["distance_class"] = _classify_edges(g2, grid)

    g4 = g2.rename(
        columns={"source_grid_id": "target_grid_id", "target_grid_id": "source_grid_id"}
    )[["graph_id", "source_grid_id", "target_grid_id", "distance_class"]]
    g4["graph_id"] = "G4"
    return _number_slots(pd.concat([g1, g2, g4], ignore_index=True))


def cut_edges(edges: pd.DataFrame, partitions: pd.DataFrame) -> pd.DataFrame:
    """Keep only edges whose endpoints share a train or test partition."""
    required_edges = {"source_grid_id", "target_grid_id"}
    if not required_edges.issubset(edges.columns):
        raise ValueError("edges miss source or target grid_id")
    if not {"grid_id", "partition"}.issubset(partitions.columns):
        raise ValueError("partitions must contain grid_id and partition")
    if partitions["grid_id"].isna().any() or partitions["grid_id"].duplicated().any():
        raise ValueError("partition grid_id values must be non-null and unique")
    if not set(partitions["partition"]).issubset({"train", "buffer", "test"}):
        raise ValueError("unknown graph partition")
    lookup = partitions.set_index("grid_id")["partition"]
    source_partition = edges["source_grid_id"].map(lookup)
    target_partition = edges["target_grid_id"].map(lookup)
    if source_partition.isna().any() or target_partition.isna().any():
        raise ValueError("edge endpoint has no partition")
    keep = source_partition.eq(target_partition) & source_partition.isin(("train", "test"))
    result = edges.loc[keep].copy()
    result["partition"] = source_partition.loc[keep].to_numpy()
    sort_columns = [
        column
        for column in ("graph_id", "partition", "source_grid_id", "target_grid_id")
        if column in result
    ]
    return result.sort_values(sort_columns, kind="stable", ignore_index=True)


def _slot_hash(source: np.ndarray, target: np.ndarray, grid_ids: list[str]) -> str:
    return _ordered_hash(
        f"{grid_ids[int(source_index)]}|{grid_ids[int(target_index)]}"
        for source_index, target_index in zip(source, target, strict=True)
    )


def rewire_g5(
    edges: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    seed_name: str,
    accepted_swap_multiplier: int,
    attempt_multiplier: int,
    overlap_max: float,
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    """Apply the frozen slot-preserving directed double-edge swap."""
    if accepted_swap_multiplier < 0 or attempt_multiplier < 0 or not 0 <= overlap_max <= 1:
        raise ValueError("invalid G5 limits")
    ordered = edges.sort_values(
        ["source_grid_id", "target_grid_id"], kind="stable", ignore_index=True
    )
    if ordered[["source_grid_id", "target_grid_id"]].duplicated().any():
        raise ValueError("G5 input edges must be unique")
    grid_ids, positions, rows, columns = _node_lookup(grid)
    edge_classes = _classify_edges(ordered, grid)
    if "distance_class" in ordered and ordered["distance_class"].tolist() != edge_classes:
        raise ValueError("G5 input distance classes do not match the grid")
    source = np.asarray([positions[value] for value in ordered["source_grid_id"]], dtype=np.int64)
    target = np.asarray([positions[value] for value in ordered["target_grid_id"]], dtype=np.int64)
    initial_source = source.copy()
    initial_target = target.copy()
    edge_set = set(zip(source.tolist(), target.tolist(), strict=True))
    original = set(edge_set)
    seed_value, seed_sha256 = _seed(seed_name)
    initial_slot_sha256 = _slot_hash(source, target, grid_ids)
    edge_count = len(source)

    def audit(
        status: str,
        reason: str | None,
        attempts: int,
        accepted: int,
        overlap: float | None,
    ) -> dict[str, object]:
        current_edges = set(zip(source.tolist(), target.tolist(), strict=True))
        current_classes = [
            _distance_class(
                int(rows[target_index] - rows[source_index]),
                int(columns[target_index] - columns[source_index]),
            )
            for source_index, target_index in zip(source, target, strict=True)
        ]
        return {
            "status": status,
            "failure_reason": reason,
            "seed_name": seed_name,
            "seed_uint128": str(seed_value),
            "seed_sha256": seed_sha256,
            "edge_count": edge_count,
            "target_accepted_swaps": accepted_swap_multiplier * edge_count,
            "attempt_cap": attempt_multiplier * edge_count,
            "attempts": attempts,
            "accepted_swaps": accepted,
            "acceptance_rate": accepted / attempts if attempts else 0.0,
            "overlap": overlap,
            "degree_preserved": Counter(source) == Counter(initial_source)
            and Counter(target) == Counter(initial_target),
            "distance_classes_preserved": current_classes == slot_classes.tolist()
            if edge_count >= 2
            else True,
            "self_loop_count": int(np.equal(source, target).sum()),
            "duplicate_edge_count": edge_count - len(current_edges),
            "initial_slot_sha256": initial_slot_sha256,
            "final_slot_sha256": _slot_hash(source, target, grid_ids),
            "final_edge_sha256": _ordered_hash(
                f"{grid_ids[source_index]}|{grid_ids[target_index]}"
                for source_index, target_index in sorted(current_edges)
            ),
        }

    if edge_count < 2:
        return None, audit("not-estimable", "fewer-than-two-edges", 0, 0, None)

    slot_classes = np.asarray(edge_classes, dtype=object)
    target_accepted = accepted_swap_multiplier * edge_count
    attempt_cap = attempt_multiplier * edge_count
    rng = np.random.Generator(np.random.PCG64DXSM(seed_value))
    attempts = 0
    accepted = 0
    while accepted < target_accepted and attempts < attempt_cap:
        first_slot, second_slot = np.sort(rng.choice(edge_count, 2, replace=False))
        first_slot = int(first_slot)
        second_slot = int(second_slot)
        a, b = int(source[first_slot]), int(target[first_slot])
        c, d = int(source[second_slot]), int(target[second_slot])
        old_first = (a, b)
        old_second = (c, d)
        proposed_first = (a, d)
        proposed_second = (c, b)
        attempts += 1
        if a == d or c == b or proposed_first == proposed_second:
            continue
        if (proposed_first in edge_set and proposed_first not in (old_first, old_second)) or (
            proposed_second in edge_set and proposed_second not in (old_first, old_second)
        ):
            continue
        first_class = _distance_class(int(rows[d] - rows[a]), int(columns[d] - columns[a]))
        second_class = _distance_class(int(rows[b] - rows[c]), int(columns[b] - columns[c]))
        if first_class != slot_classes[first_slot] or second_class != slot_classes[second_slot]:
            continue
        edge_set.remove(old_first)
        edge_set.remove(old_second)
        edge_set.add(proposed_first)
        edge_set.add(proposed_second)
        target[first_slot] = d
        target[second_slot] = b
        accepted += 1

    overlap = len(edge_set.intersection(original)) / edge_count
    if accepted < target_accepted:
        return None, audit("not-estimable", "attempt-cap-reached", attempts, accepted, overlap)
    if overlap > overlap_max:
        return None, audit("not-estimable", "overlap-above-limit", attempts, accepted, overlap)
    if Counter(source) != Counter(initial_source):
        raise RuntimeError("G5 changed node outdegree")
    if Counter(target) != Counter(initial_target):
        raise RuntimeError("G5 changed node indegree")
    result = pd.DataFrame(
        {
            "slot_id": np.arange(edge_count, dtype=np.int64),
            "source_grid_id": [grid_ids[index] for index in source],
            "target_grid_id": [grid_ids[index] for index in target],
            "distance_class": slot_classes,
        }
    )
    return result, audit("success", None, attempts, accepted, overlap)


def generate_g5_fold(
    partition_edges: Mapping[str, pd.DataFrame],
    grid: pd.DataFrame,
    *,
    seed_prefix: str,
    realization_ids: Iterable[int],
    accepted_swap_multiplier: int,
    attempt_multiplier: int,
    overlap_max: float,
) -> tuple[pd.DataFrame, list[dict[str, object]], str]:
    """Generate one fold atomically, discarding every G5 edge after a required failure."""
    pairs = [
        (realization_id, partition)
        for realization_id in realization_ids
        for partition in ("test", "train")
    ]
    staged = []
    audits = []
    failed = False
    for realization_id, partition in pairs:
        seed_name = f"{seed_prefix}/{partition}/G5/{realization_id}"
        if failed:
            edge_count = len(partition_edges[partition])
            seed_value, seed_sha256 = _seed(seed_name)
            audits.append(
                {
                    "realization_id": str(realization_id),
                    "partition": partition,
                    "status": "not-run",
                    "failure_reason": "fold-stopped-after-required-failure",
                    "seed_name": seed_name,
                    "seed_uint128": str(seed_value),
                    "seed_sha256": seed_sha256,
                    "edge_count": edge_count,
                    "target_accepted_swaps": accepted_swap_multiplier * edge_count,
                    "attempt_cap": attempt_multiplier * edge_count,
                    "attempts": 0,
                    "accepted_swaps": 0,
                    "acceptance_rate": 0.0,
                    "overlap": None,
                    "degree_preserved": None,
                    "distance_classes_preserved": None,
                    "self_loop_count": None,
                    "duplicate_edge_count": None,
                    "initial_slot_sha256": None,
                    "final_slot_sha256": None,
                    "final_edge_sha256": None,
                }
            )
            continue
        result, audit = rewire_g5(
            partition_edges[partition],
            grid,
            seed_name=seed_name,
            accepted_swap_multiplier=accepted_swap_multiplier,
            attempt_multiplier=attempt_multiplier,
            overlap_max=overlap_max,
        )
        audit.update({"realization_id": str(realization_id), "partition": partition})
        audits.append(audit)
        if result is None:
            failed = True
            continue
        result.insert(0, "partition", partition)
        result.insert(0, "realization_id", str(realization_id))
        result.insert(0, "graph_id", "G5")
        staged.append(result)
    columns = ["graph_id", "realization_id", "partition", *EDGE_COLUMNS[1:]]
    if failed:
        return pd.DataFrame(columns=columns), audits, "not-estimable"
    return pd.concat(staged, ignore_index=True)[columns], audits, "success"


def _edge_hash(edges: pd.DataFrame) -> str:
    ordered = edges.sort_values(
        ["source_grid_id", "target_grid_id"], kind="stable", ignore_index=True
    )
    return _ordered_hash(
        f"{source}|{target}"
        for source, target in ordered[["source_grid_id", "target_grid_id"]].itertuples(
            index=False, name=None
        )
    )


def _partition_hash(partitions: pd.DataFrame, partition: str) -> str:
    return _ordered_hash(sorted(partitions.loc[partitions["partition"].eq(partition), "grid_id"]))


def _index_row(
    *,
    scope: str,
    graph_id: str,
    node_count: int,
    edges: pd.DataFrame,
    outer_fold: int = -1,
    partition: str = "NA",
    realization_id: str = "NA",
    status: str = "success",
    partition_sha256: str | None = None,
) -> dict[str, object]:
    return {
        "scope": scope,
        "outer_fold": outer_fold,
        "partition": partition,
        "graph_id": graph_id,
        "realization_id": realization_id,
        "status": status,
        "node_count": node_count,
        "edge_count": len(edges),
        "partition_sha256": partition_sha256,
        "edge_sha256": _edge_hash(edges),
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)
    return path


def _file_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _read_config(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("graph_config.yaml must contain a mapping")
    return value


def _city_graphs(
    root: Path, city_id: str, config: Mapping[str, object]
) -> tuple[dict[str, object], list[Path]]:
    city_root = root / "data" / "interim" / city_id
    grid_path = city_root / "final_scorable_grid.parquet"
    d8_path = city_root / "d8_edges.parquet"
    partition_path = city_root / "evaluation_design" / "outer_partitions.parquet"
    grid = pd.read_parquet(grid_path, columns=["grid_id", "row", "column"])
    d8 = pd.read_parquet(d8_path, columns=["source_grid_id", "target_grid_id"])
    outer = pd.read_parquet(partition_path)
    expected_nodes = int(config["expected_nodes"][city_id])
    if len(grid) != expected_nodes:
        raise ValueError(f"final scorable node baseline changed for {city_id}")
    base = build_base_edges(grid, d8)
    expected_edges = config["expected_base_edges"][city_id]
    for graph_id in ("G1", "G2"):
        count = int(base["graph_id"].eq(graph_id).sum())
        if count != int(expected_edges[graph_id]):
            raise ValueError(f"{graph_id} edge baseline changed for {city_id}")

    empty = base.iloc[:0]
    index_rows = [
        _index_row(
            scope="base",
            graph_id=graph_id,
            node_count=len(grid),
            edges=empty if graph_id == "G0" else base.loc[base["graph_id"].eq(graph_id)],
        )
        for graph_id in BASE_GRAPH_IDS
    ]
    outer_frames = []
    audit_rows = []
    fold_statuses = []
    outer_cross_partition_edges = 0
    outer_buffer_incident_edges = 0
    g5 = config["g5"]
    for outer_fold in range(5):
        partitions = outer.loc[outer["outer_fold"].eq(outer_fold), ["grid_id", "partition"]]
        if len(partitions) != len(grid):
            raise ValueError(f"outer partition coverage changed for {city_id} fold {outer_fold}")
        cut = cut_edges(base, partitions)
        partition_lookup = partitions.set_index("grid_id")["partition"]
        source_partition = cut["source_grid_id"].map(partition_lookup)
        target_partition = cut["target_grid_id"].map(partition_lookup)
        outer_cross_partition_edges += int(source_partition.ne(target_partition).sum())
        outer_buffer_incident_edges += int(
            (source_partition.eq("buffer") | target_partition.eq("buffer")).sum()
        )
        cut.insert(1, "realization_id", "NA")
        cut.insert(2, "outer_fold", outer_fold)
        outer_frames.append(cut)
        for partition in ("train", "test"):
            partition_nodes = int(partitions["partition"].eq(partition).sum())
            partition_sha256 = _partition_hash(partitions, partition)
            for graph_id in BASE_GRAPH_IDS:
                graph_edges = cut.loc[cut["partition"].eq(partition) & cut["graph_id"].eq(graph_id)]
                index_rows.append(
                    _index_row(
                        scope="outer",
                        outer_fold=outer_fold,
                        partition=partition,
                        graph_id=graph_id,
                        node_count=partition_nodes,
                        edges=graph_edges,
                        partition_sha256=partition_sha256,
                    )
                )
        g2_by_partition = {
            partition: cut.loc[
                cut["partition"].eq(partition) & cut["graph_id"].eq("G2"),
                ["source_grid_id", "target_grid_id", "distance_class"],
            ]
            for partition in ("test", "train")
        }
        g5_edges, audits, fold_status = generate_g5_fold(
            g2_by_partition,
            grid,
            seed_prefix=f"{config['seed_protocol_id']}/{city_id}/{outer_fold}",
            realization_ids=tuple(int(value) for value in g5["realization_ids"]),
            accepted_swap_multiplier=int(g5["accepted_swap_multiplier"]),
            attempt_multiplier=int(g5["attempt_multiplier"]),
            overlap_max=float(g5["overlap_max"]),
        )
        for audit in audits:
            audit.update(
                {
                    "city_id": city_id,
                    "outer_fold": outer_fold,
                    "partition_sha256": _partition_hash(partitions, str(audit["partition"])),
                }
            )
        audit_rows.extend(audits)
        fold_statuses.append({"outer_fold": outer_fold, "G5": fold_status})
        if fold_status == "success":
            g5_edges.insert(2, "outer_fold", outer_fold)
            outer_frames.append(g5_edges)
        for realization_id in ("1", "2", "3"):
            for partition in ("train", "test"):
                graph_edges = g5_edges.loc[
                    g5_edges["realization_id"].eq(realization_id)
                    & g5_edges["partition"].eq(partition)
                ]
                index_rows.append(
                    _index_row(
                        scope="outer",
                        outer_fold=outer_fold,
                        partition=partition,
                        graph_id="G5",
                        realization_id=realization_id,
                        node_count=int(partitions["partition"].eq(partition).sum()),
                        edges=graph_edges,
                        status=fold_status,
                        partition_sha256=_partition_hash(partitions, partition),
                    )
                )

    graph_root = city_root / "graphs"
    base_path = _write_frame(base, graph_root / "base_edges.parquet")
    outer_edges = pd.concat(outer_frames, ignore_index=True).sort_values(
        [
            "outer_fold",
            "graph_id",
            "realization_id",
            "partition",
            "source_grid_id",
            "target_grid_id",
        ],
        kind="stable",
        ignore_index=True,
    )
    outer_path = _write_frame(outer_edges, graph_root / "outer_edges.parquet")
    index = pd.DataFrame(index_rows).sort_values(
        ["scope", "outer_fold", "graph_id", "realization_id", "partition"],
        kind="stable",
        ignore_index=True,
    )
    index_path = _write_frame(index, graph_root / "graph_index.parquet")
    audit = pd.DataFrame(audit_rows).sort_values(
        ["outer_fold", "realization_id", "partition"], kind="stable", ignore_index=True
    )
    audit_path = _write_frame(audit, graph_root / "g5_audit.parquet")
    paths = [base_path, outer_path, index_path, audit_path]
    report = {
        "city_id": city_id,
        "nodes": len(grid),
        "base_edges": {
            graph_id: int(base["graph_id"].eq(graph_id).sum()) for graph_id in BASE_GRAPH_IDS
        },
        "G5_status": (
            "success" if all(item["G5"] == "success" for item in fold_statuses) else "not-estimable"
        ),
        "qa": {
            "G2_max_outdegree": int(
                base.loc[base["graph_id"].eq("G2")].groupby("source_grid_id").size().max()
            ),
            "G4_exact_reverse": set(
                base.loc[
                    base["graph_id"].eq("G4"), ["source_grid_id", "target_grid_id"]
                ].itertuples(index=False, name=None)
            )
            == {
                (target, source)
                for source, target in base.loc[
                    base["graph_id"].eq("G2"), ["source_grid_id", "target_grid_id"]
                ].itertuples(index=False, name=None)
            },
            "outer_cross_partition_edges": outer_cross_partition_edges,
            "outer_buffer_incident_edges": outer_buffer_incident_edges,
        },
        "outer_folds": fold_statuses,
        "inputs": [_file_record(root, path) for path in (grid_path, d8_path, partition_path)],
        "outputs": [_file_record(root, path) for path in paths],
    }
    return report, paths


def prepare_graphs(root: Path) -> dict[str, object]:
    """Write frozen base, outer-cut, and G5 graph counterfactual artifacts."""
    config_path = root / "configs" / "graph_config.yaml"
    config = _read_config(config_path)
    if config.get("seed_protocol_id") != "FloodGNN-CEUS-ZZ-BJ-v1.0":
        raise ValueError("graph seed protocol ID changed")
    cities = []
    for city_id in CITY_IDS:
        report, _ = _city_graphs(root, city_id, config)
        cities.append(report)
    payload = {
        "schema_version": 1,
        "phase": "E0",
        "performance_blind": True,
        "labels_read": False,
        "features_read": False,
        "predictions_written": False,
        "metrics_written": False,
        "figures_written": False,
        "config": _file_record(root, config_path),
        "config_canonical_sha256": sha256_bytes(canonical_json_bytes(config)),
        "implementations": [
            _file_record(root, root / path) for path in config["implementation_files"]
        ],
        "cities": cities,
    }
    report_path = root / "artifacts" / "reports" / "graph_counterfactuals.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial = report_path.with_name(f"{report_path.name}.part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, report_path)
    return {"cities": cities, "report": str(report_path), "performance_blind": True}
