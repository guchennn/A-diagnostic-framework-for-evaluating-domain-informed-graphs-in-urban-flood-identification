"""Label-independent degree and distance matched local graph perturbations.

Within fixed spatial tiles, binary assignment preserves every source outdegree,
every receiver indegree, and the number of rook/diagonal edges. Cross-tile edges
stay fixed. A seeded secondary objective varies feasible perturbations without
using attributes, labels, or predictive performance.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

EDGE_COLUMNS = ["source_grid_id", "target_grid_id"]


def _pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(frame[EDGE_COLUMNS].itertuples(index=False, name=None))


def _edge_hash(edges: list[tuple[str, str]]) -> str:
    return hashlib.sha256(
        "\n".join(f"{source}|{target}" for source, target in sorted(edges)).encode()
    ).hexdigest()


def _positions(grid: pd.DataFrame) -> dict[str, tuple[int, int]]:
    if "grid_id" not in grid:
        grid = grid.reset_index()
    if grid["grid_id"].duplicated().any():
        raise ValueError("grid identifiers must be unique")
    return {
        str(node): (int(row), int(column))
        for node, row, column in grid[["grid_id", "row", "column"]].itertuples(
            index=False, name=None
        )
    }


def _diagonal(source: str, target: str, places: dict[str, tuple[int, int]]) -> int:
    sr, sc = places[source]
    tr, tc = places[target]
    dr, dc = abs(sr - tr), abs(sc - tc)
    if max(dr, dc) != 1:
        raise ValueError("input and control edges must join distinct eight-neighbours")
    return int(dr == dc == 1)


def _solve_tile(
    original: list[tuple[str, str]],
    places: dict[str, tuple[int, int]],
    *,
    seed_name: str,
    time_limit: float,
    relative_gap: float,
) -> tuple[list[tuple[str, str]], dict[str, object]]:
    sources = sorted(source for source, _ in original)
    targets = sorted({target for _, target in original})
    receiver = {target: i for i, target in enumerate(targets)}
    demand = Counter(target for _, target in original)
    original_set = set(original)
    target_at = {places[target]: target for target in targets}
    candidates = []
    row_indices, column_indices, diagonal_flags = [], [], []
    for source_index, source in enumerate(sources):
        sr, sc = places[source]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                target = target_at.get((sr + dr, sc + dc))
                if target is None:
                    continue
                j = len(candidates)
                candidates.append((source, target))
                row_indices.extend((source_index, len(sources) + receiver[target]))
                column_indices.extend((j, j))
                diagonal = int(abs(dr) == abs(dc) == 1)
                diagonal_flags.append(diagonal)
                if diagonal:
                    row_indices.append(len(sources) + len(targets))
                    column_indices.append(j)
    diagonal_count = sum(_diagonal(s, t, places) for s, t in original)
    shape = (len(sources) + len(targets) + 1, len(candidates))
    matrix = coo_array(
        (np.ones(len(row_indices)), (row_indices, column_indices)), shape=shape
    ).tocsc()
    required = np.r_[np.ones(len(sources)), [demand[t] for t in targets], diagonal_count]
    seed = int.from_bytes(hashlib.sha256(seed_name.encode()).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    # With a solved optimum, one retained edge outweighs every possible jitter change.
    cost = np.array(
        [(len(original) + 1) * int(edge in original_set) for edge in candidates], dtype=float
    )
    cost += rng.uniform(0, 1, len(candidates))
    result = milp(
        cost,
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, required, required),
        options={"time_limit": time_limit, "mip_rel_gap": relative_gap},
    )
    selected = None
    if result.x is not None:
        rounded = np.rint(result.x)
        if np.max(np.abs(result.x - rounded)) < 1e-6 and np.array_equal(matrix @ rounded, required):
            selected = [edge for edge, keep in zip(candidates, rounded, strict=True) if keep]
    fallback = selected is None
    if fallback:
        selected = original
    gap = getattr(result, "mip_gap", None)
    return selected, {
        "edges": len(original),
        "candidate_edges": len(candidates),
        "solver_status": int(result.status),
        "mip_gap": float(gap) if gap is not None and np.isfinite(gap) else None,
        "retained_original_fallback": fallback,
        "changed_edges": len(original_set - set(selected)),
    }


def build_control(
    g2: pd.DataFrame, grid: pd.DataFrame, *, seed_name: str, config: dict
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a validated G7 partition graph and label-independent construction audit."""
    started = time.perf_counter()
    tile_size = int(config.get("tile_size_cells", 32))
    minimum = float(config.get("min_changed_fraction", 0.08))
    time_limit = float(config.get("time_limit_seconds", 5))
    relative_gap = float(config.get("mip_rel_gap", 0.01))
    if tile_size < 2 or not 0 <= minimum <= 1 or time_limit <= 0 or relative_gap < 0:
        raise ValueError("invalid graph-construction settings")
    if config.get("preserve_global_distance_counts", True) is not True:
        raise ValueError("G7 requires exact distance counts")
    places = _positions(grid)
    frame = g2[EDGE_COLUMNS].astype(str).sort_values(EDGE_COLUMNS, kind="stable")
    original = _pairs(frame)
    if len(set(original)) != len(original):
        raise ValueError("duplicate input edges")
    if any(count != 1 for count in Counter(s for s, _ in original).values()):
        raise ValueError("D8 control requires outdegree at most one")
    for source, target in original:
        _diagonal(source, target, places)
    tiles: dict[tuple[int, int], list[tuple[str, str]]] = {}
    retained = []
    for edge in original:
        source, target = edge
        source_tile = tuple(value // tile_size for value in places[source])
        target_tile = tuple(value // tile_size for value in places[target])
        if source_tile != target_tile:
            retained.append(edge)
        else:
            tiles.setdefault(source_tile, []).append(edge)
    constructed = list(retained)
    audits = []
    for tile, edges in sorted(tiles.items()):
        replacement, audit = _solve_tile(
            edges,
            places,
            seed_name=f"{seed_name}/tile/{tile[0]}/{tile[1]}",
            time_limit=time_limit,
            relative_gap=relative_gap,
        )
        constructed.extend(replacement)
        audits.append({"tile_row": tile[0], "tile_column": tile[1], **audit})
    checks = {
        "edge_count_preserved": len(constructed) == len(original),
        "indegree_preserved": Counter(t for _, t in constructed) == Counter(t for _, t in original),
        "outdegree_preserved": Counter(s for s, _ in constructed)
        == Counter(s for s, _ in original),
        "distance_counts_preserved": sum(_diagonal(s, t, places) for s, t in constructed)
        == sum(_diagonal(s, t, places) for s, t in original),
        "no_duplicate_edges": len(set(constructed)) == len(constructed),
        "cross_tile_edges_unchanged": set(retained).issubset(constructed),
    }
    if not all(checks.values()):
        raise RuntimeError(f"control violated graph invariants: {checks}")
    changed = len(set(original) - set(constructed))
    fraction = changed / len(original) if original else 0.0
    audit = {
        "accepted": bool(original) and fraction >= minimum,
        "status": "success" if original and fraction >= minimum else "insufficient-perturbation",
        "label_blind": True,
        "seed_name": seed_name,
        "tile_size_cells": tile_size,
        "edges": len(original),
        "changed_edges": changed,
        "changed_fraction": fraction,
        "overlap_fraction": 1 - fraction,
        "diagonal_edges": sum(_diagonal(s, t, places) for s, t in original),
        "cross_tile_edges": len(retained),
        "solver_fallback_tiles": sum(bool(a["retained_original_fallback"]) for a in audits),
        "original_edge_sha256": _edge_hash(original),
        "control_edge_sha256": _edge_hash(constructed),
        "seconds": time.perf_counter() - started,
        **checks,
        "tiles": audits,
    }
    result = pd.DataFrame(sorted(constructed), columns=EDGE_COLUMNS)
    return result, audit
