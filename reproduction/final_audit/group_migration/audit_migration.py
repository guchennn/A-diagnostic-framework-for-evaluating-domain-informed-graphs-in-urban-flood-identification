"""Audit full-domain to frozen outer-test D8 group transitions; aggregate outputs only."""

from __future__ import annotations

import argparse
import json
import platform
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from floodgnn_ceus.evaluation_design import label_sets_from_ledger
from floodgnn_ceus.graph_support_audit import EDGE_KEYS
from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.supplementary_mechanisms import graph_structure

GROUPS = ("0", "1", "2")


def classify(nodes: list[str], edges: pd.DataFrame) -> np.ndarray:
    """0: no incoming edge; 1: incoming but no two-edge path; 2: a two-edge path."""
    incoming = set(edges.target_grid_id)
    depth_two = set(edges.loc[edges.source_grid_id.isin(incoming), "target_grid_id"])
    return np.array(["2" if n in depth_two else "1" if n in incoming else "0" for n in nodes])


def assert_acyclic(nodes: list[str], edges: pd.DataFrame) -> None:
    """Reject cycles rather than assuming the terrain-derived artifact is a DAG."""
    indegree = dict.fromkeys(nodes, 0)
    successors: dict[str, list[str]] = {}
    for s, t in edges[EDGE_KEYS].itertuples(index=False, name=None):
        indegree[t] += 1
        successors.setdefault(s, []).append(t)
    queue = deque(n for n in nodes if indegree[n] == 0)
    visited = 0
    while queue:
        n = queue.popleft()
        visited += 1
        for t in successors.get(n, []):
            indegree[t] -= 1
            if indegree[t] == 0:
                queue.append(t)
    if visited != len(nodes):
        raise ValueError("Full D8 graph contains a directed cycle")


def load_city(
    root: Path, city: str, verify_historical: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    """Return private in-memory cell alignment, edge summaries and provenance paths."""
    base = root / "data/interim" / city
    paths = [
        base / p
        for p in (
            "final_scorable_grid.parquet",
            "final_label_ledger.parquet",
            "graphs/base_edges.parquet",
            "graphs/outer_edges.parquet",
            "evaluation_design/outer_partitions.parquet",
        )
    ]
    nodes = pd.read_parquet(paths[0], columns=["grid_id"]).grid_id.tolist()
    positives = label_sets_from_ledger(pd.read_parquet(paths[1]))["STRICT-LOC"]
    edges = pd.read_parquet(paths[2], filters=[("graph_id", "=", "G2")])
    frozen = pd.read_parquet(paths[3], filters=[("graph_id", "=", "G2")])
    parts = pd.read_parquet(paths[4])
    assert_acyclic(nodes, edges)
    assert not edges.source_grid_id.duplicated().any()
    # Existing implementation independently checks frozen edges against actual partition pruning.
    frame = graph_structure(nodes, edges, parts, frozen)
    full = pd.DataFrame({"grid_id": nodes, "full_group": classify(nodes, edges)})
    frame = frame.merge(full, on="grid_id", validate="1:1").rename(columns={"L": "heldout_group"})
    frame["positive"] = frame.grid_id.isin(positives)
    frame["city"] = city
    assert positives.issubset(nodes)
    assert (frame.heldout_group.astype(int) <= frame.full_group.astype(int)).all()
    if verify_historical:
        legacy_path = (
            root / "artifacts/runtime/supp-mechanism-ab-20261007-r4/test_node_structure.parquet"
        )
        legacy = pd.read_parquet(legacy_path)
        legacy = legacy.loc[legacy.city.eq(city), ["grid_id", "outer_fold", "L"]]
        pd.testing.assert_frame_equal(
            frame[["grid_id", "outer_fold", "heldout_group"]]
            .rename(columns={"heldout_group": "L"})
            .sort_values("grid_id")
            .reset_index(drop=True),
            legacy.sort_values("grid_id").reset_index(drop=True),
        )
        paths.append(legacy_path)
    rows = []
    for fold, p in parts.groupby("outer_fold"):
        test_ids = set(p.loc[p.partition.eq("test"), "grid_id"])
        local = frozen.loc[frozen.outer_fold.eq(fold) & frozen.partition.eq("test")]
        expected = edges.loc[
            edges.source_grid_id.isin(test_ids) & edges.target_grid_id.isin(test_ids)
        ]
        assert set(local[EDGE_KEYS].itertuples(index=False, name=None)) == set(
            expected[EDGE_KEYS].itertuples(index=False, name=None)
        )
        actual = frame.loc[frame.outer_fold.eq(fold)].sort_values("grid_id")
        assert np.array_equal(classify(actual.grid_id.tolist(), local), actual.heldout_group)
        target_edges = int(edges.target_grid_id.isin(test_ids).sum())
        retained_all = int(frozen.outer_fold.eq(fold).sum())
        rows.append(
            {
                "city": city,
                "outer_fold": int(fold),
                "full_edges": len(edges),
                "train_test_retained_edges": retained_all,
                "train_test_retention_fraction": retained_all / len(edges),
                "edges_targeting_test_before": target_edges,
                "test_edges_retained": len(local),
                "test_target_retention_fraction": len(local) / target_edges,
            }
        )
    assert len(frame) == len(nodes) and not frame.grid_id.duplicated().any()
    return frame, pd.DataFrame(rows), paths


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrices, summaries = [], []
    for fold in ["POOLED", *sorted(frame.outer_fold.unique())]:
        selected = frame if fold == "POOLED" else frame.loc[frame.outer_fold.eq(fold)]
        for population in ("all", "positive", "background"):
            f = (
                selected
                if population == "all"
                else selected.loc[selected.positive.eq(population == "positive")]
            )
            keys = {"city": frame.city.iloc[0], "outer_fold": str(fold), "population": population}
            for before in GROUPS:
                n_before = int(f.full_group.eq(before).sum())
                for after in GROUPS:
                    count = int((f.full_group.eq(before) & f.heldout_group.eq(after)).sum())
                    matrices.append(
                        {
                            **keys,
                            "full_group": before,
                            "heldout_group": after,
                            "count": count,
                            "row_total": n_before,
                            "population_n": len(f),
                            "row_fraction": count / n_before if n_before else np.nan,
                            "population_fraction": count / len(f),
                        }
                    )
            changed = int(f.full_group.ne(f.heldout_group).sum())
            summaries.append(
                {
                    **keys,
                    "n": len(f),
                    "group_changed_n": changed,
                    "group_changed_fraction": changed / len(f),
                    "group_stable_n": len(f) - changed,
                    "new_no_upstream_n": int(
                        (f.full_group.ne("0") & f.heldout_group.eq("0")).sum()
                    ),
                    "heldout_no_upstream_n": int(f.heldout_group.eq("0").sum()),
                    "heldout_one_hop_n": int(f.heldout_group.eq("1").sum()),
                    "heldout_multi_hop_n": int(f.heldout_group.eq("2").sum()),
                    "any_one_hop_support_loss_n": int((f.indegree_test < f.indegree_full).sum()),
                    "any_two_hop_support_loss_n": int((f.K_test < f.K_full).sum()),
                }
            )
    return pd.DataFrame(matrices), pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reproduced/final_audit/group_migration")
    )
    parser.add_argument("--verify-historical-baseline", action="store_true")
    args = parser.parse_args()
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    required = (
        [root / "artifacts/runtime/supp-mechanism-ab-20261007-r4/test_node_structure.parquet"]
        if args.verify_historical_baseline
        else []
    )
    required += [
        root / "data/interim" / city / filename
        for city in ("zhengzhou", "beijing")
        for filename in (
            "final_scorable_grid.parquet",
            "final_label_ledger.parquet",
            "graphs/base_edges.parquet",
            "graphs/outer_edges.parquet",
            "evaluation_design/outer_partitions.parquet",
        )
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error(
            "Original prepared graph/grid/label inputs (and archived membership when requested) are required. "
            "See reproduction/final_audit/README.md. Missing: " + "; ".join(missing)
        )
    out.mkdir(parents=True, exist_ok=True)
    matrices, summaries, edge_tables, inputs = [], [], [], []
    expected = {"zhengzhou": (25155, 791, 233), "beijing": (61443, 296, 112)}
    for city, expected_counts in expected.items():
        f, e, paths = load_city(root, city, args.verify_historical_baseline)
        assert (
            len(f),
            int(f.positive.sum()),
            int((f.positive & f.heldout_group.eq("2")).sum()),
        ) == expected_counts
        m, s = summarize(f)
        matrices.append(m)
        summaries.append(s)
        edge_tables.append(e)
        inputs.extend(paths)
    tables = {
        "transition_matrix.csv": pd.concat(matrices, ignore_index=True),
        "migration_summary.csv": pd.concat(summaries, ignore_index=True),
        "edge_retention.csv": pd.concat(edge_tables, ignore_index=True),
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    repository = Path(__file__).resolve().parents[3]
    code_paths = [
        Path(__file__),
        repository / "src/floodgnn_ceus/graph_support_audit.py",
        repository / "src/floodgnn_ceus/supplementary_mechanisms.py",
        repository / "src/floodgnn_ceus/graphs.py",
        repository / "src/floodgnn_ceus/evaluation_design.py",
        root / "configs/graph_config.yaml",
        root / "configs/protocol.yaml",
        root / "artifacts/protocol_lock.json",
        repository / "requirements.txt",
    ]
    manifest = {
        "python": platform.python_version(),
        "scope": "frozen outer test; no inner-CV or rerouting",
        "replication": "each eligible cell appears in one outer test fold; pooled counts are cell weighted",
        "checks": [
            "acyclic full graphs",
            "outdegree at most one",
            "frozen pruning identity",
            "test partition exact coverage",
            "no upward group transitions",
            "legacy heldout group identity"
            if args.verify_historical_baseline
            else "historical comparison not requested",
            "manuscript positive counts",
        ],
        "inputs": {str(p): sha256_file(p) for p in sorted(set(inputs + code_paths))},
        "outputs": {name: sha256_file(out / name) for name in tables},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(tables["migration_summary.csv"].query("outer_fold == 'POOLED'").to_string(index=False))


if __name__ == "__main__":
    main()
