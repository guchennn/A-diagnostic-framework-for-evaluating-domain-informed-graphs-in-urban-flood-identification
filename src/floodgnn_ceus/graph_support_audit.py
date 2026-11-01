"""Describe frozen G2 support without fitting models or inspecting predictions."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from floodgnn_ceus.evaluation_design import label_sets_from_ledger
from floodgnn_ceus.graphs import cut_edges
from floodgnn_ceus.io import canonical_json_bytes, sha256_file
from floodgnn_ceus.reporting import markdown_table

EDGE_KEYS = ["source_grid_id", "target_grid_id"]


def upstream_sizes(nodes: list[str], edges: pd.DataFrame) -> pd.DataFrame:
    """Count unique 1-hop and <=2-hop incoming nodes, excluding the target itself."""
    if len(nodes) != len(set(nodes)):
        raise ValueError("nodes must be unique")
    if edges[EDGE_KEYS].isna().any().any() or edges[EDGE_KEYS].duplicated().any():
        raise ValueError("edges must be non-null and unique")
    node_set = set(nodes)
    reverse: dict[str, set[str]] = {}
    for source, target in edges[EDGE_KEYS].itertuples(index=False, name=None):
        if source not in node_set or target not in node_set:
            raise ValueError("edge endpoint outside node set")
        if source == target:
            raise ValueError("G2 must not contain self loops")
        reverse.setdefault(target, set()).add(source)
    rows = []
    for node in nodes:
        first = reverse.get(node, set())
        second = set(first)
        for source in first:
            second.update(reverse.get(source, set()))
        second.discard(node)
        rows.append((node, len(first), len(second)))
    return pd.DataFrame(rows, columns=["grid_id", "upstream_1hop", "upstream_2hop"])


def summarize_support(frame: pd.DataFrame, **keys: object) -> list[dict[str, object]]:
    """Summarize all cells and disjoint STRICT-LOC positive/background groups."""
    rows = []
    for group in ("all", "positive", "background"):
        part = frame if group == "all" else frame.loc[frame["sample_group"].eq(group)]
        row = {**keys, "sample_group": group, "node_count": len(part)}
        for hop in (1, 2):
            values = part[f"upstream_{hop}hop"]
            prefix = f"upstream_{hop}hop"
            row[f"{prefix}_zero_count"] = int(values.eq(0).sum())
            row[f"{prefix}_zero_fraction"] = float(values.eq(0).mean())
            row[f"{prefix}_mean"] = float(values.mean())
            for quantile, suffix in ((0.5, "median"), (0.9, "p90"), (0.95, "p95")):
                row[f"{prefix}_{suffix}"] = float(values.quantile(quantile))
            row[f"{prefix}_max"] = float(values.max())
        rows.append(row)
    return rows


def summarize_loss(before: pd.DataFrame, after: pd.DataFrame, **keys: object) -> list[dict]:
    """Report any support reduction and complete loss with explicit denominators."""
    paired = before.merge(after, on="grid_id", suffixes=("_before", "_after"), validate="1:1")
    if len(paired) != len(before) or len(paired) != len(after):
        raise ValueError("before/after must contain identical nodes")
    rows = []
    for group in ("all", "positive", "background"):
        part = paired if group == "all" else paired.loc[paired["sample_group"].eq(group)]
        for hop in (1, 2):
            pre = part[f"upstream_{hop}hop_before"]
            post = part[f"upstream_{hop}hop_after"]
            if (post > pre).any():
                raise ValueError("pruning cannot increase support")
            supported = int(pre.gt(0).sum())
            lost_all = int((pre.gt(0) & post.eq(0)).sum())
            rows.append(
                {
                    **keys,
                    "sample_group": group,
                    "hop": hop,
                    "node_count": len(part),
                    "supported_before_count": supported,
                    "any_support_loss_count": int(post.lt(pre).sum()),
                    "any_support_loss_fraction_all": float(post.lt(pre).mean()),
                    "all_support_lost_count": lost_all,
                    "all_support_lost_fraction_all": lost_all / len(part) if len(part) else np.nan,
                    "all_support_lost_fraction_supported": lost_all / supported
                    if supported
                    else np.nan,
                }
            )
    return rows


def audit_city(root: Path, city: str) -> tuple[list, list, list, list[Path]]:
    """Audit outer partitions and check pruning against the actual frozen graph files."""
    city_root = root / "data/interim" / city
    paths = [
        city_root / "final_scorable_grid.parquet",
        city_root / "final_label_ledger.parquet",
        city_root / "graphs/base_edges.parquet",
        city_root / "graphs/outer_edges.parquet",
        city_root / "evaluation_design/outer_partitions.parquet",
    ]
    nodes = pd.read_parquet(paths[0], columns=["grid_id"])["grid_id"].tolist()
    positives = label_sets_from_ledger(pd.read_parquet(paths[1]))["STRICT-LOC"]
    if not positives.issubset(nodes):
        raise ValueError("positive outside final grid")
    edges = pd.read_parquet(paths[2], filters=[("graph_id", "=", "G2")])
    frozen = pd.read_parquet(paths[3], filters=[("graph_id", "=", "G2")])
    partitions = pd.read_parquet(paths[4])
    if edges["source_grid_id"].duplicated().any():
        raise ValueError("G2 outdegree exceeds one")
    test_counts = partitions.loc[partitions["partition"].eq("test"), "grid_id"].value_counts()
    if set(test_counts.index) != set(nodes) or not test_counts.eq(1).all():
        raise ValueError("every final-grid node must occur in exactly one outer test fold")
    whole = upstream_sizes(nodes, edges)
    whole["sample_group"] = np.where(whole["grid_id"].isin(positives), "positive", "background")
    summaries = summarize_support(
        whole, city=city, graph_scope="whole_domain", outer_fold="ALL", partition="all"
    )
    retentions, losses, pooled_test = [], [], []
    for fold in sorted(partitions["outer_fold"].unique()):
        partition = partitions.loc[partitions["outer_fold"].eq(fold)]
        if set(partition["grid_id"]) != set(nodes):
            raise ValueError("partition does not cover the full grid")
        cut = cut_edges(edges, partition)
        actual = frozen.loc[frozen["outer_fold"].eq(fold)]
        comparison = [*EDGE_KEYS, "partition"]
        pd.testing.assert_frame_equal(
            cut[comparison].sort_values(comparison).reset_index(drop=True),
            actual[comparison].sort_values(comparison).reset_index(drop=True),
        )
        retentions.append(
            {
                "city": city,
                "outer_fold": int(fold),
                "partition": "train+test",
                "edges_before": len(edges),
                "edges_after": len(cut),
                "edge_retention_fraction": len(cut) / len(edges),
                "before_denominator": "all whole-domain G2 edges",
            }
        )
        for name in ("train", "test", "buffer"):
            ids = partition.loc[partition["partition"].eq(name), "grid_id"].tolist()
            before = whole.loc[whole["grid_id"].isin(ids)]
            summaries.extend(
                summarize_support(
                    before,
                    city=city,
                    graph_scope="whole_domain_for_partition_nodes",
                    outer_fold=int(fold),
                    partition=name,
                )
            )
            if name == "buffer":
                continue  # excluded from the model; not counted as model-isolated nodes
            local_edges = cut.loc[cut["partition"].eq(name)]
            local = upstream_sizes(ids, local_edges)
            local = local.merge(whole[["grid_id", "sample_group"]], on="grid_id", validate="1:1")
            summaries.extend(
                summarize_support(
                    local,
                    city=city,
                    graph_scope="partition_local",
                    outer_fold=int(fold),
                    partition=name,
                )
            )
            incoming = int(edges["target_grid_id"].isin(ids).sum())
            retentions.append(
                {
                    "city": city,
                    "outer_fold": int(fold),
                    "partition": name,
                    "edges_before": incoming,
                    "edges_after": len(local_edges),
                    "edge_retention_fraction": len(local_edges) / incoming if incoming else np.nan,
                    "before_denominator": "whole-domain G2 edges targeting this partition",
                }
            )
            losses.extend(
                summarize_loss(
                    before,
                    local.drop(columns="sample_group"),
                    city=city,
                    outer_fold=int(fold),
                    partition=name,
                )
            )
            if name == "test":
                pooled_test.append(local)
    pooled = pd.concat(pooled_test, ignore_index=True)
    summaries.extend(
        summarize_support(
            pooled,
            city=city,
            graph_scope="partition_local",
            outer_fold="POOLED",
            partition="test",
        )
    )
    losses.extend(
        summarize_loss(
            whole,
            pooled.drop(columns="sample_group"),
            city=city,
            outer_fold="POOLED",
            partition="test",
        )
    )
    return summaries, retentions, losses, paths


def run_audit(root: Path) -> None:
    """Write aggregate-only audit tables, definitions, and SHA-256 provenance."""
    protocol_path = root / "configs/protocol.yaml"
    cities = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))["cities"]
    if (
        not isinstance(cities, list)
        or not cities
        or any(not isinstance(city, str) or not city for city in cities)
        or len(cities) != len(set(cities))
    ):
        raise ValueError("protocol cities must be a non-empty list of unique city IDs")
    output = root / "results/graph_support"
    output.mkdir(parents=True, exist_ok=True)
    summaries, retentions, losses, inputs = [], [], [], []
    for city in cities:
        city_summaries, city_retentions, city_losses, city_inputs = audit_city(root, city)
        summaries.extend(city_summaries)
        retentions.extend(city_retentions)
        losses.extend(city_losses)
        inputs.extend(city_inputs)
    tables = {
        "support_summary.csv": pd.DataFrame(summaries),
        "edge_retention.csv": pd.DataFrame(retentions),
        "support_loss.csv": pd.DataFrame(losses),
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, float_format="%.10g")
    provenance_paths = [
        *inputs,
        *sorted((root / "configs").rglob("*.yaml")),
        root / "artifacts/protocol_lock.json",
        root / "uv.lock",
        root / "src/floodgnn_ceus/graph_support_audit.py",
        root / "src/floodgnn_ceus/graphs.py",
        root / "src/floodgnn_ceus/evaluation_design.py",
        root / "scripts/audit_graph_support.py",
    ]
    manifest = {
        "analysis": "post-hoc descriptive effective G2 graph-support audit; no retraining",
        "python": platform.python_version(),
        "scope": "whole-domain and frozen outer partitions; inner-CV graphs not audited",
        "label_layer": "STRICT-LOC",
        "cities": cities,
        "background": "all final-grid cells outside the STRICT-LOC ledger, not confirmed negatives",
        "weighting": "one cell one vote; pooled test contains every node exactly once",
        "two_hop": "unique incoming nodes reachable in one or two directed edges, excluding self",
        "quantiles": "numpy/pandas default linear interpolation; descriptive, no confidence interval",
        "checks": ["every node tested once", "computed pruning equals stored outer G2 edges"],
        "inputs": [
            {"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)}
            for p in provenance_paths
        ],
        "outputs": [
            {
                "path": (output / name).relative_to(root).as_posix(),
                "sha256": sha256_file(output / name),
            }
            for name in tables
        ],
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    summary = tables["support_summary.csv"]
    summary = summary.loc[summary["outer_fold"].isin(("ALL", "POOLED"))]
    columns = [
        "city",
        "graph_scope",
        "sample_group",
        "node_count",
        "upstream_1hop_zero_fraction",
        "upstream_1hop_mean",
        "upstream_2hop_mean",
        "upstream_2hop_median",
        "upstream_2hop_p95",
    ]
    loss = tables["support_loss.csv"]
    loss = loss.loc[loss["outer_fold"].eq("POOLED")]
    report = """# G2 有效图支持审计

本分析仅读取冻结的两城 G2 边、最终可评分网格、STRICT-LOC 标签账本及外层划分，
不重训、不重新准备数据、不改变 primary。统计对象是 G2 graph 提供的邻域支持，
不能直接解释为 GraphSAGE on G2 的预测贡献或因果机制。

## 定义与口径

- `upstream_1hop` 为指向节点的不同上游节点数，即无自环 G2 的入度；
  `upstream_2hop` 为一跳与两跳可达上游节点的并集，去重并排除节点自身。
- `whole_domain` 使用裁剪前全分析域 G2；`whole_domain_for_partition_nodes`
  在同一全域图上统计指定分区节点；`partition_local` 使用真实模型外层分区裁剪后的 G2。
- 每折仅保留端点同属 train 或同属 test 的边；buffer 节点不进入模型，
  因此只报告其全域支持，不将 buffer 人为记为模型中的零入度节点。
- positive 为 STRICT-LOC 账本中的网格；background 为同一分区内不在该账本的全部网格，
  可能包含未记录洪涝或只属于 ALL-LOC 的网格，不是已证实负例。
  这里没有按评估 bank 抽样，也没有按种子或 bank 重复计数。
- 每个节点恰好在一个外层 test 中出现，POOLED test 按节点等权。
  train 只逐折报告；同一节点在不同折重复训练，不能把各折训练行当独立样本。
- 总体边保留率为（train 内边 + test 内边）/全域边；分区边保留率为
  分区内保留边/裁剪前指向该分区的全域边，两者分母不同。
- `any_support_loss` 表示裁剪后上游数减少；`all_support_lost` 表示裁剪前非零、
  裁剪后归零，同时给出占全部节点和占原本有支持节点的比例。
- 仅审计外层最终拟合/评估图，不概括 inner-CV 的进一步裁剪。
  分位数为线性插值描述统计；不进行显著性检验，也不修改模型选择。

## 全域与实际测试图

所有 fraction 列均为 0–1 比例。正例与背景行的分母分别为该组的 node_count。

"""
    report += markdown_table(summary[columns]) + "\n\n## 测试节点支持损失\n\n"
    report += markdown_table(loss) + "\n\n## 解释边界与复现\n\n"
    report += """全域的无上游比例与裁剪新增的无上游比例应分开解释：前者是图本身的局部支持结构，
后者是空间隔离的代价。无上游节点仍保留 GraphSAGE 自身特征通路；零入度不等于没有特征。
两层 GraphSAGE 的 D8 两跳最大几何路径约为 2 × 200 × √2 = 566 m，
不是完整汇水区。此审计不衡量上游特征的信息量，不证明稀疏支持导致性能差异，
也不否定 G2 关系本身的信息；Fixed-filter 同样使用 G2，利用方式与 GraphSAGE 不同。

完整逐折统计见 [support_summary.csv](../results/graph_support/support_summary.csv)、
[edge_retention.csv](../results/graph_support/edge_retention.csv) 和
[support_loss.csv](../results/graph_support/support_loss.csv)。
数据、配置、代码及输出 SHA-256 见 [manifest.json](../results/graph_support/manifest.json)。
发布文件仅含汇总数，不含节点 ID、坐标或逐节点标签。

复现：`uv run python scripts/audit_graph_support.py --root .`。
"""
    (root / "docs/graph_support_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": {k: len(v) for k, v in tables.items()}}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_audit(args.root.resolve())


if __name__ == "__main__":
    main()
