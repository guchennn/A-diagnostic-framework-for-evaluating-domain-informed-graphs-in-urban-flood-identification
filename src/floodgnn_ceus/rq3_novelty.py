"""Run the v1.2 RQ3 conditional graph-scale novelty analysis (Zhengzhou, FULL budget).

RQ3 is a frozen exploratory absolute-rank analysis: does local graph-scale novelty
co-vary with positive-cell ranks from GraphSAGE on G2? This outcome alone cannot
identify graph-specific difficulty or G2-minus-G0 benefit. It is performance-blind in the sense
that every value it consumes is frozen before inspection: the feature membership,
the novelty thresholds, the neighbourhood, and the outcome all come from
``configs/rq3_novelty_freeze_v1_2.yaml`` (see ``docs/rq3_freeze_decision.md``).

The analysis never re-fits a model. It reads the already-frozen FULL-budget G2
out-of-fold bank predictions from the decisive run and measures, for each positive
test cell, the percentile-rank of its score within its bank (the empirical CDF),
then stratifies those ranks by the label-blind composite graph-scale novelty of the
cell and estimates the High-minus-Low mean rank difference over the five outer
folds with a fold-level bootstrap.

Primary estimand: Delta_novelty = mean_high - mean_low.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from floodgnn_ceus.io import canonical_json_bytes, sha256_file
from floodgnn_ceus.reporting import markdown_table

CITY = "zhengzhou"
STRATA = ("Low", "Medium", "High")
PRIMARY_CONTRAST = ("High", "Low")
CONTRASTS = (("High", "Low"), ("Medium", "Low"), ("High", "Medium"))
SECONDARY_AXES = ("built_up_fraction", "relative_elevation_1km")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _versions() -> dict[str, str]:
    names = ("numpy", "pandas", "pyarrow", "scikit-learn")
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def _stable_rng(protocol_id: str, seed_name: str) -> np.random.Generator:
    digest = hashlib.sha256(seed_name.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:16], "big")
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _reverse_adjacency(g2: pd.DataFrame) -> dict[str, list[str]]:
    """Map each target to the list of sources that flow into it."""
    reverse: dict[str, list[str]] = {}
    for source, target in g2[["source_grid_id", "target_grid_id"]].itertuples(
        index=False, name=None
    ):
        reverse.setdefault(target, []).append(source)
    return reverse


def _n2_indices(
    reverse: dict[str, list[str]],
    node_index: dict[str, int],
    node_id: str,
) -> list[int]:
    """Unique incoming nodes within at most two directed G2 edges, excluding self."""
    neighbours: set[str] = set(reverse.get(node_id, ()))
    for hop in tuple(neighbours):
        neighbours.update(reverse.get(hop, ()))
    neighbours.discard(node_id)
    return [node_index[value] for value in neighbours]


def _composite_novelty(
    features: pd.DataFrame,
    feature_names: list[str],
    reverse: dict[str, list[str]],
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Return (resolvable ids, composite novelty, per-feature novelty columns)."""
    resolvable = sorted(reverse)  # in-degree >= 1 <-> N2 non-empty
    raw = features.set_index("grid_id")[feature_names].astype("float64")
    ordered = raw.index.tolist()
    index_of = {grid_id: position for position, grid_id in enumerate(ordered)}
    per_feature: dict[str, np.ndarray] = {}
    for column in feature_names:
        values = raw[column].to_numpy(dtype=np.float64)
        reference = raw.loc[resolvable, column].to_numpy(dtype=np.float64)
        mean = float(reference.mean())
        scale = float(reference.std(ddof=0))
        per_feature[column] = (values - mean) / scale if scale > 0 else np.zeros_like(values)
    zmatrix = np.column_stack([per_feature[column] for column in feature_names])
    novelty = np.zeros((len(resolvable), len(feature_names)), dtype=np.float64)
    for position, node_id in enumerate(resolvable):
        neighbours = _n2_indices(reverse, index_of, node_id)
        z_i = zmatrix[index_of[node_id]]
        novelty[position] = np.abs(zmatrix[neighbours] - z_i).mean(axis=0)
    composite = novelty.mean(axis=1)
    columns = {column: novelty[:, offset] for offset, column in enumerate(feature_names)}
    return resolvable, composite, columns


def _strata(composite: np.ndarray, resolvable: list[str]) -> dict[str, str]:
    """Assign city-specific tertiles by a stable (composite, grid_id) sort."""
    order = sorted(range(len(resolvable)), key=lambda key: (composite[key], resolvable[key]))
    third = len(resolvable) // 3
    assignment: dict[str, str] = {}
    for rank, position in enumerate(order):
        assignment[resolvable[position]] = STRATA[min(rank // third, len(STRATA) - 1)]
    return assignment


def _fold_effects(
    fold_strata: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int]]:
    """Pivot per-fold x stratum means into a per-fold, per-contrast effect frame."""
    folded = fold_strata.pivot(index="outer_fold", columns="stratum", values="mean_pct_rank")
    counts = fold_strata.pivot(index="outer_fold", columns="stratum", values="positive_cells")
    if set(folded.columns) != set(STRATA) or len(folded) == 0:
        raise ValueError("fold-strata table does not cover every stratum in every fold")
    effects: dict[tuple[str, str], float] = {}
    sizes: dict[tuple[str, str], int] = {}
    for treatment, control in CONTRASTS:
        delta = (folded[treatment] - folded[control]).to_numpy(dtype=np.float64)
        effects[(treatment, control)] = delta
        sizes[(treatment, control)] = int(counts[treatment].sum())
    return effects, sizes


def _bootstrap(
    effects: dict[tuple[str, str], np.ndarray],
    *,
    replicates: int,
    protocol_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = _stable_rng(protocol_id, f"{protocol_id}/{CITY}/rq3-conditional-novelty")
    summaries: list[dict[str, object]] = []
    draws_rows: list[dict[str, object]] = []
    for (treatment, control), delta in effects.items():
        if len(delta) < 2:
            raise ValueError("bootstrap needs at least two outer folds")
        point = float(delta.mean())
        sampled = rng.integers(0, len(delta), size=(replicates, len(delta)))
        draws = delta[sampled].mean(axis=1)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        p_positive = (1 + int(np.count_nonzero(draws <= 0))) / (replicates + 1)
        p_negative = (1 + int(np.count_nonzero(draws >= 0))) / (replicates + 1)
        p_two = min(1.0, 2 * min(p_positive, p_negative))
        summaries.append(
            {
                "contrast": f"{treatment}-{control}",
                "treatment": treatment,
                "control": control,
                "role": "primary" if (treatment, control) == PRIMARY_CONTRAST else "secondary",
                "point_effect": point,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "two_sided_p": float(p_two),
                "positive_outer_folds": int(np.count_nonzero(delta > 0)),
                "outer_fold_count": len(delta),
                "bootstrap_unit": "outer_fold",
                "bootstrap_replicates": replicates,
            }
        )
        draws_rows.extend(
            {
                "contrast": f"{treatment}-{control}",
                "replicate": index,
                "effect": float(value),
            }
            for index, value in enumerate(draws)
        )
    return pd.DataFrame(summaries), pd.DataFrame(draws_rows)


def run_rq3_novelty(
    root: Path,
    output_root: Path,
    *,
    replicates: int = 10_000,
) -> dict[str, object]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    freeze_path = root / "configs" / "rq3_novelty_freeze_v1_2.yaml"
    freeze = yaml.safe_load(freeze_path.read_text("utf-8"))
    protocol_id = str(freeze["protocol_id"])
    if protocol_id != "FloodGNN-CEUS-ZZ-BJ-v1.0":
        raise ValueError("freeze protocol ID changed")
    feature_names = list(freeze["feature_membership"]["graph_resolvable_features"])
    if len(feature_names) != 15:
        raise ValueError("freeze feature membership changed")
    outcome = freeze["outcome"]
    if str(outcome["method"]) != "G2" or str(outcome["budget"]) != "FULL":
        raise ValueError("freeze outcome source changed")

    # 1. Graph neighbourhood from the frozen full-city G2 edges.
    graph_root = root / "data" / "interim" / CITY / "graphs"
    base_edges = pd.read_parquet(graph_root / "base_edges.parquet")
    g2 = base_edges.loc[base_edges["graph_id"].eq("G2")].copy()
    g2["source_grid_id"] = g2["source_grid_id"].astype(str)
    g2["target_grid_id"] = g2["target_grid_id"].astype(str)
    reverse = _reverse_adjacency(g2)

    # 2. Composite novelty over graph-resolvable nodes.
    features = pd.read_parquet(
        root / "data" / "interim" / CITY / "features_18.parquet",
        columns=["grid_id", *feature_names],
    )
    resolvable, composite, per_feature = _composite_novelty(features, feature_names, reverse)
    strata = _strata(composite, resolvable)
    novelty_table = pd.DataFrame(
        {
            "grid_id": resolvable,
            "composite_novelty": composite,
            "stratum": [strata[value] for value in resolvable],
            "novelty_built_up_fraction": per_feature["built_up_fraction"],
            "novelty_relative_elevation_1km": per_feature["relative_elevation_1km"],
        }
    )
    stratum_counts = novelty_table["stratum"].value_counts()
    if len(stratum_counts) != 3 or not (stratum_counts == len(resolvable) // 3).all():
        raise ValueError("tertile assignment did not split graph-resolvable nodes evenly")

    # 3. Outcome: percentile-rank of each positive test cell within its bank.
    oof_path = root / outcome["source"]
    banks = pd.read_parquet(oof_path)
    g2_banks = banks.loc[banks["method"].eq("G2")].copy()
    g2_banks["grid_id"] = g2_banks["grid_id"].astype(str)
    g2_banks["pct_rank"] = (
        g2_banks.groupby(["outer_fold", "seed", "bank_id"], sort=False)["score"]
        .rank(method="max", pct=True)
        .astype("float64")
    )
    positive = g2_banks.loc[g2_banks["sample_role"].eq("positive")].copy()
    positive["stratum"] = positive["grid_id"].map(strata)
    excluded = positive["stratum"].isna()
    excluded_rows = int(excluded.sum())
    excluded_cells = int(positive.loc[excluded, "grid_id"].nunique())
    positive = positive.loc[~excluded].copy()
    positive["stratum"] = positive["stratum"].astype(str)

    cell_table = positive[
        ["outer_fold", "seed", "bank_id", "grid_id", "stratum", "pct_rank"]
    ].sort_values(["outer_fold", "seed", "bank_id", "stratum", "grid_id"], kind="stable")

    # 4. Per-fold absolute mean rank and the frozen High-minus-Low contrast.
    fold_strata = (
        positive.groupby(["outer_fold", "stratum"], sort=False)["pct_rank"]
        .agg(mean_pct_rank="mean", positive_cells="size")
        .reset_index()
    )
    effects, _ = _fold_effects(fold_strata)
    inference, bootstrap = _bootstrap(effects, replicates=replicates, protocol_id=protocol_id)

    # 5. Write outputs.
    strata_path = output_root / "novelty_strata.parquet"
    cell_path = output_root / "cell_rank_gain.parquet"
    fold_path = output_root / "fold_strata.csv"
    inference_path = output_root / "novelty_inference.csv"
    bootstrap_path = output_root / "novelty_bootstrap.parquet"
    novelty_table.to_parquet(strata_path, index=False)
    cell_table.to_parquet(cell_path, index=False)
    fold_strata.to_csv(fold_path, index=False)
    inference.to_csv(inference_path, index=False)
    bootstrap.to_parquet(bootstrap_path, index=False)

    manifest = {
        "status": "success",
        "protocol_id": protocol_id,
        "amendment_version": "v1.2",
        "city": CITY,
        "budget": str(outcome["budget"]),
        "method": str(outcome["method"]),
        "freeze_config": str(freeze_path),
        "freeze_config_sha256": sha256_file(freeze_path),
        "freeze_sha": str(freeze["freeze_sha"]),
        "graph_resolvable_nodes": len(resolvable),
        "stratum_sizes": {stratum: int(stratum_counts[stratum]) for stratum in STRATA},
        "positive_cell_rows_total": len(cell_table),
        "positive_cell_rows_excluded_no_upstream": excluded_rows,
        "positive_cells_excluded_no_upstream": excluded_cells,
        "secondary_axes": list(SECONDARY_AXES),
        "primary_contrast": "-".join(PRIMARY_CONTRAST),
        "bootstrap_replicates": replicates,
        "environment": _versions(),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (strata_path, cell_path, fold_path, inference_path, bootstrap_path)
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    (output_root / "report.md").write_text(
        _report_text(inference, fold_strata, excluded_rows, excluded_cells, manifest),
        encoding="utf-8",
    )
    return manifest


def _report_text(
    inference: pd.DataFrame,
    fold_strata: pd.DataFrame,
    excluded_rows: int,
    excluded_cells: int,
    manifest: dict[str, object],
) -> str:
    lines = [
        "# Zhengzhou RQ3 — conditional graph-scale novelty",
        "",
        f"protocol `{manifest['protocol_id']}`, amendment v1.2, budget FULL, GraphSAGE on G2.",
        f"Freeze SHA `{manifest['freeze_sha']}`.",
        "",
        "## Per-fold mean percentile-rank by novelty stratum (pooled over seeds and banks)",
        "",
        markdown_table(fold_strata),
        "",
        "## Novelty contrasts (outer-fold bootstrap, 10,000 replicates)",
        "",
        markdown_table(inference),
        "",
        "## Reading",
        "",
        "- The frozen outcome is the absolute percentile-rank of a positive test cell's",
        "  GraphSAGE-on-G2 OOF score within its bank (empirical CDF over positive + background",
        "  cells). Bank-relative ranks do not remove all differences in stratum composition.",
        "- The historical per-fold `positive_cells` column counts seed-by-bank measurements,",
        "  not unique positive cells; the exclusion counts below distinguish those denominators.",
        (
            f"- {manifest['positive_cell_rows_excluded_no_upstream']} positive cell measurements"
            f" ({manifest['positive_cells_excluded_no_upstream']} unique cells) have no upstream"
        ),
        "  neighbour in the whole-city G2 graph and are excluded from these frozen novelty strata.",
        "  This means whole-city zero in-degree, not support lost after partition pruning;",
        "  nodes without upstream neighbours still have a self-feature pathway.",
        "- Primary estimand: Delta_novelty = mean_high - mean_low (High-Low contrast). A positive",
        "  value means high-novelty positive cells are ranked higher by GraphSAGE on G2.",
        "- Medium-Low and High-Medium are secondary; RQ3 is exploratory and attaches no gate.",
        "- This absolute-rank High-Low contrast does not isolate a graph-specific difficulty.",
        "  The supplementary paired diagnostic in docs/rq3_paired_rank_results.md instead uses",
        "  rank_G2 - rank_G0 for the same positive cells, banks, seeds and folds with matched",
        "  weighting, and asks whether that gain changes from Low to High novelty. It also",
        "  describes whole-city no-upstream cells and cells losing support after pruning separately.",
        "  The paired diagnostic supplements, and does not replace, this frozen RQ3 outcome.",
        "- The novelty neighbourhood is the whole-city union of unique one/two-hop upstream",
        "  nodes, excluding self. Actual model propagation uses partition-local edges; two hops",
        "  span at most about 566 m of geometric path and do not represent a complete catchment.",
        "- G2 is a graph relationship also used by Fixed-filter. These GraphSAGE ranks do not",
        "  establish whether that relationship itself contains useful information.",
        "- These are recorded-positive-versus-background, event-conditioned ranks; they are",
        "  neither flood probabilities nor evidence of future warning or cross-city transfer.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    result = run_rq3_novelty(root, output, replicates=args.replicates)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
