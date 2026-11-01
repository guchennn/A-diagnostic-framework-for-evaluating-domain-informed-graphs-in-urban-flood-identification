"""Frozen CPU spatial protocol recovery study with independent generated replicates.

The fitted model is logistic regression, not a surrogate claim about FF/SAGE optimization.
Edges are defined by coordinates before outcomes are generated. No truth probabilities,
labels, split identifiers or structural group labels are used as predictive features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import expit

from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.supplementary_mechanisms import ap_terms


def midpoint_statistics(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Positive AP terms and half-tie candidate/background percentile ranks."""
    y, scores = np.asarray(labels), np.asarray(scores, float)
    if y.ndim != 1 or y.shape != scores.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("aligned binary labels with both classes required")
    if not np.isfinite(scores).all():
        raise ValueError("finite scores required")
    positive = y == 1
    p, bg, ordered = scores[positive], np.sort(scores[~positive]), np.sort(scores)
    candidate = (
        np.searchsorted(ordered, p, side="left") + np.searchsorted(ordered, p, side="right")
    ) / (2 * len(scores))
    background = (np.searchsorted(bg, p, side="left") + np.searchsorted(bg, p, side="right")) / (
        2 * len(bg)
    )
    return np.column_stack((ap_terms(y, scores, np.ones(len(y)))[positive], candidate, background))


def spatial_split(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row, col = np.indices((config["grid_rows"], config["grid_columns"]))
    row, col = row.ravel(), col.ravel()
    split = np.full(len(row), -1, dtype=int)
    for code, name in enumerate(("train_rows", "validation_rows", "test_rows")):
        start, stop = config[name]
        mask = (row >= start) & (row < stop)
        if np.any(split[mask] != -1) or not 0 <= start < stop <= config["grid_rows"]:
            raise ValueError("invalid or overlapping spatial splits")
        split[mask] = code
    if set(split) != {-1, 0, 1, 2}:
        raise ValueError("spatial splits must include unused buffer rows")
    return row, col, split


def isolated_edges(edges: np.ndarray, split: np.ndarray) -> np.ndarray:
    return edges[(split[edges[:, 0]] == split[edges[:, 1]]) & (split[edges[:, 0]] >= 0)]


def degree_preserving_rewire(
    edges: np.ndarray, split: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Directed double-edge swaps within split; forbid self loops and duplicate edges."""
    result = edges.copy()
    for code in (0, 1, 2):
        indices = np.flatnonzero(split[result[:, 0]] == code)
        occupied = {tuple(edge) for edge in result[indices]}
        if len(indices) < 2:
            continue
        for _ in range(10 * len(indices)):
            i, j = rng.choice(indices, 2, replace=False)
            a, b = result[i]
            c, d = result[j]
            if a == d or c == b or (a, d) in occupied or (c, b) in occupied:
                continue
            occupied.remove((a, b))
            occupied.remove((c, d))
            occupied.update(((a, d), (c, b)))
            result[i, 1], result[j, 1] = d, b
    return result


def neighbor_mean(features: np.ndarray, edges: np.ndarray) -> np.ndarray:
    total = np.zeros_like(features)
    count = np.zeros(len(features))
    np.add.at(total, edges[:, 1], features[edges[:, 0]])
    np.add.at(count, edges[:, 1], 1)
    return total / np.maximum(count[:, None], 1)


def structural_groups(edges: np.ndarray, full: np.ndarray, n: int) -> dict[str, np.ndarray]:
    incoming = [set() for _ in range(n)]
    for source, target in edges:
        incoming[target].add(int(source))
    full_degree = np.bincount(full[:, 1], minlength=n)
    degree = np.array([len(sources) for sources in incoming])
    second = [
        set().union(*(incoming[source] for source in sources)) if sources else set()
        for sources in incoming
    ]
    depth = np.array(
        [
            "2" if upstream else "1" if sources else "0"
            for sources, upstream in zip(incoming, second, strict=True)
        ]
    )
    count = np.array(
        [
            len((sources | upstream) - {i})
            for i, (sources, upstream) in enumerate(zip(incoming, second, strict=True))
        ]
    )
    k = np.where(count <= 2, count.astype(str), np.where(count <= 4, "3-4", "5+"))
    support = np.where(full_degree == 0, "root", np.where(degree == 0, "cut", "retained"))
    return {"L": depth, "K": k, "support": support}


def generate(config: dict, seed: int) -> dict:
    row, col, split = spatial_split(config)
    rng = np.random.default_rng(seed)
    n = len(row)
    features = rng.normal(size=(n, config["feature_count"]))
    # Horizontal four-node chains are geometric and independent of labels/features.
    source = np.flatnonzero((col % 4 != 3) & (col + 1 < config["grid_columns"]))
    full = np.column_stack((source, source + 1))
    useful = isolated_edges(full, split)
    upstream = neighbor_mean(features, full)[:, 0]
    probability = expit(
        config["label_intercept"]
        + config["own_coefficient"] * features[:, 0]
        + config["upstream_coefficient"] * upstream
    )
    labels = (rng.random(n) < probability).astype(np.int8)
    random_labels = (rng.random(n) < config["random_label_probability"]).astype(np.int8)
    source = np.flatnonzero(row + 1 < config["grid_rows"])
    irrelevant = isolated_edges(np.column_stack((source, source + config["grid_columns"])), split)
    wrong = degree_preserving_rewire(useful, split, rng)
    # Shifted eight-column tiles cut one of two otherwise identical chain types.
    tile = (col - 1) // 8
    truncated = useful[tile[useful[:, 0]] == tile[useful[:, 1]]]
    return {
        "features": features,
        "labels": labels,
        "random_labels": random_labels,
        "split": split,
        "full": full,
        "edges": {
            "useful": useful,
            "irrelevant": irrelevant,
            "degree_preserving_wrong": wrong,
            "boundary_truncated": truncated,
            "random_label": useful,
        },
        "row": row,
        "col": col,
    }


def train_standardize(features: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict]:
    mean, scale = features[train].mean(axis=0), features[train].std(axis=0)
    scale = np.where(scale > 0, scale, 1)
    return (features - mean) / scale, {"mean": mean, "scale": scale}


def fit_scores(features: np.ndarray, labels: np.ndarray, split: np.ndarray, config: dict):
    """Select regularization on spatial validation AP; fit/scaling use train rows only."""
    train, validation = split == 0, split == 1
    standardized, _ = train_standardize(features, train)
    x = np.column_stack((np.ones(len(features)), standardized))
    x_train, y_train = x[train], labels[train]
    candidates = []
    for regularization in config["regularization_candidates"]:

        def objective(beta, penalty_weight=regularization):
            logits = x_train @ beta
            loss = np.mean(np.logaddexp(0, logits) - y_train * logits)
            penalty = penalty_weight * np.dot(beta[1:], beta[1:]) / 2
            gradient = x_train.T @ (expit(logits) - y_train) / len(y_train)
            gradient[1:] += penalty_weight * beta[1:]
            return loss + penalty, gradient

        fitted = minimize(
            objective,
            np.zeros(x.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": config["optimizer_max_iterations"], "ftol": 1e-12, "gtol": 1e-7},
        )
        if not fitted.success:
            raise RuntimeError(f"logistic fit failed: {fitted.message}")
        score = x @ fitted.x
        validation_ap = midpoint_statistics(labels[validation], score[validation])[:, 0].sum()
        candidates.append(
            {
                "regularization": regularization,
                "validation_ap": validation_ap,
                "iterations": fitted.nit,
                "training_objective": fitted.fun,
                "score": score,
            }
        )
    # Ordered candidates resolve exact ties; test scores never enter the selection.
    best = max(range(len(candidates)), key=lambda i: candidates[i]["validation_ap"])
    selected = candidates[best]["score"]
    audit = [
        {key: value for key, value in record.items() if key != "score"} | {"selected": i == best}
        for i, record in enumerate(candidates)
    ]
    return selected, audit


def group_records(difference: np.ndarray, labels: np.ndarray, groups: dict) -> list[dict]:
    result = []
    positive = labels == 1
    for analysis, membership in groups.items():
        totals = []
        expected = {
            "L": ["0", "1", "2"],
            "K": ["0", "1", "2", "3-4", "5+"],
            "support": ["root", "cut", "retained"],
        }[analysis]
        for group in expected:
            selected = difference[membership[positive] == group]
            n = len(selected)
            contribution = selected[:, 0].sum()
            totals.append(contribution)
            result.append(
                {
                    "analysis": analysis,
                    "group": group,
                    "positive_count": n,
                    "positive_weight_share": n / positive.sum(),
                    "ap_contribution": contribution,
                    "mean_precision_change": selected[:, 0].mean() * positive.sum()
                    if n
                    else np.nan,
                    "candidate_midrank_gain": selected[:, 1].mean() if n else np.nan,
                    "background_midrank_gain": selected[:, 2].mean() if n else np.nan,
                }
            )
        if not np.isclose(sum(totals), difference[:, 0].sum(), atol=1e-12):
            raise ValueError("synthetic group AP additivity failed")
    return result


def run(config_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("new output directory required; never overwrite observed results")
    config = yaml.safe_load(config_path.read_text("utf-8"))
    seeds = config["generation_seeds"]
    if len(seeds) < 10 or len(seeds) != len(set(seeds)):
        raise ValueError("at least ten distinct predeclared generation seeds required")
    output.mkdir(parents=True)
    frozen = {
        "config": config,
        "config_sha256": sha256_file(config_path),
        "source_sha256": sha256_file(Path(__file__)),
        "status": "frozen_before_scores",
    }
    (output / "frozen_design.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    metrics, groups, fit_records, boundary = [], [], [], []
    for seed in seeds:
        data = generate(config, seed)
        test = data["split"] == 2
        feature = data["features"]
        cached_own, details = {}, {}
        for condition in config["conditions"]:
            label_kind = "random_labels" if condition == "random_label" else "labels"
            y = data[label_kind]
            if label_kind not in cached_own:
                own, audit = fit_scores(feature, y, data["split"], config)
                cached_own[label_kind] = own
                fit_records.extend(
                    {"seed": seed, "condition": label_kind, "model": "own", **record}
                    for record in audit
                )
            own = cached_own[label_kind]
            edges = data["edges"][condition]
            expanded = np.column_stack((feature, neighbor_mean(feature, edges)))
            score, audit = fit_scores(expanded, y, data["split"], config)
            fit_records.extend(
                {"seed": seed, "condition": condition, "model": "neighbor", **record}
                for record in audit
            )
            baseline = midpoint_statistics(y[test], own[test])
            stats = midpoint_statistics(y[test], score[test])
            difference = stats - baseline
            details[condition] = stats
            metrics.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "test_count": int(test.sum()),
                    "positive_count": int(y[test].sum()),
                    "own_ap": baseline[:, 0].sum(),
                    "neighbor_ap": stats[:, 0].sum(),
                    "ap_gain": difference[:, 0].sum(),
                    "own_pb_auc": baseline[:, 2].mean(),
                    "neighbor_pb_auc": stats[:, 2].mean(),
                    "pb_auc_gain": difference[:, 2].mean(),
                    "test_edges": int((data["split"][edges[:, 0]] == 2).sum()),
                }
            )
            structural = structural_groups(edges, data["full"], len(feature))
            for record in group_records(
                difference, y[test], {k: v[test] for k, v in structural.items()}
            ):
                groups.append({"seed": seed, "condition": condition, **record})
        truncated = structural_groups(
            data["edges"]["boundary_truncated"], data["full"], len(feature)
        )
        difference = details["boundary_truncated"] - details["useful"]
        for record in group_records(
            difference, data["labels"][test], {k: v[test] for k, v in truncated.items()}
        ):
            boundary.append({"seed": seed, "condition": "truncated_minus_useful", **record})
        print(f"Completed frozen synthetic generation seed {seed}.", flush=True)
    frame = pd.DataFrame(metrics)
    summary = frame.groupby("condition")[["ap_gain", "pb_auc_gain"]].agg(
        ["mean", "min", "max", lambda values: int((values > 0).sum())]
    )
    summary.columns = [
        f"{metric}_{'positive_replicates' if name == '<lambda_0>' else name}"
        for metric, name in summary.columns
    ]
    paired_rows = []
    wide = frame.pivot(index="seed", columns="condition", values=["neighbor_ap", "neighbor_pb_auc"])
    for other in ("irrelevant", "degree_preserving_wrong", "boundary_truncated"):
        for metric in ("neighbor_ap", "neighbor_pb_auc"):
            values = wide[metric, "useful"] - wide[metric, other]
            paired_rows.extend(
                {
                    "seed": seed,
                    "contrast": f"useful_minus_{other}",
                    "metric": metric,
                    "value": value,
                }
                for seed, value in values.items()
            )
    boundary_frame = pd.DataFrame(boundary)
    for metric in ("mean_precision_change", "background_midrank_gain"):
        part = boundary_frame.loc[boundary_frame.analysis.eq("support")].pivot(
            index="seed", columns="group", values=metric
        )
        paired_rows.extend(
            {
                "seed": seed,
                "contrast": "truncation_cut_minus_retained",
                "metric": metric,
                "value": value,
            }
            for seed, value in (part["cut"] - part["retained"]).items()
        )
    paired = pd.DataFrame(paired_rows)
    paired_summary = paired.groupby(["contrast", "metric"]).value.agg(
        mean="mean",
        minimum="min",
        maximum="max",
        positive_replicates=lambda values: int((values > 0).sum()),
        negative_replicates=lambda values: int((values < 0).sum()),
    )
    tables = {
        "replicate_metrics.csv": frame,
        "summary.csv": summary.reset_index(),
        "group_replicates.csv": pd.DataFrame(groups),
        "fit_audit.csv": pd.DataFrame(fit_records),
        "boundary_group_replicates.csv": boundary_frame,
        "paired_replicates.csv": paired,
        "paired_summary.csv": paired_summary.reset_index(),
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    manifest = frozen | {
        "status": "complete",
        "replicates": len(seeds),
        "conditions": len(config["conditions"]),
        "candidate_fits": len(fit_records),
        "outputs": {name: sha256_file(output / name) for name in tables},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Synthetic spatial diagnostic validation\n\n"
        "Ten independent generated 96 x 48 grids, five predeclared conditions, no score-based "
        "generator revision. frozen_design.json was written before fitting/scoring. Spatial row "
        "bands (32 train, 24 validation, 32 test rows) have four-row unused buffers; all used "
        "edges are split-local. Own features are three independent standard normal variables. "
        "Coordinate-defined horizontal four-node chains generate Bernoulli outcomes from "
        "sigmoid(-2 + 0.5*x0 + 2*upstream_x0); missing upstream is zero. Scores are fitted "
        "logistic models on own versus own-plus-mean-neighbor features. Scaling and optimization "
        "use train rows only; three fixed regularizations are selected by validation AP. "
        "No test outcomes inform model selection; truth probabilities and group labels are not "
        "features. These are fully observed synthetic outcomes, not report ascertainment.\n\n"
        "Irrelevant relation uses vertical adjacency independent of the horizontal signal. "
        "Wrong relation uses directed degree-preserving within-split swaps. Shifted eight-column "
        "tiles truncate half the chain starts. Random-label condition uses independent Bernoulli "
        "0.2 labels. All candidate fits and all successful/unsuccessful replicate outcomes are "
        "reported. Structural groups describe the evaluated relation; paired truncation losses "
        "use the same truncated groups for both models. K counts distinct one/two-hop ancestors. "
        "All metrics use complete test bands with exact half ties. AP contributions are additive. "
        "Ranges are observed independent-generation ranges, not city confidence intervals. "
        "This recovers behavior of an evaluation protocol on a simple known generator; it does "
        "not validate original FF/SAGE optimization or real flood/report mechanisms.\n\n"
        "Run: `.venv/Scripts/python.exe -m floodgnn_ceus.synthetic_diagnostic_validation`\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/synthetic_diagnostic_validation_v1.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/synthetic_diagnostic_validation_20261012")
    )
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
