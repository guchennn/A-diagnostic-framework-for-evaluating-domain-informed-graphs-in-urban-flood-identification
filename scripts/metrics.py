"""Descriptive, paired reanalysis of existing predictions; never fits/selects models."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from floodgnn_ceus.inference import validate_paired_evaluation, weighted_auprc

SEED = 20261011
REPLICATES = 10000
METHODS = ("G0", "G1", "G2", "G4", "Fixed-filter")
GROUP = ["city", "outer_fold", "trajectory", "seed", "bank_id", "method"]


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """AP and expected recall at exact fractional population budgets, including ties."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("aligned binary labels with both classes required")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    result = {"ap": weighted_auprc(y, s, np.ones(len(y)))}
    order = np.argsort(-s, kind="stable")
    sorted_scores, sorted_labels = s[order], y[order]
    for fraction in (0.01, 0.05, 0.10):
        budget = fraction * len(y)
        cutoff = sorted_scores[int(np.ceil(budget)) - 1]
        above, tied = sorted_scores > cutoff, sorted_scores == cutoff
        inclusion = (budget - int(above.sum())) / int(tied.sum())
        recall = (sorted_labels[above].sum() + inclusion * sorted_labels[tied].sum()) / y.sum()
        result[f"recall_top_{int(fraction * 100)}pct"] = float(recall)
    return result


def paired_wide(frame: pd.DataFrame, keys: list[str], column: str, value: str) -> pd.DataFrame:
    """Fail closed if a method/condition is missing, duplicated, or nonfinite."""
    if frame.duplicated([*keys, column]).any():
        raise ValueError("duplicate paired observations")
    wide = frame.pivot(index=keys, columns=column, values=value).sort_index()
    if wide.empty or not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise ValueError("missing or nonfinite paired observations")
    return wide


def describe_folds(
    frame: pd.DataFrame, keys: list[str], value: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-fold mean, common fold bootstrap quantiles, and leave-one-fold-out means."""
    indices = np.random.default_rng(SEED).integers(0, 5, size=(REPLICATES, 5))
    summary, lofo = [], []
    for group, rows in frame.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        meta = dict(zip(keys, group, strict=True))
        rows = rows.sort_values("outer_fold")
        if rows.outer_fold.tolist() != list(range(5)):
            raise ValueError("each paired contrast must contain exactly folds 0..4")
        values = rows[value].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("fold values must be finite")
        quantiles = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
        summary.append(
            {
                **meta,
                "mean": values.mean(),
                "resampling_q025": quantiles[0],
                "resampling_q975": quantiles[1],
                "positive_folds": int((values > 0).sum()),
                "zero_folds": int((values == 0).sum()),
                "negative_folds": int((values < 0).sum()),
                "n_folds": 5,
            }
        )
        for omitted in range(5):
            lofo.append(
                {**meta, "omitted_fold": omitted, "mean": np.delete(values, omitted).mean()}
            )
    return pd.DataFrame(summary), pd.DataFrame(lofo)


def sampled_fold_metrics(path: Path, methods: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.budget.astype(str).eq("100") & frame.method.isin(methods)]
    metrics = []
    for _, city_rows in frame.groupby("city"):
        validate_paired_evaluation(city_rows, methods)
    for key, rows in frame.groupby(GROUP, sort=True):
        metrics.append(
            {
                **dict(zip(GROUP, key, strict=True)),
                "pb_ap": weighted_auprc(
                    rows.sample_role.eq("positive").to_numpy(dtype=np.int8),
                    rows.score.to_numpy(),
                    np.ones(len(rows)),
                ),
            }
        )
    units = pd.DataFrame(metrics)
    for _, rows in units.groupby(["city", "method", "outer_fold"]):
        if (
            len(rows) != 45
            or set(rows.trajectory.astype(str)) != {"1", "2", "3", "4", "5"}
            or set(rows.seed) != {"NEURAL-1", "NEURAL-2", "NEURAL-3"}
            or set(rows.bank_id.astype(str)) != {"1", "2", "3"}
        ):
            raise ValueError("expected complete 5 trajectory x 3 seed x 3 bank design")
    return units.groupby(["city", "outer_fold", "method"], as_index=False).pb_ap.mean()


def joint_feature_contrasts(full: pd.DataFrame, scarce: pd.DataFrame) -> pd.DataFrame:
    """Form interactions before resampling, preserving FULL/100 fold covariance."""
    full = full.rename(columns={"pb_auprc": "pb_ap"}).assign(budget="FULL")
    scarce = scarce.copy().assign(budget="100")
    parts = scarce.method.str.split("|", regex=False, expand=True)
    scarce["method"] = parts[0]
    scarce["regime"] = parts[1].map({"F18": "full", "F16-noD8": "noD8", "F14-noNeigh": "noNeigh"})
    all_rows = pd.concat([full, scarce], ignore_index=True)
    wide = paired_wide(all_rows, ["outer_fold", "budget", "regime"], "method", "pb_ap")
    if set(wide.columns) != {"G0", "G1", "G2", "XGB"}:
        raise ValueError("feature comparison methods incomplete")
    rows = []
    for fold in range(5):
        for baseline in ("G0", "G1", "XGB"):
            d = wide["G2"] - wide[baseline]
            for regime in ("full", "noD8", "noNeigh"):
                n100, full_value = d.loc[fold, "100", regime], d.loc[fold, "FULL", regime]
                rows.append(
                    {
                        "outer_fold": fold,
                        "baseline": baseline,
                        "contrast": f"scarcity_{regime}",
                        "n100_component": n100,
                        "full_component": full_value,
                        "effect": n100 - full_value,
                    }
                )
            for regime in ("noD8", "noNeigh"):
                n100 = d.loc[fold, "100", regime] - d.loc[fold, "100", "full"]
                full_value = d.loc[fold, "FULL", regime] - d.loc[fold, "FULL", "full"]
                rows.append(
                    {
                        "outer_fold": fold,
                        "baseline": baseline,
                        "contrast": f"interaction_{regime}",
                        "n100_component": n100,
                        "full_component": full_value,
                        "effect": n100 - full_value,
                    }
                )
    return pd.DataFrame(rows)


def full_background_metrics(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    paired_domains, positives_by_fold = {}, {}
    member_hashes = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.name.endswith("/status.json"):
                continue
            status = json.load(archive.extractfile(member))
            task = status.get("task", {})
            if task.get("budget") != "100" or task.get("experiment_id") not in {"E1", "E2"}:
                continue
            method = (
                task.get("graph_id") if task.get("model") == "GraphSAGE-G2" else task.get("model")
            )
            if method not in METHODS:
                continue
            if status["status"] != "success" or any(
                task[key] != expected
                for key, expected in {
                    "label_layer": "STRICT-LOC",
                    "feature_set": "FULL-COMMON",
                    "reveal_mechanism": "uniform",
                    "realization_id": "NA",
                }.items()
            ):
                raise ValueError("unexpected archive task status or protocol scope")
            name = member.name.rsplit("/", 1)[0] + "/oof.parquet"
            data = archive.extractfile(name).read()
            digest = hashlib.sha256(data).hexdigest()
            if digest != status["oof_sha256"]:
                raise ValueError("archive prediction hash mismatch")
            member_hashes[name] = digest
            frame = pd.read_parquet(io.BytesIO(data))
            if not frame.status.eq("success").all():
                raise ValueError("unsuccessful predictions")
            domain = frame.loc[frame.bank_id.eq("NA")].sort_values("grid_id")
            if domain.empty or domain.grid_id.duplicated().any():
                raise ValueError("missing or duplicated full-background predictions")
            positive_sets = [
                frozenset(
                    frame.loc[frame.bank_id.eq(bank) & frame.sample_role.eq("positive"), "grid_id"]
                )
                for bank in ("1", "2", "3")
            ]
            positive = positive_sets[0]
            if not positive or any(p != positive for p in positive_sets):
                raise ValueError("sampled banks disagree on held-out recorded positives")
            if not positive.issubset(set(domain.grid_id)):
                raise ValueError("recorded positive missing from full background")
            city, fold = task["city"], int(task["outer_fold"])
            unit = (city, fold, task["trajectory"], task["seed"])
            domain_hash = hashlib.sha256("\n".join(domain.grid_id).encode()).hexdigest()
            if paired_domains.setdefault(unit, domain_hash) != domain_hash:
                raise ValueError("unpaired full-background domains")
            if positives_by_fold.setdefault((city, fold), positive) != positive:
                raise ValueError("recorded positives differ between paired tasks")
            labels = domain.grid_id.isin(positive).to_numpy(dtype=np.int8)
            rows.append(
                dict(
                    city=city,
                    outer_fold=fold,
                    trajectory=task["trajectory"],
                    seed=task["seed"],
                    method=method,
                    domain_count=len(domain),
                    positive_count=len(positive),
                    **ranking_metrics(labels, domain.score.to_numpy()),
                )
            )
    units = pd.DataFrame(rows)
    paired_wide(units, ["city", "outer_fold", "trajectory", "seed"], "method", "ap")
    if len(units) != 750 or set(units.method) != set(METHODS):
        raise ValueError("expected 750 paired tasks")
    positive_counts = {}
    for city, expected in {"zhengzhou": 791, "beijing": 296}.items():
        sets = [positives_by_fold[city, fold] for fold in range(5)]
        union = set().union(*sets)
        if sum(map(len, sets)) != len(union) or len(union) != expected:
            raise ValueError("recorded-positive folds overlap or do not recover STRICT-LOC total")
        positive_counts[city] = len(union)
    counts = units.groupby(["city", "outer_fold", "method"]).size()
    if not counts.eq(15).all():
        raise ValueError("expected 15 trajectory-seed units in each fold-method")
    fold_rows = units.groupby(["city", "outer_fold", "method"], as_index=False)[
        [
            "domain_count",
            "positive_count",
            "ap",
            "recall_top_1pct",
            "recall_top_5pct",
            "recall_top_10pct",
        ]
    ].mean()
    return fold_rows, {"verified_oof_members": member_hashes, "strict_loc_counts": positive_counts}
