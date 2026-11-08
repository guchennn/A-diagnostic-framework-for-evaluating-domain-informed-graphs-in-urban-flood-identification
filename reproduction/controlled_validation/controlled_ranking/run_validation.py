"""Frozen, independent score-distribution validation; outputs are probabilities."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, roots_hermitenorm

ROOT = Path(__file__).resolve().parent
Q = [(0.2, 0.2, 0.6), (0.4, 0.4, 0.2)]
STRENGTHS = [0, 0.25, 0.5, 1]
SCENARIOS = ["shared", "specific", "redistribution"]
WEIGHTS = [1.0, 6000 / 11400]
METRICS = ["global_ap", "within_ap", "allocation", "rank"]
V = 1.0625


def offsets(scenario: str, a: float) -> np.ndarray:
    t = np.zeros((3, 3))
    if scenario == "shared":
        t[1:, 0] = -a
    elif scenario == "specific":
        t[2, 0] = -a
    elif scenario == "redistribution":
        t[:2, 0] = a
        t[2, 1] = a
    else:
        raise ValueError(scenario)
    return t


def precision_terms(y: np.ndarray, score: np.ndarray, w: float) -> np.ndarray:
    p = np.sort(score[y])
    b = np.sort(score[~y])
    sp = score[y]
    tp = len(p) - np.searchsorted(p, sp, side="left")
    fp = len(b) - np.searchsorted(b, sp, side="left")
    return tp / (tp + w * fp)


def metrics(y: np.ndarray, g: np.ndarray, score: np.ndarray, w: float) -> list[np.ndarray]:
    p = precision_terms(y, score, w)
    bg = np.sort(score[~y])
    sp = score[y]
    ranks = (np.searchsorted(bg, sp, "left") + np.searchsorted(bg, sp, "right")) / (2 * len(bg))
    allocation = np.array([p[g[y] == k].sum() / y.sum() for k in range(3)])
    within = np.array([precision_terms(y[g == k], score[g == k], w).mean() for k in range(3)])
    rank = np.array([ranks[g[y] == k].mean() for k in range(3)])
    assert abs(allocation.sum() - p.mean()) < 1e-12
    return [np.array([p.mean()]), within, allocation, rank]


def oracle(
    q: tuple[float, float, float], t: np.ndarray, w: float, order: int = 128
) -> list[np.ndarray]:
    q = np.array(q)
    x, weights = roots_hermitenorm(order)
    s = 1.5 + t[:, None] + np.sqrt(V) * x[None, :]
    tp = 0.05 * (ndtr((1.5 + t[None, None, :] - s[:, :, None]) / np.sqrt(V)) * q).sum(axis=2)
    fp = 0.95 * (ndtr((t[None, None, :] - s[:, :, None]) / np.sqrt(V)) * q).sum(axis=2)
    pr = np.divide(tp, tp + w * fp, out=np.ones_like(tp), where=(tp + w * fp) > 0)
    alloc = q * (pr * weights).sum(axis=1) / np.sqrt(2 * np.pi)
    rank = (ndtr((1.5 + t[:, None] - t[None, :]) / np.sqrt(2 * V)) * q).sum(axis=1)
    # Identical prevalence and within-group separation in every group.
    p0 = 0.05 * ndtr(-x)
    b0 = 0.95 * ndtr(-(1.5 + np.sqrt(V) * x) / np.sqrt(V))
    within = (
        np.divide(p0, p0 + w * b0, out=np.ones_like(p0), where=p0 + w * b0 > 0) * weights
    ).sum() / np.sqrt(2 * np.pi)
    return [np.array([alloc.sum()]), np.repeat(within, 3), alloc, rank]


def dataset(
    qid: int, phase: int, scenario_id: int, aid: int, rep: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Spawn-key tuples are unique; no mutable random streams are shared.
    rng = np.random.default_rng(
        np.random.SeedSequence(2026110803, spawn_key=(phase, qid, scenario_id, aid, rep))
    )
    q = np.array(Q[qid])
    y = np.r_[np.ones(600, bool), np.zeros(11400, bool)]
    g = np.r_[
        np.repeat(np.arange(3), np.rint(600 * q).astype(int)),
        np.repeat(np.arange(3), np.rint(11400 * q).astype(int)),
    ]
    z = 1.5 * y + rng.normal(size=len(y))
    scores = z[None, :] + rng.normal(0, 0.25, size=(3, len(y)))
    return y, g, scores


def differences(
    y: np.ndarray, g: np.ndarray, scores: np.ndarray
) -> dict[tuple[int, int, str], np.ndarray]:
    result = {}
    for wid, w in enumerate(WEIGHTS):
        vals = [metrics(y, g, s, w) for s in scores]
        for ref in range(2):
            for mid, name in enumerate(METRICS):
                result[(wid, ref, name)] = vals[2][mid] - vals[ref][mid]
    return result


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def run() -> None:
    start = time.time()
    thresholds = {}
    calibration_rows = []
    for qid in range(2):
        for rep in range(300):
            values = differences(*dataset(qid, 0, 0, 0, rep))
            for key, value in values.items():
                thresholds.setdefault((qid, *key), []).append(np.max(np.abs(value)))
                calibration_rows.append((qid, rep, *key, np.max(np.abs(value))))
        print(f"Calibration q={qid} complete", flush=True)
    thresholds = {
        key: float(np.quantile(value, 0.95, method="higher")) for key, value in thresholds.items()
    }
    pd.DataFrame(
        calibration_rows,
        columns=["composition", "replicate", "weight_id", "reference_id", "metric", "max_abs"],
    ).to_csv(ROOT / "calibration.csv", index=False)
    pd.DataFrame(
        [(*k, v) for k, v in thresholds.items()],
        columns=["composition", "weight_id", "reference_id", "metric", "threshold"],
    ).to_csv(ROOT / "thresholds.csv", index=False)
    oracle_rows, rows = [], []
    convergence = 0.0
    for qid, q in enumerate(Q):
        for sid, scenario in enumerate(SCENARIOS):
            for aid, a in enumerate(STRENGTHS):
                t = offsets(scenario, a)
                targets = {}
                for wid, w in enumerate(WEIGHTS):
                    vals = [oracle(q, tm, w) for tm in t]
                    check = [oracle(q, tm, w, 256) for tm in t]
                    err = max(
                        np.max(np.abs(v - c)) for vm, cm in zip(vals, check) for v, c in zip(vm, cm)
                    )
                    convergence = max(convergence, float(err))
                    assert err < 1e-7, f"Quadrature convergence {err}"
                    for ref in range(2):
                        for mid, metric in enumerate(METRICS):
                            target = vals[2][mid] - vals[ref][mid]
                            # Distribution identities fix exact zeros independently of quadrature.
                            if (
                                a == 0
                                or metric == "within_ap"
                                or (scenario == "shared" and ref == 1)
                                or (scenario == "redistribution" and metric == "global_ap")
                            ):
                                target = np.zeros_like(target)
                            targets[wid, ref, metric] = target
                            for group, v in enumerate(target):
                                oracle_rows.append((qid, scenario, a, wid, ref, metric, group, v))
                for rep in range(200):
                    y, g, scores = dataset(qid, 1, sid, aid, rep)
                    scores += t[:, g]
                    values = differences(y, g, scores)
                    for key, value in values.items():
                        threshold = thresholds[qid, *key]
                        for group, (v, target) in enumerate(zip(value, targets[key])):
                            rows.append(
                                (qid, scenario, a, rep, *key, group, v, float(target), threshold)
                            )
                print(f"Test q={qid} {scenario} a={a} complete", flush=True)
    names = [
        "composition",
        "scenario",
        "offset",
        "replicate",
        "weight_id",
        "reference_id",
        "metric",
        "group",
        "estimate",
        "oracle",
        "threshold",
    ]
    df = pd.DataFrame(rows, columns=names)
    df.to_csv(ROOT / "replicates.csv", index=False)
    pd.DataFrame(
        oracle_rows, columns=[n for n in names if n not in ["replicate", "estimate", "threshold"]]
    ).to_csv(ROOT / "oracles.csv", index=False)
    groups = ["composition", "scenario", "offset", "weight_id", "reference_id", "metric", "group"]
    out = []
    for key, d in df.groupby(groups, sort=False):
        target = d.oracle.iloc[0]
        error = d.estimate - target
        detected = d.estimate.abs() > d.threshold
        correct = detected & (np.sign(d.estimate) == np.sign(target))
        rate = float(correct.mean()) if abs(target) > 1e-6 else np.nan
        out.append(
            (
                *key,
                target,
                d.estimate.mean(),
                error.mean(),
                np.sqrt(np.mean(error**2)),
                d.estimate.std(ddof=1) / np.sqrt(len(d)),
                detected.mean(),
                rate,
            )
        )
    summary = pd.DataFrame(
        out,
        columns=groups
        + ["oracle", "mean", "bias", "rmse", "mcse", "detection_rate", "correct_sign_detection"],
    )
    summary.to_csv(ROOT / "summary.csv", index=False)
    family = []
    for key, d in df.groupby(groups[:-1], sort=False):
        hit = d.assign(hit=d.estimate.abs() > d.threshold).groupby("replicate").hit.any()
        lo, hi = wilson(int(hit.sum()), len(hit))
        family.append((*key, float(hit.mean()), lo, hi, bool((d.oracle.abs() < 1e-6).all())))
    pd.DataFrame(
        family,
        columns=groups[:-1] + ["any_detection", "wilson_low", "wilson_high", "all_targets_zero"],
    ).to_csv(ROOT / "family_detection.csv", index=False)
    manifest = {
        "protocol_sha256": hashlib.sha256(
            (Path(__file__).resolve().parent.parent / "controlled_ranking_protocol.md").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed_entropy": 2026110803,
        "spawn_key": "(phase, composition, scenario_id, strength_id, replicate); phase 0 calibration, 1 test",
        "calibration_datasets": 600,
        "test_datasets": 4800,
        "cells_per_dataset": 12000,
        "quadrature_max_error": convergence,
        "seconds": time.time() - start,
        "strong_rank_baseline": "Identical mathematical readout, therefore identical results; no separate performance advantage claimed.",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    ROOT = parser.parse_args().output_dir.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    run()
