"""Read trusted local synthetic fit caches; no model fitting occurs here."""

import pickle
from pathlib import Path

import numpy as np

EXPECTED_CONFIG = {
    "protocol_id": "synthetic-pu-regime-v1-20261029",
    "grid_rows": 192,
    "grid_columns": 96,
    "train_rows": [0, 64],
    "validation_rows": [72, 120],
    "test_rows": [128, 192],
    "feature_count": 3,
    "label_intercept": -2.0,
    "own_coefficient": 0.5,
    "upstream_coefficients": [0.0, 0.5],
    "report_probability": 0.25,
    "revealed_positives": 100,
    "unlabelled_cells": 2000,
    "trajectories": 5,
    "own_seeds": 9,
    "graph_seeds": 3,
    "grids_per_study": 5,
    "studies": 10,
    "network": {"hidden": 32, "dropout": 0.2, "learning_rate": 0.001, "weight_decay": 0.0001},
    "max_epochs": 250,
    "patience": 30,
    "null_draws": 2000,
    "base_seed": 7102901,
}


def positive_terms(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per-positive AP terms (threshold includes ties) and background mid-ranks, as for cities."""
    positive, background = scores[labels == 1], np.sort(scores[labels == 0])
    positive_sorted = np.sort(positive)
    tp = len(positive) - np.searchsorted(positive_sorted, positive, side="left")
    fp = len(background) - np.searchsorted(background, positive, side="left")
    rank = (
        np.searchsorted(background, positive, side="left")
        + np.searchsorted(background, positive, side="right")
    ) / (2 * len(background))
    return np.column_stack([tp / (tp + fp) / len(positive), rank])


def load_cache(root: Path, beta: float, study: int, grid: int) -> dict:
    """Load original local pickle artifacts only; never load untrusted downloaded pickles."""
    base = root / "results/synthetic_pu_regime_20261029_cache"
    extra = root / "results/synthetic_pu_regime_20261029_controls_cache"
    result = pickle.loads((base / f"{beta}_{study}_{grid}.pkl").read_bytes())
    if beta == 0.0:
        result["stats"].update(pickle.loads((extra / f"{beta}_{study}_{grid}.pkl").read_bytes()))
    return result
