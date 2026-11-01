"""Frozen task keys, preprocessing, selection, and failure rules for E1-E3."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import psutil

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file

TASK_FIELDS = (
    "protocol_id",
    "experiment_id",
    "city",
    "label_layer",
    "reveal_mechanism",
    "outer_fold",
    "budget",
    "trajectory",
    "feature_set",
    "model",
    "graph_id",
    "realization_id",
    "config",
    "seed",
)
ALTERNATIVE_ORDER = (
    "KDE",
    "XGBoost",
    "LightGBM",
    "Neighbor-XGB",
    "MLP",
    "Fixed-filter",
)
PROBE_CELL = {
    "budget": "20",
    "city": "zhengzhou",
    "inner_fold": 0,
    "outer_fold": 0,
    "trajectory_id": "1",
}
OOF_COLUMNS = (
    *TASK_FIELDS,
    "bank_id",
    "grid_id",
    "sample_role",
    "score",
    "status",
)


def seed_from_name(name: str) -> int:
    """Return the protocol 128-bit seed for a stable task name."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:16], "big")


@dataclass(frozen=True)
class TaskKey:
    """Complete, non-optional identity for one model task."""

    protocol_id: str
    experiment_id: str
    city: str
    label_layer: str
    reveal_mechanism: str
    outer_fold: str
    budget: str
    trajectory: str
    feature_set: str
    model: str
    graph_id: str
    realization_id: str
    config: str
    seed: str

    def __post_init__(self) -> None:
        for field, value in self.as_dict().items():
            if not value or value in {".", ".."} or "/" in value or "\\" in value:
                raise ValueError(f"invalid task key value for {field}")

    def as_dict(self) -> dict[str, str]:
        return {field: str(getattr(self, field)) for field in TASK_FIELDS}

    def relative_path(self) -> Path:
        encoded = json.dumps(self.as_dict(), ensure_ascii=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
        return Path(f"experiment={self.experiment_id}", f"city={self.city}", f"task={digest}")


@dataclass(frozen=True)
class Preprocessor:
    """Fold-specific median imputation and optional neural scaling."""

    columns: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray | None
    scales: np.ndarray | None

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(frame.columns) != self.columns:
            raise ValueError("preprocessor feature columns changed")
        values = frame.to_numpy(dtype=np.float64, copy=True)
        missing = np.isnan(values)
        values[missing] = np.take(self.medians, np.nonzero(missing)[1])
        if self.means is not None and self.scales is not None:
            values = (values - self.means) / self.scales
        return pd.DataFrame(values, index=frame.index, columns=frame.columns)

    def record(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "medians": self.medians.tolist(),
            "means": None if self.means is None else self.means.tolist(),
            "scales": None if self.scales is None else self.scales.tolist(),
        }


def array_record(values: np.ndarray) -> dict[str, object]:
    """Hash a numeric matrix without writing task-level training data."""
    array = np.ascontiguousarray(values)
    header = canonical_json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    digest = hashlib.sha256(header)
    digest.update(array.view(np.uint8))
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest.hexdigest()}


def fit_audit(
    fit_ids: Iterable[str],
    preprocessor: Preprocessor | None,
    **matrices: np.ndarray,
) -> dict[str, object]:
    """Record the exact fitting rows, preprocessing state, and matrix hashes."""
    ordered_ids = list(fit_ids)
    preprocessing = None if preprocessor is None else preprocessor.record()
    return {
        "fit_row_count": len(ordered_ids),
        "fit_rows_sha256": sha256_bytes(canonical_json_bytes(ordered_ids)),
        "matrices": {name: array_record(values) for name, values in sorted(matrices.items())},
        "preprocessing": preprocessing,
        "preprocessing_sha256": (
            None if preprocessing is None else sha256_bytes(canonical_json_bytes(preprocessing))
        ),
    }


class TaskResourceMonitor:
    """Measure absolute process RSS and resettable CUDA allocation peaks per task."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = 0
        self._torch = None

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._done.wait(0.05):
            self._peak_rss_bytes = max(self._peak_rss_bytes, process.memory_info().rss)
        self._peak_rss_bytes = max(self._peak_rss_bytes, process.memory_info().rss)

    def __enter__(self) -> Self:
        self._sample_once()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._torch = torch
        except ImportError:
            pass
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample_once(self) -> None:
        self._peak_rss_bytes = max(self._peak_rss_bytes, psutil.Process().memory_info().rss)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_once()

    def record(self) -> dict[str, int]:
        peak_vram = 0
        if self._torch is not None:
            peak_vram = int(self._torch.cuda.max_memory_allocated())
        return {"peak_rss_bytes": self._peak_rss_bytes, "peak_vram_bytes": peak_vram}


def fit_preprocessor(
    frame: pd.DataFrame,
    fit_ids: Iterable[str],
    *,
    scale: bool,
) -> Preprocessor:
    """Fit preprocessing on the explicitly supplied train P_total union U rows."""
    ordered_ids = list(fit_ids)
    if len(ordered_ids) != len(set(ordered_ids)) or not ordered_ids:
        raise ValueError("preprocessor fit IDs must be non-empty and unique")
    missing_ids = set(ordered_ids).difference(frame.index)
    if missing_ids:
        raise KeyError(f"preprocessor fit IDs are absent: {sorted(missing_ids)[:1]}")
    fit = frame.loc[ordered_ids].to_numpy(dtype=np.float64, copy=True)
    if np.isnan(fit).all(axis=0).any():
        raise ValueError("all-missing training feature column")
    medians = np.nanmedian(fit, axis=0)
    missing = np.isnan(fit)
    fit[missing] = np.take(medians, np.nonzero(missing)[1])
    if scale:
        means = fit.mean(axis=0)
        scales = fit.std(axis=0, ddof=0)
        scales[scales == 0] = 1.0
    else:
        means = None
        scales = None
    return Preprocessor(tuple(frame.columns), medians, means, scales)


def balanced_sample_weights(positive_count: int, unlabeled_count: int) -> np.ndarray:
    """Give the positive and frozen-U groups equal total loss weight."""
    if positive_count <= 0 or unlabeled_count <= 0:
        raise ValueError("both training groups must be non-empty")
    return np.concatenate(
        [
            np.full(positive_count, 0.5 / positive_count, dtype=np.float64),
            np.full(unlabeled_count, 0.5 / unlabeled_count, dtype=np.float64),
        ]
    )


def _complete_config_scores(scores: pd.DataFrame, model: str) -> pd.DataFrame:
    model_scores = scores.loc[scores["model"].eq(model) & scores["status"].eq("success")]
    rows = []
    for config, frame in model_scores.groupby("config", sort=False):
        if set(frame["inner_fold"]) == {0, 1, 2} and not frame["inner_fold"].duplicated().any():
            rows.append({"config": config, "score": float(frame["score"].mean())})
    return pd.DataFrame(rows, columns=["config", "score"])


def best_alternative(scores: pd.DataFrame) -> dict[str, object]:
    """Select the complete inner-CV alternative with the frozen family tie order."""
    required = {"model", "config", "inner_fold", "score", "status"}
    if not required.issubset(scores.columns):
        raise ValueError("inner score table is incomplete")
    candidates = []
    config_order = {"NA": 0, "SMALL": 1, "MEDIUM": 2}
    for model in ALTERNATIVE_ORDER:
        complete = _complete_config_scores(scores, model)
        if complete.empty:
            raise ValueError(f"model family has no complete config: {model}")
        complete["order"] = complete["config"].map(config_order).fillna(99)
        selected = complete.sort_values(
            ["score", "order"], ascending=[False, True], kind="stable"
        ).iloc[0]
        candidates.append(
            {"model": model, "config": str(selected["config"]), "score": selected["score"]}
        )
    maximum = max(float(item["score"]) for item in candidates)
    return next(item for item in candidates if float(item["score"]) == maximum)


def outer_refit_epochs(best_epochs: Iterable[int]) -> int:
    """Use the median of the three selected inner-fold epochs."""
    values = list(best_epochs)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("outer refit requires three positive best epochs")
    return int(statistics.median(values))


def task_action(status: str | None) -> str:
    """Return the only allowed resume action for a recorded task status."""
    if status is None:
        return "run"
    if status == "failed":
        return "retry"
    if status in {"success", "not-estimable"}:
        return "skip"
    raise ValueError(f"invalid task status: {status}")


def write_oof(frame: pd.DataFrame, path: Path) -> Path:
    """Atomically publish one immutable OOF shard with the frozen schema."""
    if tuple(frame.columns) != OOF_COLUMNS:
        raise ValueError("OOF columns differ from the frozen schema")
    if not set(frame["status"]).issubset({"success", "failed", "not-estimable"}):
        raise ValueError("OOF status is invalid")
    successful = frame["status"].eq("success")
    if not np.isfinite(frame.loc[successful, "score"]).all():
        raise ValueError("successful OOF rows require finite scores")
    if not set(frame.loc[successful, "sample_role"]).issubset({"NA", "positive", "background"}):
        raise ValueError("successful OOF rows require a bank sample role")
    full_test = frame.loc[successful & frame["bank_id"].eq("NA")]
    if not full_test["sample_role"].eq("NA").all():
        raise ValueError("full-test OOF rows must use NA sample role")
    key = [*TASK_FIELDS, "bank_id", "grid_id"]
    if frame.loc[successful, key].duplicated().any():
        raise ValueError("OOF task-bank-grid rows must be unique")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    frame.to_parquet(partial, index=False)
    if path.exists():
        if sha256_file(path) != sha256_file(partial):
            partial.unlink(missing_ok=True)
            raise FileExistsError(f"OOF shard already exists with different content: {path}")
        partial.unlink()
    else:
        os.replace(partial, path)
    return path
