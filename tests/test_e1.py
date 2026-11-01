from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from floodgnn_ceus import e1
from floodgnn_ceus.experiments import GRAPH_BY_MODEL, MODELS, E1OuterUnit
from floodgnn_ceus.training import TASK_FIELDS, TaskKey


def _task() -> TaskKey:
    values = {field: "NA" for field in TASK_FIELDS}
    values.update(
        protocol_id="base-1",
        experiment_id="E1-inner",
        city="zhengzhou",
        label_layer="STRICT-LOC",
        reveal_mechanism="uniform",
        outer_fold="0",
        budget="20",
        trajectory="1",
        feature_set="FULL-COMMON",
        model="KDE",
        config="NA",
        seed="HPO-1",
    )
    return TaskKey(**values)


def test_inner_task_publishes_three_fold_result_and_resumes(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(e1, "_inner_fold_data", lambda data, task, fold: {"fold": fold})

    def score(fold, task, model_grid):
        calls.append(fold["fold"])
        return 0.25 + fold["fold"], 0, {"fit_rows_sha256": f"fold-{fold['fold']}"}

    monkeypatch.setattr(e1, "_score_inner_fold", score)
    task = _task()

    assert e1.run_inner_task(tmp_path, tmp_path, task, {}, {}) == "success"
    assert e1.run_inner_task(tmp_path, tmp_path, task, {}, {}) == "success"
    assert calls == [0, 1, 2]
    result = pd.read_parquet(e1._status_path(tmp_path, task).with_name("inner_scores.parquet"))
    assert result["inner_fold"].tolist() == [0, 1, 2]
    assert tuple(result.columns) == e1.INNER_RESULT_COLUMNS
    status = json.loads(e1._status_path(tmp_path, task).read_text("utf-8"))
    assert [item["fit_rows_sha256"] for item in status["fit_audits"]] == [
        "fold-0",
        "fold-1",
        "fold-2",
    ]
    assert set(status["resources"]) == {"peak_rss_bytes", "peak_vram_bytes"}


def test_inner_task_records_failure_without_partial_scores(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(e1, "_inner_fold_data", lambda data, task, fold: {})
    monkeypatch.setattr(
        e1,
        "_score_inner_fold",
        lambda fold, task, model_grid: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    task = _task()

    assert e1.run_inner_task(tmp_path, tmp_path, task, {}, {}) == "failed"
    status_path = e1._status_path(tmp_path, task)
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "failed"
    assert not status_path.with_name("inner_scores.parquet").exists()


def test_outer_selection_requires_complete_families_and_uses_frozen_tie_order(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = [
        replace(_task(), model=model, graph_id=GRAPH_BY_MODEL[model], config=config)
        for model in MODELS
        for config in (("NA",) if model == "KDE" else ("SMALL", "MEDIUM"))
    ]
    monkeypatch.setattr(e1, "_unit_tasks", lambda root, unit: tasks)
    for task in tasks:
        status_path = e1._status_path(tmp_path, task)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({"status": "success"}), encoding="utf-8")
        rows = [
            {
                **task.as_dict(),
                "inner_fold": fold,
                "score": 0.5,
                "best_epoch": 10 if task.model in {"MLP", "Fixed-filter", "GraphSAGE-G2"} else 0,
                "status": "success",
            }
            for fold in range(3)
        ]
        pd.DataFrame(rows, columns=e1.INNER_RESULT_COLUMNS).to_parquet(
            status_path.with_name("inner_scores.parquet"), index=False
        )

    selection = e1.select_outer_models(
        tmp_path,
        tmp_path,
        E1OuterUnit("zhengzhou", 0, "20", "1"),
    )

    assert selection["best_alternative"]["model"] == "KDE"
    assert selection["models"]["GraphSAGE-G2"]["config"] == "SMALL"
    assert selection["models"]["GraphSAGE-G2"]["best_epochs"] == [10, 10, 10]


def test_oof_keeps_full_test_once_and_bank_views_separate() -> None:
    task = replace(_task(), experiment_id="E1", seed="NEURAL-1")
    banks = pd.DataFrame(
        {
            "bank_id": [1, 2, 3],
            "grid_id": ["n1", "n1", "n2"],
            "sample_role": ["positive", "positive", "background"],
        }
    )

    frame = e1._oof_rows(
        task,
        {"banks": banks, "test_node_ids": ["n1", "n2"]},
        np.asarray([0.2, 0.8]),
        ["NEURAL-1", "NEURAL-2", "NEURAL-3"],
    )

    full = frame.loc[frame["bank_id"].eq("NA")]
    assert len(full) == 6
    assert not full.duplicated(["seed", "grid_id"]).any()
    assert full["sample_role"].eq("NA").all()
    assert len(frame) == 15
