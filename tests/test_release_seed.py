"""Aggregate-only release regressions; synthetic ranking checks use no city inputs."""

import csv
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_aggregate_verification():
    module = load("analysis/seed/verify_aggregate_arithmetic.py")
    result = module.verify(module.DATA)
    assert result["status"] == "passed"
    assert result["fit_group_aggregates"] == 1200
    assert result["reference_rows"] == 8


@pytest.mark.parametrize("mutation", ["value", "missing", "duplicate", "reference"])
def test_seed_verifier_rejects_corruption(tmp_path, mutation):
    module = load("analysis/seed/verify_aggregate_arithmetic.py")
    shutil.copytree(module.DATA, tmp_path / "data")
    name = "observed_vs_seed_references.csv" if mutation == "reference" else "seed_pair_city.csv"
    path = tmp_path / "data" / name
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["observed_graph_effect_pp" if mutation == "reference" else "effect_pp"] = "100"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        module.verify(tmp_path / "data")


def test_raw_seed_synthetic_invariants():
    module = load("analysis/seed/analyze_seed_only.py")
    assert len(module.synthetic_tests()) == 6


def test_figure_regeneration_from_another_working_directory(tmp_path):
    output = tmp_path / "figures"
    subprocess.run(
        [sys.executable, str(ROOT / "plots/make_result_figures.py"), "--output", str(output)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    for name in ["fig3_matched_increment", "fig4_structural_gains_losses"]:
        for extension in ["png", "pdf", "svg"]:
            assert (output / f"{name}.{extension}").stat().st_size > 1000
    assert (output / "results_point_verification.json").is_file()


def test_study_area_raster_alignment():
    import numpy as np
    import pandas as pd

    module = load("plots/make_study_area.py")
    frame = pd.DataFrame(
        {
            "row": [3, 3, 4, 4],
            "column": [7, 8, 7, 8],
            "center_x": [100, 300, 100, 300],
            "center_y": [300, 300, 100, 100],
            "value": [1, 2, 3, 4],
        }
    )
    raster, extent = module.raster(frame, "value")
    np.testing.assert_array_equal(raster, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(extent, [0, 400, 0, 400])
    frame.loc[0, "center_x"] += 1
    with pytest.raises(ValueError, match="Study-area input validation failed"):
        module.raster(frame, "value")


def test_portable_seed_plan_digest():
    import hashlib
    import json

    plan = ROOT / "analysis/seed/analysis_plan.json"
    assert (
        hashlib.sha256(plan.read_bytes()).hexdigest()
        == plan.with_suffix(".sha256").read_text().split()[0]
    )
    contract = json.loads(plan.read_text())
    assert contract["randomized_reference"]["seed"] == 20261029
    assert contract["randomized_reference"]["draws"] == 10000
    assert len(contract["portable_release_provenance"]["source_sha256"]) == 64


@pytest.mark.parametrize("archived", [False, True])
def test_raw_seed_cli_archive_options(tmp_path, monkeypatch, archived):
    module = load("analysis/seed/analyze_seed_only.py")
    args = [
        "analyze_seed_only.py",
        "--root",
        str(tmp_path),
        "--run-root",
        "run",
        "--output",
        str(tmp_path / "output"),
    ]
    if archived:
        args += [
            "--seed-archive",
            str(ROOT / "provenance/primary_refit_seeds.csv"),
            "--structure-archive",
            str(ROOT / "plots/data/fig4_structure_folds.csv"),
        ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(module, "logical_run_path", lambda root, run: run)
    monkeypatch.setattr(module, "verify_provenance", lambda *args: ({}, {}))
    monkeypatch.setattr(module, "validate_selections", lambda *args: ([], [], []))
    monkeypatch.setattr(module, "structure_inputs", lambda *args: ([], []))
    monkeypatch.setattr(module, "analyze", lambda *args: args[0])
    assert module.main() == tmp_path / "run"
    assert module.PLAN == ROOT / "analysis/seed/analysis_plan.json"
    assert (module.SEED_ARCHIVE is not None) == archived
    assert (module.STRUCTURE_ARCHIVE is not None) == archived
    if archived:
        assert module.SEED_ARCHIVE.is_file()
        assert module.STRUCTURE_ARCHIVE.is_file()


def test_seed_verifier_fresh_run_without_archive(tmp_path):
    module = load("analysis/seed/verify_aggregate_arithmetic.py")
    shutil.copytree(module.DATA, tmp_path / "data")
    (tmp_path / "data/archived_graph_validation.csv").unlink()
    result = module.verify(tmp_path / "data", check_archive=False)
    assert result["status"] == "passed"
    assert result["archived_graph_folds"] == 0
    with pytest.raises(FileNotFoundError):
        module.verify(tmp_path / "data")
