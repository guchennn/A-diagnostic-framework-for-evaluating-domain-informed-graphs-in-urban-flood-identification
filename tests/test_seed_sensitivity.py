"""Aggregate-only release regressions; synthetic ranking checks use no city inputs."""

import csv
import importlib.util
import shutil
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
    module = load("scripts/check_seed_results.py")
    result = module.verify(module.DATA)
    assert result["status"] == "passed"
    assert result["fit_group_aggregates"] == 1200
    assert result["reference_rows"] == 8


@pytest.mark.parametrize("mutation", ["value", "missing", "duplicate", "reference"])
def test_seed_verifier_rejects_corruption(tmp_path, mutation):
    module = load("scripts/check_seed_results.py")
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
    module = load("scripts/analyze_seed_sensitivity.py")
    assert len(module.synthetic_tests()) == 6


@pytest.mark.parametrize("archived", [False, True])
def test_raw_seed_cli_archive_options(tmp_path, monkeypatch, archived):
    module = load("scripts/analyze_seed_sensitivity.py")
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
            str(ROOT / "results/settings/refit_seeds.csv"),
            "--structure-archive",
            str(ROOT / "results/structural_stability/contributions_by_fold.csv"),
        ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(module, "logical_run_path", lambda root, run: run)
    monkeypatch.setattr(module, "verify_provenance", lambda *args: ({}, {}))
    monkeypatch.setattr(module, "validate_selections", lambda *args: ([], [], []))
    monkeypatch.setattr(module, "structure_inputs", lambda *args: ([], []))
    monkeypatch.setattr(module, "analyze", lambda *args: args[0])
    assert module.main() == tmp_path / "run"
    assert module.RANDOM_SEED == 20261029
    assert module.RANDOM_DRAWS == 10000
    assert (module.SEED_ARCHIVE is not None) == archived
    assert (module.STRUCTURE_ARCHIVE is not None) == archived
    if archived:
        assert module.SEED_ARCHIVE.is_file()
        assert module.STRUCTURE_ARCHIVE.is_file()
