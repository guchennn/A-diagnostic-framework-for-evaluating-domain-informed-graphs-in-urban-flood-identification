"""Publication imports and numerical tests require no private artifacts."""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "reproduction/controlled_validation"


def test_controlled_metric_invariants():
    subprocess.run(
        [sys.executable, str(ROOT / "controlled_ranking/test_validation.py")], check=True
    )


def test_archive_import_and_weighted_ties(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "archive_analysis"))
    spec = importlib.util.spec_from_file_location(
        "published_archive", ROOT / "archive_analysis/reanalyze_archive.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify_formula()
    assert result["weight_cases"] == 9
    assert result["max_weighted_ap_error"] < 1e-12


def test_supplied_aggregate_accounting():
    subprocess.run([sys.executable, "-I", "-S", str(ROOT / "verify_published.py")], check=True)
