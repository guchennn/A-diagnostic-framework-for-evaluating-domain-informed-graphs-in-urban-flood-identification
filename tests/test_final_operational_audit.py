"""Published audit entry points remain portable and preserve numerical invariants."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reproduction/final_audit"


def load_test_module(relative: str, name: str):
    path = PACKAGE / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MigrationTests = load_test_module(
    "group_migration/test_migration.py", "migration_unit"
).MigrationTests
LocalReferenceTests = load_test_module(
    "local_references/test_local_references.py", "local_reference_unit"
).LocalReferenceTests


@pytest.mark.parametrize(
    "script", ["group_migration/audit_migration.py", "local_references/analyze_local_references.py"]
)
def test_missing_inputs_explained_without_input_root_imports(tmp_path, script):
    command = [
        sys.executable,
        str(PACKAGE / script),
        "--project-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    if script.startswith("local_references"):
        command += ["--run-root", "artifacts/runs/reviewer-revision-v1-missing"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "Original" in result.stderr and "required" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not (tmp_path / "output").exists()
