import json
from types import SimpleNamespace

import pytest

from floodgnn_ceus import e0, graph_support_audit


@pytest.fixture
def provenance_root(tmp_path):
    for name in (
        "pyproject.toml",
        "requirements.txt",
        "requirements-train.txt",
        "artifacts/protocol_lock.json",
        "src/floodgnn_ceus/graph_support_audit.py",
        "src/floodgnn_ceus/graphs.py",
        "src/floodgnn_ceus/evaluation_design.py",
        "artifacts/reports/evaluation_design.json",
        "artifacts/reports/data_feasibility_report.md",
        "artifacts/reports/graph_counterfactuals.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/protocol.yaml").write_text("cities: [synthetic]\n")
    (tmp_path / "docs").mkdir()
    return tmp_path


@pytest.mark.parametrize("relative", ["requirements.txt", "requirements-train.txt"])
def test_e0_dependency_change_invalidates_code_bundle(provenance_root, monkeypatch, relative):
    monkeypatch.setattr(e0, "load_protocol", lambda _: SimpleNamespace(sources={}))
    monkeypatch.setattr(e0, "_read_yaml", lambda _: {})
    monkeypatch.setattr(e0, "validate_label_schema_contract", lambda _: set())
    monkeypatch.setattr(e0, "_raw_records", lambda *_: [])
    monkeypatch.setattr(e0, "CITY_REPORTS", ())
    monkeypatch.setattr(e0, "EXPECTED_CITIES", ())
    first, _ = e0._artifact_records(provenance_root)
    first_code = [row for row in first if row["category"] == "code"]
    assert {"requirements.txt", "requirements-train.txt"} <= {r["path"] for r in first_code}
    (provenance_root / relative).write_text("changed fixture\n")
    second, _ = e0._artifact_records(provenance_root)
    second_code = [row for row in second if row["category"] == "code"]
    assert e0._bundle(first_code) != e0._bundle(second_code)


@pytest.mark.parametrize("relative", ["requirements.txt", "requirements-train.txt"])
def test_graph_audit_tracks_dependencies_and_existing_entrypoint(
    provenance_root, monkeypatch, relative
):
    summary = {
        "city": "synthetic",
        "graph_scope": "whole_domain",
        "sample_group": "all",
        "outer_fold": "ALL",
        "node_count": 1,
        "upstream_1hop_zero_fraction": 1,
        "upstream_1hop_mean": 0,
        "upstream_2hop_mean": 0,
        "upstream_2hop_median": 0,
        "upstream_2hop_p95": 0,
    }
    monkeypatch.setattr(
        graph_support_audit,
        "audit_city",
        lambda *_: ([summary], [], [{"outer_fold": "POOLED"}], []),
    )
    graph_support_audit.run_audit(provenance_root)
    manifest_path = provenance_root / "results/graph_support/manifest.json"
    first = json.loads(manifest_path.read_text())["inputs"]
    first_hashes = {row["path"]: row["sha256"] for row in first}
    assert {"requirements.txt", "requirements-train.txt"} <= first_hashes.keys()
    (provenance_root / relative).write_text("changed fixture\n")
    graph_support_audit.run_audit(provenance_root)
    second = json.loads(manifest_path.read_text())["inputs"]
    second_hashes = {row["path"]: row["sha256"] for row in second}
    assert first_hashes[relative] != second_hashes[relative]
    report = (provenance_root / "docs/graph_support_audit.md").read_text(encoding="utf-8")
    assert "python -m floodgnn_ceus.graph_support_audit --root ." in report
    assert "scripts/audit_graph_support.py" not in report
    assert "uv run" not in report
