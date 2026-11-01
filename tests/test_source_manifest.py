from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import floodgnn_ceus.sources as sources_module
from floodgnn_ceus.protocol import load_protocol
from floodgnn_ceus.sources import IntegrityError, SourceRecord
from floodgnn_ceus.sources import build_raw_source_manifest as _build_manifest
from floodgnn_ceus.sources import write_source_receipt as _write_receipt

EXPECTED_COLUMNS = [
    "schema_version",
    "source_id",
    "city_id",
    "asset_key",
    "provider",
    "version_or_release",
    "source_url_or_doi",
    "license",
    "privacy_status",
    "reference_date_or_period",
    "relative_path",
    "retrieved_at",
    "bytes",
    "sha256",
    "upstream_checksum_algorithm",
    "upstream_checksum",
    "request_sha256",
    "api_version",
    "replication_sequence",
]
SENSITIVE_FIELDS = {"raw_id", "text", "toponym", "cookie", "token", "credential"}
PROTOCOL = load_protocol(Path(__file__).parents[1] / "configs")
PUBLIC_LOCATOR = PROTOCOL.sources["geoboundaries_chn_adm2"]["url"]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _asset(raw_root: Path, relative_path: str, content: bytes = b"payload") -> Path:
    path = raw_root.joinpath(*relative_path.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _record(relative_path: str, content: bytes = b"payload", **changes: object) -> SourceRecord:
    values: dict[str, object] = {
        "source_id": "geoboundaries_chn_adm2",
        "city_id": "NA",
        "asset_key": "chn-adm2",
        "provider": "geoBoundaries",
        "version_or_release": "9469f09",
        "source_url_or_doi": PUBLIC_LOCATOR,
        "license": "PDDL-1.0",
        "privacy_status": "approved",
        "reference_date_or_period": "2026-08-07",
        "relative_path": relative_path,
        "retrieved_at": "2026-08-08T08:30:00Z",
        "bytes": len(content),
        "sha256": _sha256(content),
        "upstream_checksum_algorithm": "sha256",
        "upstream_checksum": _sha256(content),
        "request_sha256": _sha256(b"public request"),
    }
    values.update(changes)
    return SourceRecord(**values)  # type: ignore[arg-type]


def _write(record: SourceRecord, raw_root: Path) -> Path:
    return _write_receipt(record, raw_root, PROTOCOL)


def _build(raw_root: Path, artifacts_root: Path) -> Path:
    return _build_manifest(raw_root, artifacts_root, PROTOCOL)


def test_receipt_and_manifest_use_the_fixed_secret_free_schema(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    relative_path = "geoboundaries/chn-adm2.geojson"
    asset = _asset(raw_root, relative_path)

    receipt = _write(_record(relative_path), raw_root)
    manifest = _build(raw_root, tmp_path / "artifacts")

    assert receipt == asset.with_name(f"{asset.name}.source.json")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert set(payload) == set(EXPECTED_COLUMNS)
    assert not SENSITIVE_FIELDS.intersection(payload)
    table = pq.read_table(manifest)
    assert table.column_names == EXPECTED_COLUMNS
    assert table.to_pylist() == [payload]


def test_receipt_is_idempotent_and_never_overwrites_a_conflict(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    relative_path = "source/asset.bin"
    _asset(raw_root, relative_path)
    record = _record(relative_path)
    receipt = _write(record, raw_root)
    original = receipt.read_bytes()

    assert _write(record, raw_root) == receipt
    with pytest.raises(IntegrityError, match="conflict"):
        _write(replace(record, provider="changed"), raw_root)

    assert receipt.read_bytes() == original


@pytest.mark.parametrize("relative_path", ["../secret.bin", "/absolute.bin", "a/../b.bin"])
def test_receipt_rejects_paths_outside_raw(tmp_path: Path, relative_path: str) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    with pytest.raises(ValueError, match="validation"):
        _write(_record(relative_path), raw_root)


@pytest.mark.parametrize("privacy_status", ["unknown", "", "public"])
def test_receipt_rejects_unfrozen_privacy_states(
    tmp_path: Path,
    privacy_status: str,
) -> None:
    raw_root = tmp_path / "raw"
    relative_path = "source/asset.bin"
    _asset(raw_root, relative_path)

    with pytest.raises(ValueError, match="validation"):
        _write(_record(relative_path, privacy_status=privacy_status), raw_root)


def test_receipt_recomputes_size_and_sha256(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    relative_path = "source/asset.bin"
    _asset(raw_root, relative_path, b"different")

    with pytest.raises(IntegrityError, match="validation"):
        _write(_record(relative_path), raw_root)


def test_receipt_rejects_a_locator_not_frozen_in_sources_yaml(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    relative_path = "source/asset.bin"
    _asset(raw_root, relative_path)

    with pytest.raises(ValueError, match="validation"):
        _write(_record(relative_path, source_url_or_doi="https://example.com/file"), raw_root)


def test_manifest_is_deterministic_and_does_not_replace_existing_bytes(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    relative_path = "source/asset.bin"
    _asset(raw_root, relative_path)
    _write(_record(relative_path), raw_root)
    artifacts = tmp_path / "artifacts"

    first = _build(raw_root, artifacts)
    first_bytes = first.read_bytes()
    second = _build(raw_root, artifacts)

    assert second == first
    assert second.read_bytes() == first_bytes


def test_manifest_revalidates_assets_immediately_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    relative_path = "source/asset.bin"
    asset = _asset(raw_root, relative_path)
    _write(_record(relative_path), raw_root)
    original_write = sources_module.pq.write_table

    def mutate_after_write(*args: object, **kwargs: object) -> None:
        original_write(*args, **kwargs)
        asset.write_bytes(b"changed after scan")

    monkeypatch.setattr(sources_module.pq, "write_table", mutate_after_write)

    with pytest.raises(IntegrityError, match="validation"):
        _build(raw_root, tmp_path / "artifacts")

    assert not (tmp_path / "artifacts/manifests/raw_source_manifest.parquet").exists()
