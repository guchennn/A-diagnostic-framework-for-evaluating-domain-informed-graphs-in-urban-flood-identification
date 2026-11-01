"""Performance-blind source audits for Phase 1."""

from __future__ import annotations

import csv
import hashlib
import io
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

from floodgnn_ceus.io import canonical_json_bytes, sha256_file

CROWD_FLOOD_FIELDS = (
    "id",
    "wgs_lng",
    "wgs_lat",
    "geometry",
    "pt_type_1",
    "toponym",
    "SSA",
    "similarity",
    "flood_prob",
)


def _event_member(city_event: str) -> str:
    filename = city_event.replace(" ", "_") + ".csv"
    return f"CrowdFlood dataset/csv/{filename}"


def _validated_files(archive: ZipFile) -> list[ZipInfo]:
    files: list[ZipInfo] = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
            raise ValueError(f"unsafe archive member: {info.filename}")
        file_type = (info.external_attr >> 16) & 0o170000
        if file_type == 0o120000:
            raise ValueError(f"archive symlink is forbidden: {info.filename}")
        if not info.is_dir():
            files.append(info)
    return files


def _inspect_csv(data: bytes, member: str) -> tuple[list[str], int]:
    try:
        rows = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
        header = next(rows)
    except (StopIteration, UnicodeDecodeError) as error:
        raise ValueError(f"invalid UTF-8 CSV: {member}") from error
    if tuple(header) != CROWD_FLOOD_FIELDS:
        raise ValueError(f"unexpected CrowdFlood fields: {member}")
    row_count = 0
    for row_count, row in enumerate(rows, start=1):
        if len(row) != len(CROWD_FLOOD_FIELDS):
            raise ValueError(f"invalid CrowdFlood row width at {member}:{row_count + 1}")
    return header, row_count


def inspect_crowdflood_archive(
    archive_path: Path,
    city_events: Mapping[str, str],
) -> dict[str, object]:
    """Audit archive structure and selected event schemas without exposing row values."""
    with ZipFile(archive_path) as archive:
        files = _validated_files(archive)
        names = {info.filename for info in files}
        selected_events: list[dict[str, object]] = []
        for city_id, city_event in city_events.items():
            member = _event_member(city_event)
            if member not in names:
                raise ValueError(f"missing CrowdFlood event: {member}")
            data = archive.read(member)
            fields, row_count = _inspect_csv(data, member)
            selected_events.append(
                {
                    "bytes": len(data),
                    "city_id": city_id,
                    "fields": fields,
                    "row_count": row_count,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "source_member": member,
                }
            )
    suffix_counts = Counter(PurePosixPath(info.filename).suffix.lower() for info in files)
    return {
        "archive_sha256": sha256_file(archive_path),
        "file_count": len(files),
        "selected_events": selected_events,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "uncompressed_bytes": sum(info.file_size for info in files),
    }


def extract_crowdflood_events(
    archive_path: Path,
    destination: Path,
    city_events: Mapping[str, str],
) -> list[Path]:
    """Extract only fixed study events, preserving their source bytes."""
    inspect_crowdflood_archive(archive_path, city_events)
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with ZipFile(archive_path) as archive:
        for city_event in city_events.values():
            member = _event_member(city_event)
            data = archive.read(member)
            output = destination / PurePosixPath(member).name
            if output.exists():
                if output.read_bytes() != data:
                    raise FileExistsError(f"existing extracted file differs: {output}")
            else:
                partial = output.with_name(output.name + ".part")
                partial.write_bytes(data)
                os.replace(partial, output)
            extracted.append(output)
    return extracted


def write_crowdflood_audit(
    destination: Path,
    audit: Mapping[str, object],
    *,
    record_id: int,
    record_doi: str,
    license_id: str,
    privacy_status: str,
) -> Path:
    """Write a compact public audit without source-row values."""
    if privacy_status not in {"approved", "processing-only", "prohibited"}:
        raise ValueError(f"invalid privacy status: {privacy_status}")
    payload = {
        "audit": dict(audit),
        "license": license_id,
        "privacy_status": privacy_status,
        "raw_values_committed": False,
        "record_doi": record_doi,
        "record_id": record_id,
        "schema_version": 1,
        "source_id": "crowdflood_v1",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(canonical_json_bytes(payload))
    os.replace(partial, destination)
    return destination
