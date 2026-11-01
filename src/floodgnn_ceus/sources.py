"""Fail-closed HTTP downloads with resumable partial files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from ipaddress import IPv6Address
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO
from urllib.parse import urlsplit, urlunsplit

import idna
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes
from floodgnn_ceus.protocol import EXPECTED_CONFIG_SHA256, LoadedProtocol

if os.name == "nt":
    import msvcrt
else:
    import fcntl

DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_EXTRA_BYTES = 20 * 1024**3
MAX_SOURCE_RECEIPT_BYTES = 64 * 1024
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_UNSATISFIED_RANGE = re.compile(r"bytes \*/(\d+)")
_LOWER_HEX_32 = re.compile(r"[0-9a-f]{32}")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")
_DOI = re.compile(r"10\.\d{4,9}/[A-Za-z0-9._;()/:+\-]+")
_UTC_SECONDS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_INVALID_PORTABLE_PATH_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "clock$"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)
RAW_SOURCE_COLUMNS = (
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
)
RAW_SOURCE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("city_id", pa.string(), nullable=False),
        pa.field("asset_key", pa.string(), nullable=False),
        pa.field("provider", pa.string(), nullable=False),
        pa.field("version_or_release", pa.string(), nullable=False),
        pa.field("source_url_or_doi", pa.string(), nullable=False),
        pa.field("license", pa.string(), nullable=False),
        pa.field("privacy_status", pa.string(), nullable=False),
        pa.field("reference_date_or_period", pa.string(), nullable=False),
        pa.field("relative_path", pa.string(), nullable=False),
        pa.field("retrieved_at", pa.string(), nullable=False),
        pa.field("bytes", pa.int64(), nullable=False),
        pa.field("sha256", pa.string(), nullable=False),
        pa.field("upstream_checksum_algorithm", pa.string()),
        pa.field("upstream_checksum", pa.string()),
        pa.field("request_sha256", pa.string()),
        pa.field("api_version", pa.string()),
        pa.field("replication_sequence", pa.string()),
    ]
)


class DownloadError(RuntimeError):
    """Raised when an HTTP transfer cannot complete."""


class IntegrityError(DownloadError):
    """Raised when downloaded or existing bytes fail validation."""


class InsufficientSpaceError(DownloadError):
    """Raised when the target filesystem lacks protocol-required space."""


@dataclass(frozen=True)
class SourceRecord:
    """Secret-free provenance for one acquired raw asset."""

    source_id: str
    city_id: str
    asset_key: str
    provider: str
    version_or_release: str
    source_url_or_doi: str
    license: str
    privacy_status: str
    reference_date_or_period: str
    relative_path: str
    retrieved_at: str
    bytes: int
    sha256: str
    upstream_checksum_algorithm: str | None = None
    upstream_checksum: str | None = None
    request_sha256: str | None = None
    api_version: str | None = None
    replication_sequence: str | None = None
    schema_version: int = 1


@dataclass(frozen=True)
class DownloadReceipt:
    path: Path
    bytes_written: int
    sha256: str
    resumed: bool
    etag: str | None


@dataclass(frozen=True)
class _VerifiedFile:
    size: int
    sha256: str


def _non_negative_integer(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def required_free_bytes(
    planned_bytes: int,
    multiplier: int = 2,
    extra_bytes: int = DEFAULT_EXTRA_BYTES,
) -> int:
    """Return the exact protocol reserve for a planned download."""
    planned = _non_negative_integer("planned_bytes", planned_bytes)
    factor = _non_negative_integer("multiplier", multiplier)
    extra = _non_negative_integer("extra_bytes", extra_bytes)
    return factor * planned + extra


def ensure_free_space(
    root: str | PathLike[str],
    planned_bytes: int,
    multiplier: int = 2,
    extra_bytes: int = DEFAULT_EXTRA_BYTES,
) -> None:
    """Create a download root and require the protocol disk reserve."""
    required = required_free_bytes(planned_bytes, multiplier, extra_bytes)
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path).free < required:
        raise InsufficientSpaceError(f"download root requires {required} free bytes")


def _safe_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return urlunsplit((parsed.scheme, f"{host}{port}", "/<redacted>", "", ""))
    except ValueError:
        return "<redacted-url>"


def _part_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part")


def _sidecar_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part.meta.json")


def _journal_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.verified.json")


def _lock_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part.lock")


def _plain_root(value: str | PathLike[str], *, create: bool) -> Path:
    path = Path(value)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path.resolve(strict=True)


def _atomic_stream_write(
    path: Path,
    write: Callable[[BinaryIO], None],
    before_publish: Callable[[], None] | None = None,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if before_publish is not None:
            before_publish()
        try:
            os.link(temporary, path)
        except FileExistsError:
            if temporary.read_bytes() != path.read_bytes():
                raise IntegrityError("existing artifact differs from rebuilt content") from None
        _best_effort_sync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _destination_lock(path: Path) -> Iterator[None]:
    handle = path.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise DownloadError("download already in progress for destination") from None
    try:
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _origin_sha256(url: str) -> str:
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.casefold()
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{IPv6Address(host).compressed}]"
        else:
            host = idna.encode(host, uts46=True).decode("ascii").casefold()
        port = parsed.port
        if port is None:
            port = {"http": 80, "https": 443}.get(scheme)
        port_text = f":{port}" if port is not None else ""
        origin = urlunsplit((scheme, f"{host}{port_text}", "/", "", ""))
    except (ValueError, idna.IDNAError, UnicodeError):
        raise DownloadError("invalid download origin") from None
    return hashlib.sha256(origin.encode("utf-8")).hexdigest()


def _request_asset_identity_sha256(
    asset_identity: str | None,
    expected_size: int | None,
    checksums: Mapping[str, str],
) -> str:
    if asset_identity is not None:
        if not isinstance(asset_identity, str):
            raise TypeError("asset_identity must be a string")
        if not asset_identity:
            raise DownloadError("asset_identity must be non-empty")
        return hashlib.sha256(asset_identity.encode("utf-8")).hexdigest()
    expected_sha256 = checksums.get("sha256")
    if (
        expected_size is not None
        and expected_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None
    ):
        evidence = f"content-sha256:{expected_size}:{expected_sha256}"
        return hashlib.sha256(evidence.encode("utf-8")).hexdigest()
    raise DownloadError("request requires asset_identity or expected size and sha256")


def _strong_etag(value: str | None) -> bool:
    if value is None or len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return False
    return all(
        code == 0x21 or 0x23 <= code <= 0x7E or 0x80 <= code <= 0xFF
        for code in map(ord, value[1:-1])
    )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    for key, value in headers.items():
        if key.casefold() == wanted:
            return value
    return None


def _response_length(headers: Mapping[str, str]) -> int | None:
    value = _header(headers, "Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as error:
        raise IntegrityError("invalid Content-Length") from error
    if length < 0:
        raise IntegrityError("invalid Content-Length")
    return length


def _normalize_checksums(expected: Mapping[str, str] | None) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for name, value in (expected or {}).items():
        algorithm = name.casefold()
        if algorithm not in {"md5", "sha256"}:
            raise ValueError(f"unsupported checksum: {name}")
        if not isinstance(value, str):
            raise TypeError("checksum values must be strings")
        checksums[algorithm] = value.casefold()
    return checksums


def _validate_file(
    path: Path,
    expected_size: int | None,
    checksums: Mapping[str, str],
) -> _VerifiedFile:
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise IntegrityError("file size failed integrity validation")
    digests = {"sha256": hashlib.sha256()}
    if "md5" in checksums:
        digests["md5"] = hashlib.new("md5")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(DEFAULT_CHUNK_SIZE), b""):
            for digest in digests.values():
                digest.update(chunk)
    observed = {name: digest.hexdigest() for name, digest in digests.items()}
    for algorithm, expected in checksums.items():
        if observed[algorithm] != expected:
            raise IntegrityError(f"{algorithm} failed integrity validation")
    return _VerifiedFile(size, observed["sha256"])


def _load_sidecar(
    path: Path,
    url: str,
    asset_identity_sha256: str,
    expected_size: int | None,
) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or set(value) != {
        "asset_identity_sha256",
        "etag",
        "expected_total",
        "origin_sha256",
    }:
        return None
    etag = value["etag"]
    total = value["expected_total"]
    origin_sha256 = value["origin_sha256"]
    stored_asset_identity_sha256 = value["asset_identity_sha256"]
    if not isinstance(etag, str) or not _strong_etag(etag):
        return None
    if type(total) is not int or total < 0:
        return None
    if (
        not isinstance(origin_sha256, str)
        or origin_sha256 != _origin_sha256(url)
        or not isinstance(stored_asset_identity_sha256, str)
        or stored_asset_identity_sha256 != asset_identity_sha256
    ):
        return None
    if expected_size is not None and total != expected_size:
        return None
    return value


def _best_effort_sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _required_text(value: object) -> bool:
    return (
        isinstance(value, str) and bool(value) and all(ord(character) >= 32 for character in value)
    )


def _canonical_relative_path(value: object) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError("source record path is invalid")
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or path.as_posix() != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ValueError("source record path is invalid")
    for part in path.parts:
        reserved_stem = part.split(".", 1)[0].casefold()
        if (
            part.endswith((".", " "))
            or reserved_stem in _WINDOWS_RESERVED_NAMES
            or any(
                ord(character) < 32 or character in _INVALID_PORTABLE_PATH_CHARACTERS
                for character in part
            )
        ):
            raise ValueError("source record path is invalid")
    return path


def _public_locator(value: object) -> bool:
    if not _required_text(value) or any(character.isspace() for character in value):
        return False
    assert isinstance(value, str)
    if _DOI.fullmatch(value) is not None:
        return True
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _frozen_source_references(protocol: LoadedProtocol) -> dict[str, frozenset[str]]:
    if not isinstance(protocol, LoadedProtocol):
        raise TypeError("source references must come from the loaded protocol")
    inputs = {
        "protocol": protocol.protocol,
        "sources": protocol.sources,
        "cities": protocol.cities,
    }
    if (
        protocol.input_sha256 != EXPECTED_CONFIG_SHA256
        or sha256_bytes(canonical_json_bytes(inputs)) != protocol.input_sha256
    ):
        raise ValueError("source references are not frozen")
    references: dict[str, frozenset[str]] = {}
    for source_id, source in protocol.sources.items():
        public: set[str] = set()
        pending: list[object] = [source]
        while pending:
            value = pending.pop()
            if isinstance(value, Mapping):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, str) and _public_locator(value):
                public.add(value)
        if not public:
            raise ValueError("source references are not frozen")
        references[source_id] = frozenset(public)
    return references


def _record_payload(
    record: SourceRecord,
    source_references: Mapping[str, frozenset[str]],
) -> dict[str, object]:
    if type(record.schema_version) is not int or record.schema_version != 1:
        raise ValueError("source record failed validation")
    required_text = (
        record.source_id,
        record.city_id,
        record.asset_key,
        record.provider,
        record.version_or_release,
        record.license,
        record.reference_date_or_period,
    )
    if not all(_required_text(value) for value in required_text):
        raise ValueError("source record failed validation")
    if record.privacy_status not in {"approved", "processing-only", "prohibited"}:
        raise ValueError("source record failed validation")
    allowed = source_references.get(record.source_id)
    if (
        not isinstance(allowed, frozenset)
        or record.source_url_or_doi not in allowed
        or not _public_locator(record.source_url_or_doi)
    ):
        raise ValueError("source record locator is invalid")
    _canonical_relative_path(record.relative_path)
    if (
        not isinstance(record.retrieved_at, str)
        or _UTC_SECONDS.fullmatch(record.retrieved_at) is None
    ):
        raise ValueError("source record failed validation")
    try:
        datetime.fromisoformat(record.retrieved_at)
    except (TypeError, ValueError):
        raise ValueError("source record failed validation") from None
    if type(record.bytes) is not int or record.bytes < 0:
        raise ValueError("source record failed validation")
    if not isinstance(record.sha256, str) or _LOWER_HEX_64.fullmatch(record.sha256) is None:
        raise ValueError("source record failed validation")
    algorithm = record.upstream_checksum_algorithm
    checksum = record.upstream_checksum
    if (algorithm is None) != (checksum is None):
        raise ValueError("source record failed validation")
    if algorithm is not None:
        expected_pattern = {"md5": _LOWER_HEX_32, "sha256": _LOWER_HEX_64}.get(algorithm)
        if expected_pattern is None or not isinstance(checksum, str):
            raise ValueError("source record failed validation")
        if expected_pattern.fullmatch(checksum) is None:
            raise ValueError("source record failed validation")
    if record.request_sha256 is not None and (
        not isinstance(record.request_sha256, str)
        or _LOWER_HEX_64.fullmatch(record.request_sha256) is None
    ):
        raise ValueError("source record failed validation")
    if not all(
        value is None or _required_text(value)
        for value in (record.api_version, record.replication_sequence)
    ):
        raise ValueError("source record failed validation")
    return {name: getattr(record, name) for name in RAW_SOURCE_COLUMNS}


def _asset_path(raw_root: Path, record: SourceRecord) -> Path:
    relative = _canonical_relative_path(record.relative_path)
    root = raw_root.resolve(strict=True)
    try:
        asset = root.joinpath(*relative.parts).resolve(strict=True)
        if not asset.is_relative_to(root) or not asset.is_file():
            raise OSError
        _validate_file(asset, record.bytes, {"sha256": record.sha256})
    except (OSError, IntegrityError):
        raise IntegrityError("source record failed validation") from None
    return asset


def _atomic_write_bytes(path: Path, payload: bytes, root: Path) -> None:
    try:
        if not path.parent.resolve(strict=True).is_relative_to(root):
            raise OSError
        _atomic_stream_write(path, lambda stream: stream.write(payload))
    except OSError:
        raise IntegrityError("source receipt write failed") from None


def _read_regular_bytes(path: Path, root: Path) -> bytes:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise OSError
    if resolved.stat().st_size > MAX_SOURCE_RECEIPT_BYTES:
        raise OSError
    return resolved.read_bytes()


def _snapshot_lock_path(raw_root: Path) -> Path:
    return raw_root / ".source-snapshot.lock"


def write_source_receipt(
    record: SourceRecord,
    raw_root: str | PathLike[str],
    protocol: LoadedProtocol,
) -> Path:
    """Write immutable provenance beside a validated raw asset."""
    try:
        source_references = _frozen_source_references(protocol)
        payload = canonical_json_bytes(_record_payload(record, source_references))
        root = _plain_root(raw_root, create=False)
        if not root.is_dir():
            raise ValueError
    except IntegrityError:
        raise
    except (OSError, TypeError, ValueError):
        raise ValueError("source record failed validation") from None
    with _destination_lock(_snapshot_lock_path(root)):
        asset = _asset_path(root, record)
        receipt = asset.with_name(f"{asset.name}.source.json")
        try:
            existing = _read_regular_bytes(receipt, root)
        except FileNotFoundError:
            existing = None
        except OSError:
            raise IntegrityError("source receipt validation failed") from None
        if existing is not None:
            if existing != payload:
                raise IntegrityError("source receipt conflict")
            return receipt
        _atomic_write_bytes(receipt, payload, root)
    return receipt


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _load_source_receipt(
    path: Path,
    raw_root: Path,
    source_references: Mapping[str, frozenset[str]],
) -> tuple[SourceRecord, dict[str, object]]:
    try:
        value = json.loads(
            _read_regular_bytes(path, raw_root).decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
        if not isinstance(value, dict) or set(value) != set(RAW_SOURCE_COLUMNS):
            raise ValueError
        record = SourceRecord(**value)
        payload = _record_payload(record, source_references)
        asset = _asset_path(raw_root, record)
        expected_receipt = asset.with_name(f"{asset.name}.source.json")
        if path.resolve(strict=True) != expected_receipt:
            raise ValueError
    except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError, IntegrityError):
        raise IntegrityError("source receipt failed validation") from None
    return record, payload


def _write_manifest(
    path: Path,
    rows: list[dict[str, object]],
    before_publish: Callable[[], None],
) -> None:
    try:
        table = pa.Table.from_pylist(rows, schema=RAW_SOURCE_SCHEMA)

        def write(stream: BinaryIO) -> None:
            pq.write_table(
                table,
                stream,
                compression="zstd",
                use_dictionary=False,
                write_statistics=False,
            )

        _atomic_stream_write(path, write, before_publish)
    except (OSError, pa.ArrowException):
        raise IntegrityError("source manifest write failed") from None


def _plain_output_directory(path: Path, artifacts_root: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(artifacts_root):
            raise OSError
    except OSError:
        raise IntegrityError("source manifest output directory is invalid") from None
    return resolved


def build_raw_source_manifest(
    raw_root: str | PathLike[str],
    artifacts_root: str | PathLike[str],
    protocol: LoadedProtocol,
) -> Path:
    """Rebuild a deterministic Parquet manifest from validated raw receipts."""
    try:
        root = _plain_root(raw_root, create=False)
    except OSError:
        raise IntegrityError("no source receipts found") from None
    if not root.is_dir():
        raise IntegrityError("no source receipts found")
    try:
        source_references = _frozen_source_references(protocol)
    except (TypeError, ValueError):
        raise ValueError("source references are not frozen") from None
    try:
        artifacts = _plain_root(artifacts_root, create=True)
    except OSError:
        raise IntegrityError("source manifest output directory is invalid") from None
    if artifacts.is_relative_to(root):
        raise ValueError("source manifest artifacts root must be outside raw root")
    output_parent = _plain_output_directory(artifacts / "manifests", artifacts)
    output = output_parent / "raw_source_manifest.parquet"
    lock = artifacts / ".raw-source-manifest.lock"
    with (
        _destination_lock(_snapshot_lock_path(root)),
        _destination_lock(lock),
    ):
        receipts = sorted(root.rglob("*.source.json"), key=lambda path: path.as_posix())
        if not receipts:
            raise IntegrityError("no source receipts found")
        rows: list[dict[str, object]] = []
        records: list[SourceRecord] = []
        keys: set[tuple[str, str, str]] = set()
        relative_paths: set[str] = set()
        for receipt in receipts:
            record, payload = _load_source_receipt(receipt, root, source_references)
            key = (record.source_id, record.city_id, record.asset_key)
            relative_path = unicodedata.normalize("NFC", record.relative_path).casefold()
            if key in keys or relative_path in relative_paths:
                raise IntegrityError("duplicate source receipt key or relative path")
            keys.add(key)
            relative_paths.add(relative_path)
            records.append(record)
            rows.append(payload)
        rows.sort(
            key=lambda row: (
                *[str(row[name]) for name in RAW_SOURCE_COLUMNS[1:4]],
                str(row["relative_path"]),
            )
        )

        def revalidate_assets() -> None:
            for record in records:
                _asset_path(root, record)

        _write_manifest(output, rows, revalidate_assets)
        _plain_output_directory(output_parent, artifacts)
    return output


def _retrieved_at(existing: SourceRecord | None) -> str:
    if existing is not None:
        return existing.retrieved_at
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _request_sha256(url: str) -> str:
    return sha256_bytes(canonical_json_bytes({"method": "GET", "url": url}))


def _existing_source_record(
    asset: Path,
    raw_root: Path,
    protocol: LoadedProtocol,
) -> SourceRecord | None:
    receipt = asset.with_name(f"{asset.name}.source.json")
    if not receipt.exists():
        return None
    record, _ = _load_source_receipt(receipt, raw_root, _frozen_source_references(protocol))
    return record


def _remote_size(session: requests.Session, url: str) -> int:
    try:
        response = session.head(url, timeout=(10, 60), allow_redirects=False)
    except requests.RequestException:
        raise DownloadError("source size request failed") from None
    try:
        if response.status_code != 200:
            raise DownloadError("source size request returned an unexpected status")
        length = _response_length(response.headers)
        if length is None:
            raise DownloadError("source size is unavailable")
        return length
    finally:
        _best_effort_close(response)


def _finish_public_source(
    *,
    raw_root: Path,
    protocol: LoadedProtocol,
    download: DownloadReceipt,
    existing: SourceRecord | None,
    source_id: str,
    asset_key: str,
    provider: str,
    version_or_release: str,
    source_url_or_doi: str,
    license_name: str,
    privacy_status: str,
    reference_date_or_period: str,
    upstream_checksum_algorithm: str | None = None,
    upstream_checksum: str | None = None,
) -> SourceRecord:
    record = SourceRecord(
        source_id=source_id,
        city_id="all",
        asset_key=asset_key,
        provider=provider,
        version_or_release=version_or_release,
        source_url_or_doi=source_url_or_doi,
        license=license_name,
        privacy_status=privacy_status,
        reference_date_or_period=reference_date_or_period,
        relative_path=download.path.relative_to(raw_root).as_posix(),
        retrieved_at=_retrieved_at(existing),
        bytes=download.bytes_written,
        sha256=download.sha256,
        upstream_checksum_algorithm=upstream_checksum_algorithm,
        upstream_checksum=upstream_checksum,
        request_sha256=_request_sha256(source_url_or_doi),
    )
    write_source_receipt(record, raw_root, protocol)
    return record


def fetch_crowdflood(
    session: requests.Session,
    raw_root: Path,
    protocol: LoadedProtocol,
) -> SourceRecord:
    """Fetch the frozen CrowdFlood v1 archive."""
    source = protocol.sources["crowdflood_v1"]
    destination = raw_root / "crowdflood_v1" / str(source["record_id"]) / str(source["filename"])
    existing = _existing_source_record(destination, raw_root, protocol)
    download = download_http(
        session,
        str(source["url"]),
        destination,
        asset_identity=f"zenodo:{source['record_id']}:{source['filename']}",
        expected_size=int(source["bytes"]),
        expected_checksums={"md5": str(source["md5"])},
    )
    return _finish_public_source(
        raw_root=raw_root,
        protocol=protocol,
        download=download,
        existing=existing,
        source_id="crowdflood_v1",
        asset_key="crowdflood-v1-archive",
        provider=str(source["provider"]),
        version_or_release=str(source["record_id"]),
        source_url_or_doi=str(source["url"]),
        license_name=str(source["license"]),
        privacy_status=str(source["privacy_status"]),
        reference_date_or_period="v1",
        upstream_checksum_algorithm="md5",
        upstream_checksum=str(source["md5"]),
    )


def fetch_geoboundaries(
    session: requests.Session,
    raw_root: Path,
    protocol: LoadedProtocol,
) -> SourceRecord:
    """Fetch the frozen China ADM2 boundary file."""
    source = protocol.sources["geoboundaries_chn_adm2"]
    commit = str(source["commit"])
    destination = (
        raw_root / "geoboundaries_chn_adm2" / commit[:8] / "geoBoundaries-CHN-ADM2.geojson"
    )
    existing = _existing_source_record(destination, raw_root, protocol)
    download = download_http(
        session,
        str(source["url"]),
        destination,
        asset_identity=f"geoboundaries:{commit}:CHN:ADM2",
        expected_size=int(source["bytes"]),
        expected_checksums={"sha256": str(source["sha256"])},
    )
    return _finish_public_source(
        raw_root=raw_root,
        protocol=protocol,
        download=download,
        existing=existing,
        source_id="geoboundaries_chn_adm2",
        asset_key="CHN-ADM2",
        provider=str(source["provider"]),
        version_or_release=commit,
        source_url_or_doi=str(source["url"]),
        license_name=str(source["license"]),
        privacy_status=str(source["privacy_status"]),
        reference_date_or_period=f"commit-{commit[:8]}",
        upstream_checksum_algorithm="sha256",
        upstream_checksum=str(source["sha256"]),
    )


def fetch_worldcover_grid(
    session: requests.Session,
    raw_root: Path,
    protocol: LoadedProtocol,
    expected_size: int | None = None,
) -> SourceRecord:
    """Fetch the versioned WorldCover tile grid, not the raster tiles."""
    source = protocol.sources["worldcover_2021_v200"]
    url = str(source["grid_fgb_url"])
    destination = raw_root / "worldcover_2021_v200" / "v200" / "esa_worldcover_grid.fgb"
    existing = _existing_source_record(destination, raw_root, protocol)
    size = _remote_size(session, url) if expected_size is None else expected_size
    checksums = {"sha256": existing.sha256} if existing is not None else None
    download = download_http(
        session,
        url,
        destination,
        asset_identity="worldcover:2021:v200:grid-fgb",
        expected_size=size,
        expected_checksums=checksums,
    )
    return _finish_public_source(
        raw_root=raw_root,
        protocol=protocol,
        download=download,
        existing=existing,
        source_id="worldcover_2021_v200",
        asset_key="worldcover-grid-fgb",
        provider=str(source["provider"]),
        version_or_release=str(source["version"]),
        source_url_or_doi=url,
        license_name=str(source["license"]),
        privacy_status=str(source["privacy_status"]),
        reference_date_or_period=str(source["year"]),
    )


def fetch_public_sources(
    session: requests.Session,
    raw_root: str | PathLike[str],
    protocol: LoadedProtocol,
    source_id: str = "all",
) -> list[SourceRecord]:
    """Fetch one or all currently supported unauthenticated sources."""
    allowed = {
        "all",
        "crowdflood_v1",
        "geoboundaries_chn_adm2",
        "worldcover_2021_v200",
    }
    if source_id not in allowed:
        raise ValueError("unsupported public source")
    root = Path(raw_root).resolve()
    worldcover_size = 0
    planned_bytes = 0
    if source_id in {"all", "crowdflood_v1"}:
        planned_bytes += int(protocol.sources["crowdflood_v1"]["bytes"])
    if source_id in {"all", "geoboundaries_chn_adm2"}:
        planned_bytes += int(protocol.sources["geoboundaries_chn_adm2"]["bytes"])
    if source_id in {"all", "worldcover_2021_v200"}:
        worldcover_size = _remote_size(
            session,
            str(protocol.sources["worldcover_2021_v200"]["grid_fgb_url"]),
        )
        planned_bytes += worldcover_size
    ensure_free_space(root, planned_bytes)

    records: list[SourceRecord] = []
    if source_id in {"all", "crowdflood_v1"}:
        records.append(fetch_crowdflood(session, root, protocol))
    if source_id in {"all", "geoboundaries_chn_adm2"}:
        records.append(fetch_geoboundaries(session, root, protocol))
    if source_id in {"all", "worldcover_2021_v200"}:
        records.append(fetch_worldcover_grid(session, root, protocol, worldcover_size))
    return records


def write_public_source_report(
    records: list[SourceRecord],
    artifacts_root: str | PathLike[str],
) -> Path:
    """Write compact, secret-free evidence for the public-source fetch."""
    root = Path(artifacts_root)
    report = root / "reports" / "public_sources.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record.source_id)
    payload = {
        "schema_version": 1,
        "manifest": "manifests/raw_source_manifest.parquet",
        "file_count": len(ordered),
        "total_bytes": sum(record.bytes for record in ordered),
        "sources": [
            {
                "source_id": record.source_id,
                "relative_path": record.relative_path,
                "bytes": record.bytes,
                "sha256": record.sha256,
            }
            for record in ordered
        ],
    }
    _atomic_stream_write(report, lambda stream: stream.write(canonical_json_bytes(payload)))
    return report


def _save_sidecar(
    path: Path,
    url: str,
    asset_identity_sha256: str,
    etag: str,
    expected_total: int,
) -> None:
    value = {
        "asset_identity_sha256": asset_identity_sha256,
        "etag": etag,
        "expected_total": expected_total,
        "origin_sha256": _origin_sha256(url),
    }
    temporary = path.with_name(f"{path.name}.tmp")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _best_effort_sync_directory(path.parent)


def _save_journal(
    path: Path,
    verified: _VerifiedFile,
    url: str,
    asset_identity_sha256: str,
) -> None:
    value = {
        "asset_identity_sha256": asset_identity_sha256,
        "origin_sha256": _origin_sha256(url),
        "schema": 1,
        "sha256": verified.sha256,
        "size": verified.size,
    }
    temporary = path.with_name(f"{path.name}.tmp")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _best_effort_sync_directory(path.parent)


def _load_journal(
    path: Path,
    url: str,
    asset_identity_sha256: str,
) -> _VerifiedFile | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise IntegrityError("verified journal failed integrity validation") from None
    if not isinstance(value, dict) or set(value) != {
        "asset_identity_sha256",
        "origin_sha256",
        "schema",
        "sha256",
        "size",
    }:
        raise IntegrityError("verified journal failed integrity validation")
    size = value["size"]
    sha256 = value["sha256"]
    origin_sha256 = value["origin_sha256"]
    stored_asset_identity_sha256 = value["asset_identity_sha256"]
    if (
        type(value["schema"]) is not int
        or value["schema"] != 1
        or type(size) is not int
        or size < 0
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        or not isinstance(origin_sha256, str)
        or origin_sha256 != _origin_sha256(url)
        or not isinstance(stored_asset_identity_sha256, str)
        or stored_asset_identity_sha256 != asset_identity_sha256
    ):
        raise IntegrityError("verified journal failed integrity validation")
    return _VerifiedFile(size, sha256)


def _remove_sidecar(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _best_effort_sync_directory(path.parent)


def _remove_orphan_journal(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        raise DownloadError("orphan verified journal cleanup failed") from None
    _best_effort_sync_directory(path.parent)


def _reset_part(part: Path, sidecar: Path) -> None:
    with part.open("wb") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    _remove_sidecar(sidecar)


def _discard_part(part: Path, sidecar: Path) -> None:
    _remove_sidecar(sidecar)
    try:
        part.unlink()
    except FileNotFoundError:
        return
    except OSError:
        with part.open("wb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
    _best_effort_sync_directory(part.parent)


def _best_effort_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _cleanup_committed_auxiliary(part: Path, sidecar: Path) -> None:
    _best_effort_unlink(part)
    _best_effort_unlink(sidecar)
    _best_effort_sync_directory(part.parent)


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def _probe_hardlink(parent: Path) -> None:
    try:
        descriptor, source_name = tempfile.mkstemp(prefix=".floodgnn-link-probe-", dir=parent)
    except OSError:
        raise DownloadError("hard-link capability unavailable") from None
    os.close(descriptor)
    source = Path(source_name)
    target = source.with_name(f"{source.name}.link")
    failed = False
    try:
        os.link(source, target)
    except OSError:
        failed = True
    finally:
        for path in (target, source):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                failed = True
    if failed:
        raise DownloadError("hard-link capability unavailable")


def _existing_receipt(
    destination: Path,
    expected_size: int | None,
    checksums: Mapping[str, str],
) -> DownloadReceipt:
    if expected_size is None or not checksums:
        raise IntegrityError("existing destination requires size and checksum validation")
    verified = _validate_file(destination, expected_size, checksums)
    return DownloadReceipt(destination, verified.size, verified.sha256, False, None)


def _validate_journaled_file(
    path: Path,
    journal: _VerifiedFile,
    expected_size: int | None,
    checksums: Mapping[str, str],
) -> _VerifiedFile:
    if expected_size is not None and expected_size != journal.size:
        raise IntegrityError("file size failed integrity validation")
    if "sha256" in checksums and checksums["sha256"] != journal.sha256:
        raise IntegrityError("sha256 failed integrity validation")
    combined_checksums = dict(checksums)
    combined_checksums["sha256"] = journal.sha256
    return _validate_file(path, journal.size, combined_checksums)


def _publish(
    part: Path,
    destination: Path,
    url: str,
    asset_identity_sha256: str,
    verified: _VerifiedFile,
    expected_size: int | None,
    checksums: Mapping[str, str],
    *,
    resumed: bool,
    etag: str | None,
) -> DownloadReceipt:
    if destination.exists():
        receipt = _existing_receipt(destination, expected_size, checksums)
        _best_effort_unlink(part)
        return receipt
    _save_journal(
        _journal_path(destination),
        verified,
        url,
        asset_identity_sha256,
    )
    try:
        os.link(part, destination)
    except FileExistsError:
        if not destination.exists():
            raise IntegrityError("destination appeared during publication") from None
        committed = _validate_journaled_file(
            destination,
            verified,
            expected_size,
            checksums,
        )
        receipt = DownloadReceipt(destination, committed.size, committed.sha256, False, None)
        _best_effort_unlink(part)
        return receipt
    except OSError:
        raise DownloadError("atomic publish failed") from None
    _best_effort_unlink(part)
    _best_effort_sync_directory(destination.parent)
    return DownloadReceipt(destination, verified.size, verified.sha256, resumed, etag)


def _request_headers(
    headers: Mapping[str, str] | None,
    offset: int,
    sidecar: Mapping[str, object] | None,
) -> dict[str, str]:
    excluded = {"accept-encoding", "range", "if-range"}
    result = {
        key: value for key, value in (headers or {}).items() if key.casefold() not in excluded
    }
    result["Accept-Encoding"] = "identity"
    if offset:
        result["Range"] = f"bytes={offset}-"
        if sidecar is not None:
            result["If-Range"] = str(sidecar["etag"])
    return result


def _resume_offset(
    part: Path,
    sidecar: Mapping[str, object] | None,
    checksums: Mapping[str, str],
) -> int:
    if not part.is_file():
        return 0
    size = part.stat().st_size
    if size == 0:
        return 0
    if checksums or sidecar is not None:
        return size
    return 0


def _preflight_partial(
    part: Path,
    sidecar_path: Path,
    sidecar: Mapping[str, object] | None,
    expected_size: int | None,
) -> Mapping[str, object] | None:
    trusted_total = expected_size
    if trusted_total is None and sidecar is not None:
        trusted_total = int(sidecar["expected_total"])
    if part.is_file() and trusted_total is not None and part.stat().st_size > trusted_total:
        _discard_part(part, sidecar_path)
        return None
    return sidecar


def _verify_complete_part(
    part: Path,
    sidecar: Path,
    expected_size: int | None,
    checksums: Mapping[str, str],
) -> _VerifiedFile:
    try:
        return _validate_file(part, expected_size, checksums)
    except IntegrityError:
        if checksums and (expected_size is None or part.stat().st_size == expected_size):
            _discard_part(part, sidecar)
        raise


def _write_response_body(
    response: requests.Response,
    part: Path,
    sidecar: Path,
    maximum_size: int | None,
    chunk_size: int,
    safe_url: str,
) -> int:
    transferred = 0
    interrupted = False
    oversized = False
    observed_size = part.stat().st_size
    with part.open("ab") as stream:
        try:
            for data in response.iter_content(chunk_size=chunk_size):
                if data:
                    if maximum_size is not None and observed_size + len(data) > maximum_size:
                        oversized = True
                        break
                    stream.write(data)
                    transferred += len(data)
                    observed_size += len(data)
        except requests.RequestException:
            stream.flush()
            os.fsync(stream.fileno())
            interrupted = True
        stream.flush()
        os.fsync(stream.fileno())
    if oversized:
        _discard_part(part, sidecar)
        raise IntegrityError("downloaded size exceeded integrity boundary")
    if interrupted:
        raise DownloadError(f"download failed for {safe_url}") from None
    return transferred


def _handle_response(
    response: requests.Response,
    url: str,
    safe_url: str,
    final: Path,
    part: Path,
    sidecar_path: Path,
    sidecar: Mapping[str, object] | None,
    offset: int,
    expected_size: int | None,
    checksums: Mapping[str, str],
    asset_identity_sha256: str,
    chunk_size: int,
) -> DownloadReceipt:
    status = response.status_code
    response_headers = response.headers
    response_etag = _header(response_headers, "ETag")

    if status == 416:
        if sidecar is not None and response_etag != sidecar["etag"]:
            _discard_part(part, sidecar_path)
            raise IntegrityError("ETag changed during range completion")
        match = _UNSATISFIED_RANGE.fullmatch(_header(response_headers, "Content-Range") or "")
        if match is None:
            raise IntegrityError("invalid unsatisfied range response")
        total = int(match.group(1))
        if sidecar is not None and total != sidecar["expected_total"]:
            _discard_part(part, sidecar_path)
            raise IntegrityError("unsatisfied range total disagrees with resume metadata")
        if offset == 0:
            raise IntegrityError("invalid unsatisfied range response")
        if offset != total:
            if offset > total:
                _discard_part(part, sidecar_path)
            raise IntegrityError("unsatisfied range total failed integrity validation")
        if expected_size is not None and total != expected_size:
            raise IntegrityError("unsatisfied range total failed integrity validation")
        verified = _verify_complete_part(part, sidecar_path, total, checksums)
        receipt = _publish(
            part,
            final,
            url,
            asset_identity_sha256,
            verified,
            expected_size if expected_size is not None else total,
            checksums,
            resumed=True,
            etag=response_etag or (str(sidecar["etag"]) if sidecar else None),
        )
        _cleanup_committed_auxiliary(part, sidecar_path)
        return receipt

    if status == 206:
        if sidecar is not None and response_etag != sidecar["etag"]:
            _discard_part(part, sidecar_path)
            raise IntegrityError("ETag changed during range download")
        if offset == 0 or (_header(response_headers, "Content-Type") or "").lower().startswith(
            "multipart/"
        ):
            raise IntegrityError("invalid range response")
        match = _CONTENT_RANGE.fullmatch(_header(response_headers, "Content-Range") or "")
        if match is None:
            raise IntegrityError("invalid Content-Range")
        start, end, total = map(int, match.groups())
        if sidecar is not None and total != sidecar["expected_total"]:
            _discard_part(part, sidecar_path)
            raise IntegrityError("Content-Range total disagrees with resume metadata")
        if start != offset or end < start or end >= total:
            raise IntegrityError("Content-Range failed integrity validation")
        if expected_size is not None and total != expected_size:
            raise IntegrityError("Content-Range total failed integrity validation")
        response_length = _response_length(response_headers)
        range_bytes = end - start + 1
        if response_length is not None and response_length != range_bytes:
            raise IntegrityError("Content-Length disagrees with Content-Range")
        resumed = True
        maximum_size = end + 1
    elif status == 200:
        _reset_part(part, sidecar_path)
        response_length = _response_length(response_headers)
        total = response_length
        if (
            expected_size is not None
            and response_length is not None
            and response_length != expected_size
        ):
            raise IntegrityError("Content-Length disagrees with expected size")
        expected_total = expected_size if expected_size is not None else total
        if _strong_etag(response_etag) and expected_total is not None:
            _save_sidecar(
                sidecar_path,
                url,
                asset_identity_sha256,
                response_etag,
                expected_total,
            )
        resumed = False
        range_bytes = None
        maximum_size = expected_size if expected_size is not None else response_length
    else:
        raise DownloadError(f"unexpected HTTP status for {safe_url}")

    transferred = _write_response_body(
        response,
        part,
        sidecar_path,
        maximum_size,
        chunk_size,
        safe_url,
    )
    if range_bytes is not None and transferred != range_bytes:
        if transferred > range_bytes:
            _discard_part(part, sidecar_path)
        raise IntegrityError("range response byte count failed integrity validation")
    observed_size = part.stat().st_size
    if total is not None and observed_size != total:
        if observed_size > total:
            _discard_part(part, sidecar_path)
        raise IntegrityError("downloaded size disagrees with HTTP metadata")
    if expected_size is not None and observed_size != expected_size:
        if observed_size > expected_size:
            _discard_part(part, sidecar_path)
        raise IntegrityError("file size failed integrity validation")
    verified = _verify_complete_part(part, sidecar_path, expected_size, checksums)
    receipt = _publish(
        part,
        final,
        url,
        asset_identity_sha256,
        verified,
        expected_size if expected_size is not None else verified.size,
        checksums,
        resumed=resumed,
        etag=response_etag,
    )
    _cleanup_committed_auxiliary(part, sidecar_path)
    return receipt


def _best_effort_close(response: requests.Response) -> None:
    try:
        response.close()
    except Exception:  # noqa: BLE001 - cleanup cannot change the primary result
        return


def _download_locked(
    session: requests.Session,
    url: str,
    final: Path,
    headers: Mapping[str, str] | None,
    expected_size: int | None,
    checksums: Mapping[str, str],
    asset_identity_sha256: str,
    timeout: tuple[int, int],
    chunk_size: int,
) -> DownloadReceipt:
    part = _part_path(final)
    sidecar_path = _sidecar_path(final)
    journal_path = _journal_path(final)
    if final.exists():
        journal = _load_journal(journal_path, url, asset_identity_sha256)
        if journal is not None:
            verified = _validate_journaled_file(final, journal, expected_size, checksums)
            receipt = DownloadReceipt(final, verified.size, verified.sha256, False, None)
            _cleanup_committed_auxiliary(part, sidecar_path)
            return receipt
        committed_part = part.exists() and _same_file(part, final)
        receipt = _existing_receipt(final, expected_size, checksums)
        _save_journal(
            journal_path,
            _VerifiedFile(receipt.bytes_written, receipt.sha256),
            url,
            asset_identity_sha256,
        )
        if committed_part:
            _cleanup_committed_auxiliary(part, sidecar_path)
        return receipt
    if not part.exists():
        _remove_orphan_journal(journal_path)
    journal = _load_journal(journal_path, url, asset_identity_sha256)
    if journal is not None:
        if part.exists():
            verified = _validate_journaled_file(part, journal, expected_size, checksums)
            receipt = _publish(
                part,
                final,
                url,
                asset_identity_sha256,
                verified,
                journal.size,
                {"sha256": journal.sha256},
                resumed=False,
                etag=None,
            )
            _cleanup_committed_auxiliary(part, sidecar_path)
            return receipt
        _best_effort_unlink(journal_path)
        _best_effort_sync_directory(journal_path.parent)
    sidecar_exists = sidecar_path.exists()
    sidecar = _load_sidecar(sidecar_path, url, asset_identity_sha256, expected_size)
    if sidecar_exists and sidecar is None:
        _discard_part(part, sidecar_path)
    sidecar = _preflight_partial(part, sidecar_path, sidecar, expected_size)
    offset = _resume_offset(part, sidecar, checksums)
    safe_url = _safe_url(url)
    _probe_hardlink(final.parent)
    try:
        response = session.get(
            url,
            headers=_request_headers(headers, offset, sidecar),
            timeout=timeout,
            stream=True,
            allow_redirects=False,
        )
    except requests.RequestException:
        raise DownloadError(f"download failed for {safe_url}") from None
    try:
        try:
            return _handle_response(
                response,
                url,
                safe_url,
                final,
                part,
                sidecar_path,
                sidecar,
                offset,
                expected_size,
                checksums,
                asset_identity_sha256,
                chunk_size,
            )
        except OSError:
            raise DownloadError(f"download failed for {safe_url}") from None
    finally:
        _best_effort_close(response)


def download_http(
    session: requests.Session,
    url: str,
    destination: str | PathLike[str],
    *,
    headers: Mapping[str, str] | None = None,
    asset_identity: str | None = None,
    expected_size: int | None = None,
    expected_checksums: Mapping[str, str] | None = None,
    timeout: tuple[int, int] = (10, 120),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DownloadReceipt:
    """Download one HTTP asset without exposing unverified final bytes."""
    if expected_size is not None:
        _non_negative_integer("expected_size", expected_size)
    chunk = _non_negative_integer("chunk_size", chunk_size)
    if chunk == 0:
        raise ValueError("chunk_size must be positive")
    checksums = _normalize_checksums(expected_checksums)
    asset_identity_sha256 = _request_asset_identity_sha256(
        asset_identity,
        expected_size,
        checksums,
    )
    final = Path(destination)
    final.parent.mkdir(parents=True, exist_ok=True)
    with _destination_lock(_lock_path(final)):
        return _download_locked(
            session,
            url,
            final,
            headers,
            expected_size,
            checksums,
            asset_identity_sha256,
            timeout,
            chunk,
        )
