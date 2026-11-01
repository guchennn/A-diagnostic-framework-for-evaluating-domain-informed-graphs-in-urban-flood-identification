from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import pytest
import requests

import floodgnn_ceus.sources as sources_module
from floodgnn_ceus.sources import (
    DownloadError,
    InsufficientSpaceError,
    IntegrityError,
    ensure_free_space,
    required_free_bytes,
)
from floodgnn_ceus.sources import (
    download_http as _download_http,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: bytes = b"",
        *,
        headers: Mapping[str, str] | None = None,
        stream_error: Exception | None = None,
        stream_started: threading.Event | None = None,
        stream_release: threading.Event | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = dict(headers or {})
        self.stream_error = stream_error
        self.stream_started = stream_started
        self.stream_release = stream_release
        self.close_error = close_error
        self.closed = False

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        if self.stream_started is not None:
            self.stream_started.set()
        if self.stream_release is not None:
            assert self.stream_release.wait(timeout=5)
        midpoint = min(chunk_size, max(1, len(self.body) // 2))
        yield self.body[:midpoint]
        if self.stream_error is not None:
            raise self.stream_error
        yield self.body[midpoint:]

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"server returned {self.status_code}")

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class CountingResponse(FakeResponse):
    def __init__(
        self,
        status_code: int,
        chunks: list[bytes],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(status_code, headers=headers)
        self.chunks = chunks
        self.consumed = 0

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        del chunk_size
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk


class FakeSession:
    def __init__(
        self,
        *responses: FakeResponse | Exception,
        before_get: Callable[[], None] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.before_get = before_get
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        if self.before_get is not None:
            self.before_get()
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _md5(value: bytes) -> str:
    return hashlib.md5(value).hexdigest()


_TEST_ASSET_IDENTITY = "floodgnn:test-asset:v1"
download_http = partial(_download_http, asset_identity=_TEST_ASSET_IDENTITY)


def _asset_identity_sha256(value: str) -> str:
    return _sha256(value.encode())


def _content_asset_identity_sha256(size: int, sha256: str) -> str:
    return _asset_identity_sha256(f"content-sha256:{size}:{sha256}")


def _origin_sha256(url: str) -> str:
    parsed = urlsplit(url)
    scheme = parsed.scheme.casefold()
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    port_text = f":{port}" if port is not None else ""
    origin = urlunsplit((scheme, f"{host}{port_text}", "/", "", ""))
    return _sha256(origin.encode())


def _part(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part")


def _sidecar(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part.meta.json")


def _lock(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.part.lock")


def _journal(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.verified.json")


def _write_sidecar(
    destination: Path,
    url: str,
    etag: str,
    expected_total: int,
    *,
    asset_identity: str = _TEST_ASSET_IDENTITY,
) -> None:
    _sidecar(destination).write_text(
        json.dumps(
            {
                "asset_identity_sha256": _asset_identity_sha256(asset_identity),
                "etag": etag,
                "expected_total": expected_total,
                "origin_sha256": _origin_sha256(url),
            }
        ),
        encoding="utf-8",
    )


def _write_bound_sidecar(
    destination: Path,
    *,
    origin: str,
    asset_identity: str,
    etag: str,
    expected_total: int,
    extra: Mapping[str, object] | None = None,
) -> None:
    value: dict[str, object] = {
        "asset_identity_sha256": _asset_identity_sha256(asset_identity),
        "etag": etag,
        "expected_total": expected_total,
        "origin_sha256": _sha256(origin.encode()),
    }
    value.update(extra or {})
    _sidecar(destination).write_text(json.dumps(value), encoding="utf-8")


def _write_journal(
    destination: Path,
    content: bytes,
    url: str = "https://example.test/asset.bin",
    asset_identity: str = _TEST_ASSET_IDENTITY,
) -> None:
    _journal(destination).write_text(
        json.dumps(
            {
                "schema": 1,
                "size": len(content),
                "sha256": _sha256(content),
                "origin_sha256": _origin_sha256(url),
                "asset_identity_sha256": _asset_identity_sha256(asset_identity),
            }
        ),
        encoding="utf-8",
    )


def test_required_free_bytes_uses_exact_protocol_formula() -> None:
    assert required_free_bytes(123) == 2 * 123 + 20 * 1024**3
    assert required_free_bytes(7, multiplier=3, extra_bytes=11) == 32


@pytest.mark.parametrize(
    ("planned_bytes", "multiplier", "extra_bytes"),
    [(-1, 2, 0), (1.5, 2, 0), (1, -1, 0), (1, 2.0, 0), (1, 2, -1)],
)
def test_required_free_bytes_rejects_non_negative_integer_contract(
    planned_bytes: object,
    multiplier: object,
    extra_bytes: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        required_free_bytes(  # type: ignore[arg-type]
            planned_bytes,
            multiplier=multiplier,
            extra_bytes=extra_bytes,
        )


def test_ensure_free_space_creates_root_and_accepts_exact_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "new" / "downloads"
    required = required_free_bytes(9, multiplier=2, extra_bytes=3)
    monkeypatch.setattr(
        "floodgnn_ceus.sources.shutil.disk_usage",
        lambda path: SimpleNamespace(total=100, used=100 - required, free=required),
    )

    ensure_free_space(root, 9, multiplier=2, extra_bytes=3)

    assert root.is_dir()


def test_ensure_free_space_rejects_one_byte_below_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = required_free_bytes(9, multiplier=2, extra_bytes=3)
    monkeypatch.setattr(
        "floodgnn_ceus.sources.shutil.disk_usage",
        lambda path: SimpleNamespace(total=100, used=100 - required + 1, free=required - 1),
    )

    with pytest.raises(InsufficientSpaceError):
        ensure_free_space(tmp_path / "downloads", 9, multiplier=2, extra_bytes=3)


def test_download_full_response_publishes_verified_bytes(tmp_path: Path) -> None:
    content = b"complete payload"
    destination = tmp_path / "nested" / "asset.bin"
    response = FakeResponse(
        200,
        content,
        headers={"Content-Length": str(len(content)), "ETag": '"asset-v1"'},
    )
    session = FakeSession(response)

    receipt = download_http(
        session,
        "https://example.test/asset.bin",
        destination,
        expected_size=len(content),
        expected_checksums={"sha256": _sha256(content), "md5": _md5(content)},
        chunk_size=4,
    )

    assert destination.read_bytes() == content
    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    assert receipt.path == destination
    assert receipt.bytes_written == len(content)
    assert receipt.sha256 == _sha256(content)
    assert receipt.resumed is False
    assert receipt.etag == '"asset-v1"'
    assert session.calls[0]["headers"] == {"Accept-Encoding": "identity"}
    assert session.calls[0]["allow_redirects"] is False
    assert response.closed is True


def test_download_resumes_206_with_strong_if_range(tmp_path: Path) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    _write_sidecar(destination, url, '"asset-v1"', 6)
    session = FakeSession(
        FakeResponse(
            206,
            b"def",
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "ETag": '"asset-v1"',
            },
        )
    )

    receipt = download_http(session, url, destination, expected_size=6, chunk_size=2)

    assert destination.read_bytes() == b"abcdef"
    assert receipt.resumed is True
    assert session.calls[0]["headers"] == {
        "Accept-Encoding": "identity",
        "Range": "bytes=3-",
        "If-Range": '"asset-v1"',
    }


def test_range_request_returning_200_restarts_from_zero(tmp_path: Path) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"stale")
    _write_sidecar(destination, url, '"asset-v1"', 5)
    session = FakeSession(
        FakeResponse(200, b"fresh", headers={"Content-Length": "5", "ETag": '"asset-v2"'})
    )

    receipt = download_http(session, url, destination, expected_size=5)

    assert destination.read_bytes() == b"fresh"
    assert receipt.resumed is False
    assert session.calls[0]["headers"] == {
        "Accept-Encoding": "identity",
        "Range": "bytes=5-",
        "If-Range": '"asset-v1"',
    }


@pytest.mark.parametrize("trusted_by", ["expected_size", "sidecar"])
def test_oversized_part_is_discarded_before_request(
    tmp_path: Path,
    trusted_by: str,
) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"oversized")
    expected_size = 5 if trusted_by == "expected_size" else None
    if trusted_by == "sidecar":
        _write_sidecar(destination, url, '"asset-v1"', 5)
    part_state_at_request: list[bool] = []
    session = FakeSession(
        FakeResponse(200, b"fresh", headers={"Content-Length": "5"}),
        before_get=lambda: part_state_at_request.append(_part(destination).exists()),
    )

    download_http(session, url, destination, expected_size=expected_size)

    assert destination.read_bytes() == b"fresh"
    assert part_state_at_request == [False]
    assert session.calls[0]["headers"] == {"Accept-Encoding": "identity"}


def test_200_restart_invalidates_old_generation_before_saving_new_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"stale-prefix")
    _write_sidecar(destination, url, '"asset-v1"', 5)
    interrupted = FakeResponse(
        200,
        b"fresh",
        headers={"Content-Length": "5", "ETag": '"asset-v2"'},
    )
    original_save = sources_module._save_sidecar

    def save_then_fail(
        path: Path,
        request_url: str,
        asset_identity_sha256: str,
        etag: str,
        total: int,
    ) -> None:
        original_save(path, request_url, asset_identity_sha256, etag, total)
        raise OSError("simulated crash after sidecar replace")

    with monkeypatch.context() as context:
        context.setattr(sources_module, "_save_sidecar", save_then_fail)
        with pytest.raises((DownloadError, OSError)):
            download_http(FakeSession(interrupted), url, destination, expected_size=5)

    assert _part(destination).read_bytes() == b""
    assert json.loads(_sidecar(destination).read_text(encoding="utf-8"))["etag"] == '"asset-v2"'
    assert interrupted.closed is True

    session = FakeSession(FakeResponse(200, b"fresh", headers={"Content-Length": "5"}))
    download_http(session, url, destination, expected_size=5)

    assert destination.read_bytes() == b"fresh"
    assert "Range" not in session.calls[0]["headers"]


def test_same_destination_lock_rejects_concurrent_download(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    started = threading.Event()
    release = threading.Event()
    first_session = FakeSession(
        FakeResponse(
            200,
            b"first",
            headers={"Content-Length": "5"},
            stream_started=started,
            stream_release=release,
        )
    )
    second_session = FakeSession(FakeResponse(200, b"other"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            download_http,
            first_session,
            "https://example.test/asset.bin",
            destination,
            expected_size=5,
        )
        assert started.wait(timeout=5)
        with pytest.raises(DownloadError, match="in progress|locked"):
            download_http(
                second_session,
                "https://example.test/asset.bin",
                destination,
                expected_size=5,
            )
        release.set()
        assert future.result().path == destination

    assert second_session.calls == []
    assert _lock(destination).is_file()


@pytest.mark.parametrize(
    "content_range",
    ["bytes 2-5/6", "bytes 3-5/7", "bytes 3-5/*", "items 3-5/6"],
)
def test_invalid_content_range_never_appends(tmp_path: Path, content_range: str) -> None:
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    session = FakeSession(
        FakeResponse(
            206,
            b"def",
            headers={"Content-Length": "3", "Content-Range": content_range},
        )
    )

    with pytest.raises(IntegrityError):
        download_http(
            session,
            "https://example.test/asset.bin",
            destination,
            expected_size=6,
            expected_checksums={"sha256": _sha256(b"abcdef")},
        )

    assert _part(destination).read_bytes() == b"abc"
    assert not destination.exists()


def test_multipart_range_response_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    session = FakeSession(
        FakeResponse(
            206,
            b"def",
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "Content-Type": "multipart/byteranges; boundary=x",
            },
        )
    )

    with pytest.raises(IntegrityError, match="range"):
        download_http(
            session,
            "https://example.test/asset.bin",
            destination,
            expected_size=6,
            expected_checksums={"sha256": _sha256(b"abcdef")},
        )

    assert _part(destination).read_bytes() == b"abc"


def test_206_sidecar_total_mismatch_discards_generation_and_full_retry_recovers(
    tmp_path: Path,
) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    _write_sidecar(destination, url, '"asset-v1"', 6)
    session = FakeSession(
        FakeResponse(
            206,
            b"defg",
            headers={
                "Content-Length": "4",
                "Content-Range": "bytes 3-6/7",
                "ETag": '"asset-v1"',
            },
        )
    )

    with pytest.raises(IntegrityError, match="total"):
        download_http(session, url, destination)

    assert not destination.exists()
    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    assert not _journal(destination).exists()
    retry = FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"}))

    receipt = download_http(retry, url, destination)

    assert receipt.sha256 == _sha256(b"abcdef")
    assert destination.read_bytes() == b"abcdef"
    assert "Range" not in retry.calls[0]["headers"]


@pytest.mark.parametrize(
    ("status", "body", "response_headers"),
    [
        (
            206,
            b"bbbb",
            {
                "Content-Length": "4",
                "Content-Range": "bytes 3-6/7",
                "ETag": '"asset-v2"',
            },
        ),
        (416, b"", {"Content-Range": "bytes */3", "ETag": '"asset-v2"'}),
    ],
)
def test_resume_metadata_conflict_precedes_caller_size_and_forces_clean_retry(
    tmp_path: Path,
    status: int,
    body: bytes,
    response_headers: dict[str, str],
) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"aaa")
    _write_sidecar(destination, url, '"asset-v1"', 6)

    with pytest.raises(IntegrityError, match="total|ETag"):
        download_http(
            FakeSession(FakeResponse(status, body, headers=response_headers)),
            url,
            destination,
            expected_size=6,
        )

    assert not destination.exists()
    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    assert not _journal(destination).exists()
    retry = FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"}))

    download_http(retry, url, destination, expected_size=6)

    assert destination.read_bytes() == b"abcdef"
    assert "Range" not in retry.calls[0]["headers"]


def test_416_etag_conflict_precedes_offset_and_total_checks_then_retries_cleanly(
    tmp_path: Path,
) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"aaa")
    _write_sidecar(destination, url, '"asset-v1"', 6)

    with pytest.raises(IntegrityError, match="ETag"):
        download_http(
            FakeSession(
                FakeResponse(
                    416,
                    headers={"Content-Range": "bytes */7", "ETag": '"asset-v2"'},
                )
            ),
            url,
            destination,
            expected_size=6,
        )

    assert not destination.exists()
    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    assert not _journal(destination).exists()
    retry = FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"}))

    download_http(retry, url, destination, expected_size=6)

    assert destination.read_bytes() == b"abcdef"
    assert "Range" not in retry.calls[0]["headers"]


def test_206_etag_conflict_precedes_content_range_structure_and_retries_cleanly(
    tmp_path: Path,
) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"aaa")
    _write_sidecar(destination, url, '"asset-v1"', 6)

    with pytest.raises(IntegrityError, match="ETag"):
        download_http(
            FakeSession(
                FakeResponse(
                    206,
                    b"untrusted",
                    headers={"Content-Range": "invalid", "ETag": '"asset-v2"'},
                )
            ),
            url,
            destination,
            expected_size=6,
        )

    assert not destination.exists()
    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    assert not _journal(destination).exists()
    retry = FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"}))

    download_http(retry, url, destination, expected_size=6)

    assert destination.read_bytes() == b"abcdef"
    assert "Range" not in retry.calls[0]["headers"]


def test_206_etag_conflict_discards_generation_and_forces_full_retry(tmp_path: Path) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    _write_sidecar(destination, url, '"asset-v1"', 6)
    response = FakeResponse(
        206,
        b"def",
        headers={
            "Content-Length": "3",
            "Content-Range": "bytes 3-5/6",
            "ETag": '"asset-v2"',
        },
    )

    with pytest.raises(IntegrityError, match="ETag"):
        download_http(FakeSession(response), url, destination)

    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    retry = FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"}))

    download_http(retry, url, destination)

    assert "Range" not in retry.calls[0]["headers"]


def test_206_missing_etag_discards_sidecar_generation_and_forces_full_retry(
    tmp_path: Path,
) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    _write_sidecar(destination, url, '"asset-v1"', 6)
    response = FakeResponse(
        206,
        b"def",
        headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"},
    )

    with pytest.raises(IntegrityError, match="ETag"):
        download_http(FakeSession(response), url, destination)

    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    retry = FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"}))

    download_http(retry, url, destination)

    assert destination.read_bytes() == b"abcdef"
    assert "Range" not in retry.calls[0]["headers"]


@pytest.mark.parametrize(
    ("body", "retained_part"),
    [(b"de", b"abcde"), (b"defg", None)],
)
def test_206_without_content_length_requires_exact_range_bytes(
    tmp_path: Path,
    body: bytes,
    retained_part: bytes | None,
) -> None:
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    response = FakeResponse(206, body, headers={"Content-Range": "bytes 3-5/6"})

    with pytest.raises(IntegrityError, match="range|size|metadata"):
        download_http(
            FakeSession(response),
            "https://example.test/asset.bin",
            destination,
            expected_size=6,
            expected_checksums={"sha256": _sha256(b"abcdef")},
        )

    assert not destination.exists()
    if retained_part is None:
        assert not _part(destination).exists()
    else:
        assert _part(destination).read_bytes() == retained_part
    assert response.closed is True


def test_416_publishes_part_only_when_server_total_matches(tmp_path: Path) -> None:
    content = b"already complete"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(content)
    response = FakeResponse(416, headers={"Content-Range": f"bytes */{len(content)}"})
    session = FakeSession(response)

    receipt = download_http(
        session,
        "https://example.test/asset.bin",
        destination,
        expected_size=len(content),
        expected_checksums={"sha256": _sha256(content)},
    )

    assert destination.read_bytes() == content
    assert not _part(destination).exists()
    assert receipt.resumed is True
    assert response.closed is True


def test_416_smaller_total_discards_untrusted_oversized_part_for_retry(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abcd")
    first = FakeSession(FakeResponse(416, headers={"Content-Range": "bytes */3"}))

    with pytest.raises(IntegrityError, match="total"):
        download_http(
            first,
            "https://example.test/asset.bin",
            destination,
            expected_checksums={"sha256": _sha256(b"xyz")},
        )

    assert not _part(destination).exists()
    retry = FakeSession(FakeResponse(200, b"xyz", headers={"Content-Length": "3"}))

    download_http(
        retry,
        "https://example.test/asset.bin",
        destination,
        expected_checksums={"sha256": _sha256(b"xyz")},
    )

    assert destination.read_bytes() == b"xyz"
    assert "Range" not in retry.calls[0]["headers"]


def test_416_etag_must_match_resume_sidecar(tmp_path: Path) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    _write_sidecar(destination, url, '"asset-v1"', 3)
    response = FakeResponse(
        416,
        headers={"Content-Range": "bytes */3", "ETag": '"asset-v2"'},
    )

    with pytest.raises(IntegrityError, match="ETag"):
        download_http(FakeSession(response), url, destination)

    assert not destination.exists()
    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    assert response.closed is True


def test_416_missing_etag_discards_sidecar_generation_and_forces_full_retry(
    tmp_path: Path,
) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    _write_sidecar(destination, url, '"asset-v1"', 3)
    response = FakeResponse(416, headers={"Content-Range": "bytes */3"})

    with pytest.raises(IntegrityError, match="ETag"):
        download_http(FakeSession(response), url, destination)

    assert not destination.exists()
    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    retry = FakeSession(FakeResponse(200, b"abc", headers={"Content-Length": "3"}))

    download_http(retry, url, destination)

    assert destination.read_bytes() == b"abc"
    assert "Range" not in retry.calls[0]["headers"]


def test_416_total_must_match_resume_sidecar_and_discards_generation(tmp_path: Path) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"abc")
    _write_sidecar(destination, url, '"asset-v1"', 4)

    with pytest.raises(IntegrityError, match="total"):
        download_http(
            FakeSession(
                FakeResponse(
                    416,
                    headers={"Content-Range": "bytes */3", "ETag": '"asset-v1"'},
                )
            ),
            url,
            destination,
        )

    assert not destination.exists()
    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()


@pytest.mark.parametrize("status", [200, 206])
def test_oversized_chunk_is_rejected_before_write_and_stops_consuming(
    tmp_path: Path,
    status: int,
) -> None:
    destination = tmp_path / "asset.bin"
    if status == 200:
        response = CountingResponse(200, [b"abc", b"d", b"must-not-consume"])
        expected_size = 3
        checksums = None
    else:
        _part(destination).write_bytes(b"abc")
        response = CountingResponse(
            206,
            [b"def", b"g", b"must-not-consume"],
            headers={"Content-Range": "bytes 3-5/6"},
        )
        expected_size = 6
        checksums = {"sha256": _sha256(b"abcdef")}

    with pytest.raises(IntegrityError, match="size|range|metadata"):
        download_http(
            FakeSession(response),
            "https://example.test/asset.bin",
            destination,
            expected_size=expected_size,
            expected_checksums=checksums,
        )

    assert response.consumed == 2
    assert not destination.exists()
    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()


def test_stream_interruption_keeps_partial_bytes_and_safe_sidecar(tmp_path: Path) -> None:
    url = "https://user:password@example.test/asset.bin?token=signed-secret"
    destination = tmp_path / "asset.bin"
    response = FakeResponse(
        200,
        b"abcdef",
        headers={"Content-Length": "6", "ETag": '"asset-v1"'},
        stream_error=requests.ConnectionError(f"connection lost for {url}"),
    )
    session = FakeSession(response)

    with pytest.raises(DownloadError) as error:
        download_http(
            session,
            url,
            destination,
            headers={"Authorization": "Bearer authorization-secret"},
            asset_identity="interrupted-private-asset-v1",
            expected_size=6,
            chunk_size=3,
        )

    assert _part(destination).read_bytes() == b"abc"
    assert not destination.exists()
    assert "signed-secret" not in str(error.value)
    assert "password" not in str(error.value)
    assert "authorization-secret" not in str(error.value)
    assert str(error.value).endswith("https://example.test/<redacted>")
    sidecar_text = _sidecar(destination).read_text(encoding="utf-8")
    assert url not in sidecar_text
    assert "signed-secret" not in sidecar_text
    assert "authorization-secret" not in sidecar_text
    assert json.loads(sidecar_text) == {
        "asset_identity_sha256": _asset_identity_sha256("interrupted-private-asset-v1"),
        "etag": '"asset-v1"',
        "expected_total": 6,
        "origin_sha256": _origin_sha256(url),
    }
    assert response.closed is True


def test_stream_interruption_discards_part_that_exceeds_expected_size(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    response = FakeResponse(
        200,
        b"abcdefgh",
        stream_error=requests.ConnectionError("interrupted after oversized chunk"),
    )

    with pytest.raises(DownloadError):
        download_http(
            FakeSession(response),
            "https://example.test/asset.bin",
            destination,
            expected_size=3,
            chunk_size=4,
        )

    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()


@pytest.mark.parametrize(
    "etag",
    ['W/"weak"', "unquoted", ""],
)
def test_weak_or_missing_etag_does_not_create_resume_sidecar(tmp_path: Path, etag: str) -> None:
    destination = tmp_path / "asset.bin"
    headers = {"Content-Length": "6"}
    if etag:
        headers["ETag"] = etag
    session = FakeSession(
        FakeResponse(
            200,
            b"abcdef",
            headers=headers,
            stream_error=requests.ConnectionError("interrupted"),
        )
    )

    with pytest.raises(DownloadError):
        download_http(
            session,
            "https://example.test/asset.bin",
            destination,
            expected_size=6,
            chunk_size=3,
        )

    assert _part(destination).read_bytes() == b"abc"
    assert not _sidecar(destination).exists()


def test_part_without_checksum_or_strong_etag_restarts_without_range(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"stale")
    session = FakeSession(FakeResponse(200, b"fresh", headers={"Content-Length": "5"}))

    download_http(
        session,
        "https://example.test/asset.bin",
        destination,
        expected_size=5,
    )

    assert destination.read_bytes() == b"fresh"
    assert session.calls[0]["headers"] == {"Accept-Encoding": "identity"}


@pytest.mark.parametrize(
    ("body", "retained"),
    [(b"abcd", None), (b"ab", b"ab")],
)
def test_200_without_content_length_handles_expected_size_boundary(
    tmp_path: Path,
    body: bytes,
    retained: bytes | None,
) -> None:
    destination = tmp_path / "asset.bin"
    response = FakeResponse(200, body)

    with pytest.raises(IntegrityError, match="size|metadata"):
        download_http(
            FakeSession(response),
            "https://example.test/asset.bin",
            destination,
            expected_size=3,
        )

    if retained is None:
        assert not _part(destination).exists()
    else:
        assert _part(destination).read_bytes() == retained


@pytest.mark.parametrize(
    ("expected_size", "checksums", "retained_part"),
    [
        (7, {"sha256": _sha256(b"abcdef")}, b""),
        (6, {"sha256": "0" * 64}, None),
        (6, {"md5": "0" * 32}, None),
    ],
)
def test_size_or_hash_mismatch_does_not_publish(
    tmp_path: Path,
    expected_size: int,
    checksums: dict[str, str],
    retained_part: bytes | None,
) -> None:
    destination = tmp_path / "asset.bin"
    session = FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"}))

    with pytest.raises(IntegrityError):
        download_http(
            session,
            "https://example.test/asset.bin",
            destination,
            expected_size=expected_size,
            expected_checksums=checksums,
        )

    assert not destination.exists()
    if retained_part is None:
        assert not _part(destination).exists()
    else:
        assert _part(destination).read_bytes() == retained_part


def test_wrong_complete_checksum_is_quarantined_before_retry(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    expected = b"abcdef"
    wrong = FakeResponse(
        200,
        b"badbad",
        headers={"Content-Length": "6", "ETag": '"wrong-generation"'},
    )

    with pytest.raises(IntegrityError, match="sha256"):
        download_http(
            FakeSession(wrong),
            "https://example.test/asset.bin",
            destination,
            expected_size=6,
            expected_checksums={"sha256": _sha256(expected)},
        )

    assert not _part(destination).exists()
    assert not _sidecar(destination).exists()
    retry = FakeSession(FakeResponse(200, expected, headers={"Content-Length": "6"}))

    download_http(
        retry,
        "https://example.test/asset.bin",
        destination,
        expected_size=6,
        expected_checksums={"sha256": _sha256(expected)},
    )

    assert destination.read_bytes() == expected
    assert "Range" not in retry.calls[0]["headers"]


def test_existing_verified_final_returns_without_network(tmp_path: Path) -> None:
    content = b"already verified"
    destination = tmp_path / "asset.bin"
    destination.write_bytes(content)
    session = FakeSession(AssertionError("network must not be called"))

    receipt = download_http(
        session,
        "https://example.test/asset.bin",
        destination,
        expected_size=len(content),
        expected_checksums={"sha256": _sha256(content)},
    )

    assert receipt.bytes_written == len(content)
    assert receipt.sha256 == _sha256(content)
    assert session.calls == []


def test_plain_public_request_requires_asset_identity_before_network(tmp_path: Path) -> None:
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(DownloadError, match="asset_identity"):
        sources_module.download_http(
            session,
            "https://example.test/asset.bin",
            tmp_path / "asset.bin",
        )

    assert session.calls == []


def test_empty_asset_identity_is_rejected_before_network(tmp_path: Path) -> None:
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(DownloadError, match="non-empty"):
        sources_module.download_http(
            session,
            "https://example.test/asset.bin",
            tmp_path / "asset.bin",
            asset_identity="",
        )

    assert session.calls == []


def test_netrc_capable_session_cannot_bypass_asset_identity(tmp_path: Path) -> None:
    session = FakeSession(AssertionError("network must not be called"))
    session.trust_env = True

    with pytest.raises(DownloadError, match="asset_identity"):
        sources_module.download_http(
            session,
            "https://example.test/asset.bin",
            tmp_path / "asset.bin",
        )

    assert session.calls == []


def test_content_evidence_is_an_asset_identity_for_public_request(tmp_path: Path) -> None:
    content = b"abcdef"
    destination = tmp_path / "asset.bin"
    session = FakeSession(FakeResponse(200, content, headers={"Content-Length": "6"}))

    receipt = sources_module.download_http(
        session,
        "https://example.test/asset.bin",
        destination,
        expected_size=6,
        expected_checksums={"sha256": _sha256(content)},
    )

    assert receipt.sha256 == _sha256(content)
    assert json.loads(_journal(destination).read_text(encoding="utf-8"))[
        "asset_identity_sha256"
    ] == _content_asset_identity_sha256(6, _sha256(content))


def test_changed_asset_identity_discards_interrupted_generation_before_request(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "asset.bin"
    url = "https://example.test/asset.bin"
    interrupted = FakeResponse(
        200,
        b"AAABBB",
        headers={"Content-Length": "6", "ETag": '"asset-a"'},
        stream_error=requests.ConnectionError("tenant-a-token"),
    )

    with pytest.raises(DownloadError):
        sources_module.download_http(
            FakeSession(interrupted),
            url,
            destination,
            headers={"Authorization": "Bearer tenant-a-token"},
            asset_identity="tenant-a-asset-v1",
            expected_size=6,
            chunk_size=3,
        )

    assert _part(destination).read_bytes() == b"AAA"
    part_exists_at_request: list[bool] = []
    session = FakeSession(
        FakeResponse(200, b"BBBBBB", headers={"Content-Length": "6", "ETag": '"asset-b"'}),
        before_get=lambda: part_exists_at_request.append(_part(destination).exists()),
    )

    sources_module.download_http(
        session,
        url,
        destination,
        headers={"Authorization": "Bearer tenant-b-token"},
        asset_identity="tenant-b-asset-v1",
        expected_size=6,
    )

    assert part_exists_at_request == [False]
    assert session.calls[0]["headers"] == {
        "Authorization": "Bearer tenant-b-token",
        "Accept-Encoding": "identity",
    }
    assert destination.read_bytes() == b"BBBBBB"


@pytest.mark.parametrize(
    ("origin", "identity", "expected_total", "extra"),
    [
        ("https://other.test/", "asset-v1", 6, None),
        ("https://example.test/", "asset-v1", 7, None),
        ("https://example.test/", "asset-v1", 6, {"unexpected": "field"}),
    ],
)
def test_invalid_or_mismatched_sidecar_discards_part_before_request(
    tmp_path: Path,
    origin: str,
    identity: str,
    expected_total: int,
    extra: Mapping[str, object] | None,
) -> None:
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"AAA")
    _write_bound_sidecar(
        destination,
        origin=origin,
        asset_identity=identity,
        etag='"asset-v1"',
        expected_total=expected_total,
        extra=extra,
    )
    part_exists_at_request: list[bool] = []
    session = FakeSession(
        FakeResponse(200, b"BBBBBB", headers={"Content-Length": "6"}),
        before_get=lambda: part_exists_at_request.append(_part(destination).exists()),
    )

    sources_module.download_http(
        session,
        "https://example.test/asset.bin",
        destination,
        asset_identity="asset-v1",
        expected_size=6,
    )

    assert part_exists_at_request == [False]
    assert "Range" not in session.calls[0]["headers"]
    assert "If-Range" not in session.calls[0]["headers"]
    assert not _sidecar(destination).exists()


def test_same_origin_and_identity_resume_across_token_and_query_refresh(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "asset.bin"
    first_url = "https://example.test/download/path-token-a?token=query-token-a"
    second_url = "https://example.test/download/path-token-b?token=query-token-b"
    asset_identity = "stable-public-asset-v1"
    interrupted = FakeResponse(
        200,
        b"AAABBB",
        headers={"Content-Length": "6", "ETag": '"asset-v1"'},
        stream_error=requests.ConnectionError("token-a-secret"),
    )

    with pytest.raises(DownloadError):
        sources_module.download_http(
            FakeSession(interrupted),
            first_url,
            destination,
            headers={"Authorization": "Bearer token-a-secret"},
            asset_identity=asset_identity,
            expected_size=6,
            chunk_size=3,
        )

    recovery = FakeSession(
        FakeResponse(
            206,
            b"BBB",
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "ETag": '"asset-v1"',
            },
        )
    )
    receipt = sources_module.download_http(
        recovery,
        second_url,
        destination,
        headers={"Authorization": "Bearer token-b-secret"},
        asset_identity=asset_identity,
        expected_size=6,
        chunk_size=3,
    )

    assert receipt.resumed is True
    assert recovery.calls[0]["headers"] == {
        "Authorization": "Bearer token-b-secret",
        "Accept-Encoding": "identity",
        "Range": "bytes=3-",
        "If-Range": '"asset-v1"',
    }
    assert destination.read_bytes() == b"AAABBB"


def test_sidecar_and_journal_do_not_contain_path_or_query_derived_fingerprints(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "asset.bin"
    path_token = "p7"
    query_token = "q3"
    first_url = f"https://example.test/download/{path_token}?token={query_token}"
    second_url = "https://example.test/refresh?token=new"
    asset_identity = "stable-public-asset-v1"
    interrupted = FakeResponse(
        200,
        b"AAABBB",
        headers={"Content-Length": "6", "ETag": '"asset-v1"'},
        stream_error=requests.ConnectionError("interrupted"),
    )

    with pytest.raises(DownloadError):
        sources_module.download_http(
            FakeSession(interrupted),
            first_url,
            destination,
            asset_identity=asset_identity,
            expected_size=6,
            chunk_size=3,
        )
    sidecar_text = _sidecar(destination).read_text(encoding="utf-8")

    sources_module.download_http(
        FakeSession(
            FakeResponse(
                206,
                b"BBB",
                headers={
                    "Content-Length": "3",
                    "Content-Range": "bytes 3-5/6",
                    "ETag": '"asset-v1"',
                },
            )
        ),
        second_url,
        destination,
        asset_identity=asset_identity,
        expected_size=6,
        chunk_size=3,
    )
    journal_text = _journal(destination).read_text(encoding="utf-8")

    for persisted in (sidecar_text, journal_text):
        for secret in (path_token, query_token, first_url, f"download/{path_token}"):
            assert secret not in persisted
            assert _sha256(secret.encode()) not in persisted
        assert _sha256(first_url.encode()) not in persisted
        assert json.loads(persisted)["origin_sha256"] == _origin_sha256(first_url)


@pytest.mark.parametrize(
    ("initial_url", "recovery_url"),
    [
        ("http://example.test/asset?a=1", "http://example.test:80/refresh?b=2"),
        ("https://example.test:443/asset?a=1", "https://example.test/refresh?b=2"),
    ],
)
def test_journal_recovers_across_equivalent_default_port_origins_without_network(
    tmp_path: Path,
    initial_url: str,
    recovery_url: str,
) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-default-port-asset-v1"
    sources_module.download_http(
        FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
        initial_url,
        destination,
        asset_identity=asset_identity,
    )
    recovery = FakeSession(AssertionError("network must not be called"))

    receipt = sources_module.download_http(
        recovery,
        recovery_url,
        destination,
        asset_identity=asset_identity,
    )

    assert receipt.sha256 == _sha256(b"abcdef")
    assert recovery.calls == []


@pytest.mark.parametrize(
    ("initial_url", "recovery_url"),
    [
        ("http://example.test/asset?a=1", "http://example.test:80/refresh?b=2"),
        ("https://example.test:443/asset?a=1", "https://example.test/refresh?b=2"),
    ],
)
def test_sidecar_resumes_across_equivalent_default_port_origins(
    tmp_path: Path,
    initial_url: str,
    recovery_url: str,
) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-default-port-asset-v1"
    interrupted = FakeResponse(
        200,
        b"AAABBB",
        headers={"Content-Length": "6", "ETag": '"asset-v1"'},
        stream_error=requests.ConnectionError("interrupted"),
    )
    with pytest.raises(DownloadError):
        sources_module.download_http(
            FakeSession(interrupted),
            initial_url,
            destination,
            asset_identity=asset_identity,
            expected_size=6,
            chunk_size=3,
        )
    recovery = FakeSession(
        FakeResponse(
            206,
            b"BBB",
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "ETag": '"asset-v1"',
            },
        )
    )

    receipt = sources_module.download_http(
        recovery,
        recovery_url,
        destination,
        asset_identity=asset_identity,
        expected_size=6,
        chunk_size=3,
    )

    assert receipt.resumed is True
    assert recovery.calls[0]["headers"] == {
        "Accept-Encoding": "identity",
        "Range": "bytes=3-",
        "If-Range": '"asset-v1"',
    }
    assert destination.read_bytes() == b"AAABBB"


def test_origin_normalizes_unicode_hostname_to_idna_ascii() -> None:
    assert sources_module._origin_sha256(
        "https://bücher.example/asset"
    ) == sources_module._origin_sha256("https://xn--bcher-kva.example:443/other")


def test_journal_recovers_across_unicode_and_idna_hosts_without_network(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-idna-asset-v1"
    sources_module.download_http(
        FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
        "https://bücher.example/asset",
        destination,
        asset_identity=asset_identity,
    )
    recovery = FakeSession(AssertionError("network must not be called"))

    receipt = sources_module.download_http(
        recovery,
        "https://xn--bcher-kva.example:443/refresh",
        destination,
        asset_identity=asset_identity,
    )

    assert receipt.sha256 == _sha256(b"abcdef")
    assert recovery.calls == []


def test_origin_uses_idna2008_without_mapping_sharp_s_to_ss() -> None:
    unicode_origin = sources_module._origin_sha256("https://fa\u00df.de/asset")

    assert unicode_origin == sources_module._origin_sha256("https://xn--fa-hia.de:443/other")
    assert unicode_origin != sources_module._origin_sha256("https://fass.de/asset")


def test_journal_does_not_recover_from_sharp_s_host_to_ss_host(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-sharp-s-asset-v1"
    sources_module.download_http(
        FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
        "https://fa\u00df.de/asset",
        destination,
        asset_identity=asset_identity,
    )
    recovery = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="journal"):
        sources_module.download_http(
            recovery,
            "https://fass.de/refresh",
            destination,
            asset_identity=asset_identity,
        )

    assert destination.read_bytes() == b"abcdef"
    assert recovery.calls == []


def test_sidecar_resumes_across_unicode_and_idna_hosts(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-idna-asset-v1"
    interrupted = FakeResponse(
        200,
        b"AAABBB",
        headers={"Content-Length": "6", "ETag": '"asset-v1"'},
        stream_error=requests.ConnectionError("interrupted"),
    )
    with pytest.raises(DownloadError):
        sources_module.download_http(
            FakeSession(interrupted),
            "https://bücher.example/asset",
            destination,
            asset_identity=asset_identity,
            expected_size=6,
            chunk_size=3,
        )
    recovery = FakeSession(
        FakeResponse(
            206,
            b"BBB",
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "ETag": '"asset-v1"',
            },
        )
    )

    receipt = sources_module.download_http(
        recovery,
        "https://xn--bcher-kva.example:443/refresh",
        destination,
        asset_identity=asset_identity,
        expected_size=6,
        chunk_size=3,
    )

    assert receipt.resumed is True
    assert recovery.calls[0]["headers"]["Range"] == "bytes=3-"
    assert destination.read_bytes() == b"AAABBB"


def test_sidecar_does_not_resume_from_sharp_s_host_to_ss_host(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-sharp-s-asset-v1"
    interrupted = FakeResponse(
        200,
        b"AAABBB",
        headers={"Content-Length": "6", "ETag": '"shared-etag"'},
        stream_error=requests.ConnectionError("interrupted"),
    )
    with pytest.raises(DownloadError):
        sources_module.download_http(
            FakeSession(interrupted),
            "https://fa\u00df.de/asset",
            destination,
            asset_identity=asset_identity,
            expected_size=6,
            chunk_size=3,
        )
    recovery = FakeSession(
        FakeResponse(
            200,
            b"CCCDDD",
            headers={"Content-Length": "6", "ETag": '"shared-etag"'},
        )
    )

    sources_module.download_http(
        recovery,
        "https://fass.de/refresh",
        destination,
        asset_identity=asset_identity,
        expected_size=6,
        chunk_size=3,
    )

    assert "Range" not in recovery.calls[0]["headers"]
    assert destination.read_bytes() == b"CCCDDD"


def test_journal_rejects_nondefault_port_without_network(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-port-bound-asset-v1"
    sources_module.download_http(
        FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
        "https://example.test/asset",
        destination,
        asset_identity=asset_identity,
    )
    recovery = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="journal"):
        sources_module.download_http(
            recovery,
            "https://example.test:444/asset",
            destination,
            asset_identity=asset_identity,
        )

    assert destination.read_bytes() == b"abcdef"
    assert recovery.calls == []


def test_sidecar_nondefault_port_discards_partial_before_request(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-port-bound-asset-v1"
    interrupted = FakeResponse(
        200,
        b"AAABBB",
        headers={"Content-Length": "6", "ETag": '"asset-v1"'},
        stream_error=requests.ConnectionError("interrupted"),
    )
    with pytest.raises(DownloadError):
        sources_module.download_http(
            FakeSession(interrupted),
            "https://example.test/asset",
            destination,
            asset_identity=asset_identity,
            expected_size=6,
            chunk_size=3,
        )
    part_exists_at_request: list[bool] = []
    recovery = FakeSession(
        FakeResponse(200, b"BBBBBB", headers={"Content-Length": "6"}),
        before_get=lambda: part_exists_at_request.append(_part(destination).exists()),
    )

    sources_module.download_http(
        recovery,
        "https://example.test:444/asset",
        destination,
        asset_identity=asset_identity,
        expected_size=6,
    )

    assert part_exists_at_request == [False]
    assert "Range" not in recovery.calls[0]["headers"]
    assert "If-Range" not in recovery.calls[0]["headers"]
    assert destination.read_bytes() == b"BBBBBB"


def test_origin_normalization_preserves_ipv6_brackets_and_default_port() -> None:
    assert sources_module._origin_sha256("https://[2001:db8::1]/asset") == _sha256(
        b"https://[2001:db8::1]:443/"
    )
    assert sources_module._origin_sha256(
        "https://[2001:db8::1]/asset"
    ) == sources_module._origin_sha256("https://[2001:db8::1]:443/other")
    assert sources_module._origin_sha256(
        "https://[2001:db8::1]/asset"
    ) != sources_module._origin_sha256("https://[2001:db8::1]:444/asset")


@pytest.mark.parametrize(
    "url",
    [
        "https://authority-secret.test:port-secret/path-secret?query-secret",
        "https://user-secret:password-secret@[not-ipv6]/path-secret?query-secret",
    ],
)
def test_invalid_download_origin_error_is_redacted(url: str) -> None:
    with pytest.raises(DownloadError, match=r"^invalid download origin$") as error:
        sources_module._origin_sha256(url)

    message = str(error.value)
    for secret in (
        "authority-secret",
        "port-secret",
        "user-secret",
        "password-secret",
        "not-ipv6",
        "path-secret",
        "query-secret",
    ):
        assert secret not in message


def test_journal_recovers_across_expanded_and_compressed_ipv6_without_network(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-ipv6-asset-v1"
    sources_module.download_http(
        FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
        "https://[2001:0db8:0000:0000:0000:0000:0000:0001]/asset?a=1",
        destination,
        asset_identity=asset_identity,
    )
    recovery = FakeSession(AssertionError("network must not be called"))

    receipt = sources_module.download_http(
        recovery,
        "https://[2001:db8::1]:443/refresh?b=2",
        destination,
        asset_identity=asset_identity,
    )

    assert receipt.sha256 == _sha256(b"abcdef")
    assert recovery.calls == []


def test_sidecar_resumes_across_expanded_and_compressed_ipv6(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-ipv6-asset-v1"
    interrupted = FakeResponse(
        200,
        b"AAABBB",
        headers={"Content-Length": "6", "ETag": '"asset-v1"'},
        stream_error=requests.ConnectionError("interrupted"),
    )
    with pytest.raises(DownloadError):
        sources_module.download_http(
            FakeSession(interrupted),
            "https://[2001:0db8:0000:0000:0000:0000:0000:0001]/asset?a=1",
            destination,
            asset_identity=asset_identity,
            expected_size=6,
            chunk_size=3,
        )
    recovery = FakeSession(
        FakeResponse(
            206,
            b"BBB",
            headers={
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "ETag": '"asset-v1"',
            },
        )
    )

    receipt = sources_module.download_http(
        recovery,
        "https://[2001:db8::1]:443/refresh?b=2",
        destination,
        asset_identity=asset_identity,
        expected_size=6,
        chunk_size=3,
    )

    assert receipt.resumed is True
    assert recovery.calls[0]["headers"] == {
        "Accept-Encoding": "identity",
        "Range": "bytes=3-",
        "If-Range": '"asset-v1"',
    }
    assert destination.read_bytes() == b"AAABBB"


def test_journal_rejects_different_ipv6_without_network(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-ipv6-asset-v1"
    sources_module.download_http(
        FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
        "https://[2001:db8::1]/asset",
        destination,
        asset_identity=asset_identity,
    )
    recovery = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="journal"):
        sources_module.download_http(
            recovery,
            "https://[2001:db8::2]/asset",
            destination,
            asset_identity=asset_identity,
        )

    assert destination.read_bytes() == b"abcdef"
    assert recovery.calls == []


def test_sidecar_different_ipv6_discards_partial_before_request(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    asset_identity = "stable-ipv6-asset-v1"
    interrupted = FakeResponse(
        200,
        b"AAABBB",
        headers={"Content-Length": "6", "ETag": '"asset-v1"'},
        stream_error=requests.ConnectionError("interrupted"),
    )
    with pytest.raises(DownloadError):
        sources_module.download_http(
            FakeSession(interrupted),
            "https://[2001:db8::1]/asset",
            destination,
            asset_identity=asset_identity,
            expected_size=6,
            chunk_size=3,
        )
    part_exists_at_request: list[bool] = []
    recovery = FakeSession(
        FakeResponse(200, b"BBBBBB", headers={"Content-Length": "6"}),
        before_get=lambda: part_exists_at_request.append(_part(destination).exists()),
    )

    sources_module.download_http(
        recovery,
        "https://[2001:db8::2]/asset",
        destination,
        asset_identity=asset_identity,
        expected_size=6,
    )

    assert part_exists_at_request == [False]
    assert "Range" not in recovery.calls[0]["headers"]
    assert destination.read_bytes() == b"BBBBBB"


def test_scoped_ipv6_normalization_preserves_scope_and_compresses_address() -> None:
    expanded = "https://[fe80:0000:0000:0000:0000:0000:0000:0001%25eth0]/asset"
    compressed = "https://[fe80::1%25eth0]:443/other"

    assert sources_module._origin_sha256(expanded) == sources_module._origin_sha256(compressed)
    assert sources_module._origin_sha256(expanded) != sources_module._origin_sha256(
        "https://[fe80::1%25eth1]:443/other"
    )


@pytest.mark.parametrize("tenant", ["tenant-a-secret", "tenant-b-secret"])
def test_custom_request_headers_require_explicit_or_content_asset_identity(
    tmp_path: Path,
    tenant: str,
) -> None:
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(DownloadError, match="asset_identity") as error:
        sources_module.download_http(
            session,
            "https://example.test/asset.bin",
            tmp_path / "asset.bin",
            headers={"X-Tenant": tenant},
        )

    assert tenant not in str(error.value)
    assert session.calls == []


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("auth", ("user-secret", "password-secret")),
        ("cookies", {"session": "cookie-secret"}),
        ("params", {"token": "query-secret"}),
        ("headers", {"Authorization": "Bearer header-secret"}),
    ],
)
def test_nondefault_session_request_context_requires_asset_identity(
    tmp_path: Path,
    attribute: str,
    value: object,
) -> None:
    session = FakeSession(AssertionError("network must not be called"))
    setattr(session, attribute, value)

    with pytest.raises(DownloadError, match="asset_identity") as error:
        sources_module.download_http(
            session,
            "https://example.test/asset.bin",
            tmp_path / "asset.bin",
        )

    assert "secret" not in str(error.value)
    assert session.calls == []


def test_same_asset_identity_recovers_across_authorization_token_refresh(tmp_path: Path) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    asset_identity = "tenant-a-public-asset-v1"

    download_http(
        FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
        url,
        destination,
        headers={"Authorization": "Bearer token-a-secret"},
        asset_identity=asset_identity,
    )
    recovery = FakeSession(AssertionError("network must not be called"))

    receipt = download_http(
        recovery,
        url,
        destination,
        headers={"Authorization": "Bearer token-b-secret"},
        asset_identity=asset_identity,
    )

    assert receipt.sha256 == _sha256(b"abcdef")
    assert recovery.calls == []


def test_different_asset_identity_rejects_journal_and_preserves_final(tmp_path: Path) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"

    download_http(
        FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
        url,
        destination,
        headers={"Authorization": "Bearer token-a-secret"},
        asset_identity="tenant-a-public-asset-v1",
    )
    recovery = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="journal"):
        download_http(
            recovery,
            url,
            destination,
            headers={"Authorization": "Bearer token-b-secret"},
            asset_identity="tenant-b-public-asset-v1",
        )

    assert destination.read_bytes() == b"abcdef"
    assert _journal(destination).exists()
    assert recovery.calls == []


def test_content_evidence_provides_stable_identity_across_custom_headers(tmp_path: Path) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    content = b"abcdef"
    evidence = {"sha256": _sha256(content)}

    sources_module.download_http(
        FakeSession(FakeResponse(200, content, headers={"Content-Length": "6"})),
        url,
        destination,
        headers={"Authorization": "Bearer token-a-secret"},
        expected_size=6,
        expected_checksums=evidence,
    )
    recovery = FakeSession(AssertionError("network must not be called"))

    receipt = sources_module.download_http(
        recovery,
        url,
        destination,
        headers={"Authorization": "Bearer token-b-secret"},
        expected_size=6,
        expected_checksums=evidence,
    )

    assert receipt.sha256 == evidence["sha256"]
    assert json.loads(_journal(destination).read_text(encoding="utf-8"))[
        "asset_identity_sha256"
    ] == _content_asset_identity_sha256(6, evidence["sha256"])
    assert recovery.calls == []


def test_legacy_hardlink_without_journal_rejects_cross_url_without_evidence(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "asset.bin"
    part = _part(destination)
    content = b"committed"
    part.write_bytes(content)
    sources_module.os.link(part, destination)
    _write_sidecar(destination, "https://source-a.test/asset.bin", '"asset-v1"', len(content))
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="size and checksum"):
        download_http(session, "https://source-b.test/asset.bin", destination)

    assert destination.read_bytes() == content
    assert part.exists()
    assert sources_module.os.path.samefile(part, destination)
    assert _sidecar(destination).exists()
    assert not _journal(destination).exists()
    assert session.calls == []


def test_startup_migrates_verified_legacy_hardlink_without_network_or_rescan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "asset.bin"
    part = _part(destination)
    content = b"committed"
    source_url = "https://source-a.test/asset.bin"
    current_url = "https://source-b.test/asset.bin"
    part.write_bytes(content)
    sources_module.os.link(part, destination)
    _write_sidecar(destination, source_url, '"asset-v1"', len(content))
    reads: list[Path] = []
    original_open = Path.open

    def count_reads(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if mode == "rb":
            reads.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_reads)
    session = FakeSession(AssertionError("network must not be called"))

    receipt = download_http(
        session,
        current_url,
        destination,
        expected_size=len(content),
        expected_checksums={"sha256": _sha256(content)},
    )

    assert receipt.sha256 == _sha256(content)
    assert reads == [destination]
    assert destination.read_bytes() == content
    assert not part.exists()
    assert not _sidecar(destination).exists()
    assert json.loads(_journal(destination).read_text(encoding="utf-8")) == {
        "schema": 1,
        "sha256": _sha256(content),
        "size": len(content),
        "origin_sha256": _origin_sha256(current_url),
        "asset_identity_sha256": _asset_identity_sha256(_TEST_ASSET_IDENTITY),
    }
    assert session.calls == []


def test_startup_recovers_committed_final_from_journal_without_part_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "asset.bin"
    content = b"committed"
    destination.write_bytes(content)
    _write_journal(destination, content)
    reads: list[Path] = []
    original_open = Path.open

    def count_reads(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if mode == "rb":
            reads.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_reads)
    session = FakeSession(AssertionError("network must not be called"))

    receipt = download_http(session, "https://example.test/asset.bin", destination)

    assert receipt.bytes_written == len(content)
    assert receipt.sha256 == _sha256(content)
    assert reads == [destination]
    assert session.calls == []


def test_committed_journal_from_other_url_is_rejected_without_network(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    content = b"committed"
    destination.write_bytes(content)
    _write_journal(destination, content)
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="journal"):
        download_http(session, "https://other.test/asset.bin", destination)

    assert destination.read_bytes() == content
    assert _journal(destination).exists()
    assert session.calls == []


def test_tampered_final_is_rejected_against_journal_without_cleanup_or_network(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "asset.bin"
    original = b"committed"
    destination.write_bytes(original)
    _write_journal(destination, original)
    destination.write_bytes(b"tampered!")
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="sha256"):
        download_http(session, "https://example.test/asset.bin", destination)

    assert destination.read_bytes() == b"tampered!"
    assert _journal(destination).exists()
    assert session.calls == []


@pytest.mark.parametrize(
    "journal",
    [
        {
            "schema": True,
            "size": 9,
            "sha256": _sha256(b"committed"),
            "origin_sha256": _origin_sha256("https://example.test/asset.bin"),
            "asset_identity_sha256": _asset_identity_sha256(_TEST_ASSET_IDENTITY),
        },
        {
            "schema": 1,
            "size": 9,
            "sha256": _sha256(b"committed"),
            "origin_sha256": _origin_sha256("https://example.test/asset.bin"),
            "asset_identity_sha256": _asset_identity_sha256(_TEST_ASSET_IDENTITY),
            "raw_path": "C:/secret/raw/asset.bin",
        },
        {
            "schema": 1,
            "size": 9,
            "sha256": _sha256(b"committed"),
            "origin_sha256": _origin_sha256("https://example.test/asset.bin"),
        },
    ],
)
def test_invalid_or_nonminimal_journal_is_rejected_and_preserved(
    tmp_path: Path,
    journal: dict[str, object],
) -> None:
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"committed")
    _journal(destination).write_text(json.dumps(journal), encoding="utf-8")
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="journal"):
        download_http(session, "https://example.test/asset.bin", destination)

    assert destination.read_bytes() == b"committed"
    assert json.loads(_journal(destination).read_text(encoding="utf-8")) == journal
    assert session.calls == []


def test_prepared_commit_recovers_from_journal_and_part_without_network(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    content = b"prepared"
    _part(destination).write_bytes(content)
    _write_journal(destination, content)
    session = FakeSession(AssertionError("network must not be called"))

    receipt = download_http(session, "https://example.test/asset.bin", destination)

    assert receipt.sha256 == _sha256(content)
    assert destination.read_bytes() == content
    assert not _part(destination).exists()
    assert _journal(destination).exists()
    assert session.calls == []


def test_prepared_journal_from_other_url_is_rejected_without_network(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    content = b"prepared"
    _part(destination).write_bytes(content)
    _write_journal(destination, content)
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="journal"):
        download_http(session, "https://other.test/asset.bin", destination)

    assert not destination.exists()
    assert _part(destination).read_bytes() == content
    assert _journal(destination).exists()
    assert session.calls == []


def test_prepared_commit_mismatch_fails_closed_and_preserves_evidence(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"tampered")
    _write_journal(destination, b"prepared")
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="size|sha256"):
        download_http(session, "https://example.test/asset.bin", destination)

    assert not destination.exists()
    assert _part(destination).read_bytes() == b"tampered"
    assert _journal(destination).exists()
    assert session.calls == []


def test_stale_journal_without_final_or_part_is_removed_before_download(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    _write_journal(destination, b"stale")
    response = FakeResponse(200, b"fresh", headers={"Content-Length": "5"})
    session = FakeSession(response)

    receipt = download_http(session, "https://example.test/asset.bin", destination)

    assert receipt.sha256 == _sha256(b"fresh")
    assert destination.read_bytes() == b"fresh"
    assert session.calls[0]["headers"] == {"Accept-Encoding": "identity"}
    assert json.loads(_journal(destination).read_text(encoding="utf-8")) == {
        "schema": 1,
        "sha256": _sha256(b"fresh"),
        "size": 5,
        "origin_sha256": _origin_sha256("https://example.test/asset.bin"),
        "asset_identity_sha256": _asset_identity_sha256(_TEST_ASSET_IDENTITY),
    }


@pytest.mark.parametrize("payload", ["cross-url", "malformed"])
def test_orphan_journal_is_deleted_before_parsing_and_download_recovers(
    tmp_path: Path,
    payload: str,
) -> None:
    destination = tmp_path / "asset.bin"
    if payload == "cross-url":
        _write_journal(destination, b"orphan", url="https://other.test/asset.bin")
    else:
        _journal(destination).write_text("{malformed-secret", encoding="utf-8")
    session = FakeSession(FakeResponse(200, b"fresh", headers={"Content-Length": "5"}))

    receipt = download_http(session, "https://example.test/asset.bin", destination)

    assert receipt.sha256 == _sha256(b"fresh")
    assert destination.read_bytes() == b"fresh"
    assert len(session.calls) == 1


def test_orphan_journal_unlink_failure_is_redacted_and_stops_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "asset.bin"
    journal = _journal(destination)
    _write_journal(destination, b"orphan")
    original_unlink = Path.unlink

    def fail_journal_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == journal:
            raise OSError("unlink-credential-secret")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_journal_unlink)
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(DownloadError, match="orphan verified journal cleanup failed") as error:
        download_http(session, "https://example.test/path-secret", destination)

    assert "credential-secret" not in str(error.value)
    assert "path-secret" not in str(error.value)
    assert journal.exists()
    assert session.calls == []


def test_startup_committed_hardlink_rejects_wrong_checksum_without_cleanup(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "asset.bin"
    part = _part(destination)
    content = b"committed"
    part.write_bytes(content)
    sources_module.os.link(part, destination)
    _write_sidecar(destination, "https://example.test/asset.bin", '"asset-v1"', len(content))
    session = FakeSession(AssertionError("network must not be called"))

    with pytest.raises(IntegrityError, match="sha256"):
        download_http(
            session,
            "https://example.test/asset.bin",
            destination,
            expected_size=len(content),
            expected_checksums={"sha256": "0" * 64},
        )

    assert destination.read_bytes() == content
    assert part.exists()
    assert sources_module.os.path.samefile(part, destination)
    assert _sidecar(destination).exists()
    assert session.calls == []


def test_existing_final_mismatch_is_not_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"keep me")
    session = FakeSession(FakeResponse(200, b"replacement"))

    with pytest.raises(IntegrityError):
        download_http(
            session,
            "https://example.test/asset.bin",
            destination,
            expected_size=7,
            expected_checksums={"sha256": _sha256(b"different")},
        )

    assert destination.read_bytes() == b"keep me"
    assert session.calls == []


def test_existing_final_without_verifiable_hash_fails_closed(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"present")
    session = FakeSession(FakeResponse(200, b"replacement"))

    with pytest.raises(IntegrityError, match="checksum"):
        download_http(
            session,
            "https://example.test/asset.bin",
            destination,
            expected_size=7,
        )

    assert destination.read_bytes() == b"present"
    assert session.calls == []


def test_200_resets_old_generation_before_content_length_validation_and_retry(
    tmp_path: Path,
) -> None:
    url = "https://example.test/asset.bin"
    destination = tmp_path / "asset.bin"
    _part(destination).write_bytes(b"old")
    _write_sidecar(destination, url, '"old-generation"', 6)

    with pytest.raises(IntegrityError, match="Content-Length"):
        download_http(
            FakeSession(FakeResponse(200, b"ignored", headers={"Content-Length": "5"})),
            url,
            destination,
            expected_size=6,
        )

    assert _part(destination).read_bytes() == b""
    assert not _sidecar(destination).exists()
    assert not destination.exists()
    assert not _journal(destination).exists()
    retry = FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"}))

    download_http(retry, url, destination, expected_size=6)

    assert destination.read_bytes() == b"abcdef"
    assert "Range" not in retry.calls[0]["headers"]


def test_content_length_disagreement_resets_generation_before_writing(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    response = FakeResponse(200, b"abcdef", headers={"Content-Length": "5"})
    session = FakeSession(response)

    with pytest.raises(IntegrityError, match="length|size"):
        download_http(
            session,
            "https://example.test/asset.bin",
            destination,
            expected_size=6,
            expected_checksums={"sha256": _sha256(b"abcdef")},
        )

    assert not destination.exists()
    assert _part(destination).read_bytes() == b""
    assert not _sidecar(destination).exists()
    assert response.closed is True


def test_redirect_is_not_followed_and_redacts_all_secrets(tmp_path: Path) -> None:
    url = "https://alice:password@example.test/path-secret/file?token=query-secret"
    secret = "api-key-secret"
    response = FakeResponse(302, headers={"Location": "https://other.test/asset"})
    session = FakeSession(response)

    with pytest.raises(DownloadError) as error:
        download_http(
            session,
            url,
            tmp_path / "asset.bin",
            headers={"X-API-Key": secret},
            asset_identity="redirect-private-asset-v1",
        )

    assert session.calls[0]["allow_redirects"] is False
    assert response.closed is True
    assert str(error.value) == "unexpected HTTP status for https://example.test/<redacted>"
    assert "path-secret" not in str(error.value)
    assert "query-secret" not in str(error.value)
    assert secret not in str(error.value)


def test_close_failure_does_not_override_successful_receipt(tmp_path: Path) -> None:
    destination = tmp_path / "asset.bin"
    response = FakeResponse(
        200,
        b"abc",
        headers={"Content-Length": "3"},
        close_error=RuntimeError("close-secret"),
    )

    receipt = download_http(FakeSession(response), "https://example.test/asset.bin", destination)

    assert receipt.path == destination
    assert destination.read_bytes() == b"abc"
    assert response.closed is True


def test_close_failure_does_not_override_primary_error(tmp_path: Path) -> None:
    response = FakeResponse(
        200,
        b"abc",
        headers={"Content-Length": "2"},
        close_error=RuntimeError("close-secret"),
    )

    with pytest.raises(IntegrityError, match="Content-Length") as error:
        download_http(
            FakeSession(response),
            "https://example.test/path-secret",
            tmp_path / "asset.bin",
            expected_size=3,
        )

    assert "close-secret" not in str(error.value)
    assert "path-secret" not in str(error.value)
    assert response.closed is True


def test_request_error_redacts_query_userinfo_and_authorization(tmp_path: Path) -> None:
    url = "https://alice:password@example.test/path-secret?X-Amz-Signature=signed-secret"
    secret = "Bearer authorization-secret"
    session = FakeSession(requests.ConnectionError(f"failed {url} with {secret}"))

    with pytest.raises(DownloadError) as error:
        download_http(
            session,
            url,
            tmp_path / "asset.bin",
            headers={"Authorization": secret},
            asset_identity="request-error-private-asset-v1",
        )

    message = str(error.value)
    assert message == "download failed for https://example.test/<redacted>"
    assert "path-secret" not in message
    assert "signed-secret" not in message
    assert "password" not in message
    assert "authorization-secret" not in message


@pytest.mark.parametrize(
    "etag",
    ['""', '"opaque"', '"back\\slash"', '"obs-\x80"'],
)
def test_strong_etag_accepts_rfc_etagc(etag: str) -> None:
    assert sources_module._strong_etag(etag) is True


@pytest.mark.parametrize(
    "etag",
    [None, "", 'W/"weak"', "unquoted", '"bad"quote"', '"line\nbreak"', '"control\x1f"'],
)
def test_strong_etag_rejects_invalid_syntax(etag: str | None) -> None:
    assert sources_module._strong_etag(etag) is False


def test_publish_capability_failure_happens_before_network_and_cleans_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "asset.bin"
    session = FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"}))
    monkeypatch.setattr(
        sources_module.os,
        "link",
        lambda source, target: (_ for _ in ()).throw(OSError("capability-secret")),
    )

    with pytest.raises(DownloadError, match="hard-link") as error:
        download_http(session, "https://example.test/path-secret", destination)

    assert "capability-secret" not in str(error.value)
    assert "path-secret" not in str(error.value)
    assert session.calls == []
    assert not destination.exists()
    assert not _part(destination).exists()
    assert list(tmp_path.glob(".floodgnn-link-probe-*")) == []


def test_publish_failure_after_successful_probe_keeps_verified_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "asset.bin"
    original_link = sources_module.os.link
    calls = 0

    def probe_then_fail(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_link(source, target)
            return
        raise OSError("commit-link-secret")

    monkeypatch.setattr(sources_module.os, "link", probe_then_fail)

    with pytest.raises(DownloadError, match="publish") as error:
        download_http(
            FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
            "https://example.test/path-secret",
            destination,
        )

    assert "commit-link-secret" not in str(error.value)
    assert "path-secret" not in str(error.value)
    assert not destination.exists()
    assert _part(destination).read_bytes() == b"abcdef"
    assert json.loads(_journal(destination).read_text(encoding="utf-8")) == {
        "schema": 1,
        "sha256": _sha256(b"abcdef"),
        "size": 6,
        "origin_sha256": _origin_sha256("https://example.test/path-secret"),
        "asset_identity_sha256": _asset_identity_sha256(_TEST_ASSET_IDENTITY),
    }


def test_verified_journal_is_minimal_and_contains_no_request_secrets(tmp_path: Path) -> None:
    url = "https://alice:password@example.test/path-secret?token=query-secret"
    secret = "Bearer authorization-secret"
    asset_identity = "private-secret-free-asset-v1"
    destination = tmp_path / "asset.bin"

    download_http(
        FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
        url,
        destination,
        headers={"Authorization": secret},
        asset_identity=asset_identity,
    )

    journal_text = _journal(destination).read_text(encoding="utf-8")
    assert json.loads(journal_text) == {
        "schema": 1,
        "sha256": _sha256(b"abcdef"),
        "size": 6,
        "origin_sha256": _origin_sha256(url),
        "asset_identity_sha256": _asset_identity_sha256(asset_identity),
    }
    for forbidden in (url, "password", "path-secret", "query-secret", secret, asset_identity):
        assert forbidden not in journal_text


def test_publish_auxiliary_cleanup_failure_keeps_committed_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "asset.bin"
    part = _part(destination)
    sidecar = _sidecar(destination)
    original_unlink = Path.unlink

    def fail_auxiliary_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if destination.exists() and path in {part, sidecar}:
            raise OSError("simulated auxiliary cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_auxiliary_cleanup)

    receipt = download_http(
        FakeSession(
            FakeResponse(
                200,
                b"abcdef",
                headers={"Content-Length": "6", "ETag": '"asset-v1"'},
            )
        ),
        "https://example.test/asset.bin",
        destination,
        expected_size=6,
        expected_checksums={"sha256": _sha256(b"abcdef")},
    )

    assert receipt.path == destination
    assert destination.read_bytes() == b"abcdef"
    assert part.exists()
    assert sidecar.exists()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    recovered = download_http(
        FakeSession(AssertionError("network must not be called")),
        "https://example.test/asset.bin",
        destination,
        expected_size=6,
        expected_checksums={"sha256": _sha256(b"abcdef")},
    )

    assert recovered.sha256 == _sha256(b"abcdef")
    assert not part.exists()
    assert not sidecar.exists()


def test_concurrent_final_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "asset.bin"

    original_link = sources_module.os.link

    def concurrent_create(source: Path, target: Path) -> None:
        if target == destination:
            destination.write_bytes(b"concurrent")
            raise FileExistsError
        original_link(source, target)

    monkeypatch.setattr(sources_module.os, "link", concurrent_create)

    with pytest.raises(IntegrityError):
        download_http(
            FakeSession(FakeResponse(200, b"downloaded", headers={"Content-Length": "10"})),
            "https://example.test/asset.bin",
            destination,
            expected_size=10,
            expected_checksums={"sha256": _sha256(b"downloaded")},
        )

    assert destination.read_bytes() == b"concurrent"


def test_stream_and_sidecar_are_fsynced_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(sources_module.os, "fsync", calls.append)

    download_http(
        FakeSession(
            FakeResponse(
                200,
                b"abcdef",
                headers={"Content-Length": "6", "ETag": '"asset-v1"'},
            )
        ),
        "https://example.test/asset.bin",
        tmp_path / "asset.bin",
    )

    assert len(calls) >= 2


def test_fsync_failure_cannot_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "asset.bin"
    monkeypatch.setattr(
        sources_module.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    with pytest.raises((DownloadError, OSError), match="fsync|download"):
        download_http(
            FakeSession(FakeResponse(200, b"abcdef", headers={"Content-Length": "6"})),
            "https://example.test/asset.bin",
            destination,
        )

    assert not destination.exists()


def test_directory_sync_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sources_module.os, "name", "posix")
    monkeypatch.setattr(sources_module.os, "open", lambda path, flags: 42)
    monkeypatch.setattr(
        sources_module.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError("directory fsync failed")),
    )
    monkeypatch.setattr(
        sources_module.os,
        "close",
        lambda descriptor: (_ for _ in ()).throw(OSError("directory close failed")),
    )

    sources_module._best_effort_sync_directory(tmp_path)


def test_sha256_and_md5_validation_scans_completed_part_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "asset.bin"
    content = b"abcdef"
    reads: list[Path] = []
    original_open = Path.open

    def count_reads(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if mode == "rb":
            reads.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_reads)

    download_http(
        FakeSession(FakeResponse(200, content, headers={"Content-Length": "6"})),
        "https://example.test/asset.bin",
        destination,
        expected_size=6,
        expected_checksums={"sha256": _sha256(content), "md5": _md5(content)},
    )

    assert reads == [_part(destination)]
