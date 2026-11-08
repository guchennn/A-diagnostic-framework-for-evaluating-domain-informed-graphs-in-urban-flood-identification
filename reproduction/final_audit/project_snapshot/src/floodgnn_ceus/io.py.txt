"""Deterministic byte and file hashing helpers."""

from __future__ import annotations

import hashlib
import json
import math
from os import PathLike
from pathlib import Path


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON floats must be finite")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("canonical JSON object keys must be strings")
            _validate_json_value(item)
        return
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON data to its protocol-canonical UTF-8 representation."""
    _validate_json_value(value)
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 hex digest for bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | PathLike[str]) -> str:
    """Return the SHA-256 digest of a file without loading it in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
