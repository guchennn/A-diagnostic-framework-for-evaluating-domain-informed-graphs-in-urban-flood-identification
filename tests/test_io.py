from __future__ import annotations

import json

import pytest

from floodgnn_ceus.io import canonical_json_bytes, sha256_bytes, sha256_file


def test_canonical_json_is_sorted_compact_utf8_and_hashable(tmp_path) -> None:
    value = {"z": [True, None, 2.5], "a": "郑州"}
    encoded = canonical_json_bytes(value)
    path = tmp_path / "value.json"
    path.write_bytes(encoded)

    assert encoded == '{"a":"郑州","z":[true,null,2.5]}'.encode()
    assert json.loads(encoded) == value
    assert sha256_bytes(encoded) == sha256_file(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "value"}, ("tuple",)])
def test_canonical_json_rejects_non_protocol_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_json_bytes(value)


def test_file_hash_streams_the_same_bytes(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    payload = b"FloodGNN" * 200_000
    path.write_bytes(payload)

    assert sha256_file(path) == sha256_bytes(payload)
