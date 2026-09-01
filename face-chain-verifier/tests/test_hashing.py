import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.hashing import (  # noqa: E402
    canonical_json,
    from_bytes32,
    sha256_bytes,
    sha256_canonical_json,
    to_bytes32,
)


def test_sha256_bytes_matches_hashlib():
    data = b"hello face chain"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_bytes_rejects_non_bytes():
    with pytest.raises(TypeError):
        sha256_bytes("not bytes")  # type: ignore[arg-type]


def test_canonical_json_is_key_order_independent():
    a = {"b": 1, "a": {"y": 2, "x": 3}}
    b = {"a": {"x": 3, "y": 2}, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert sha256_canonical_json(a) == sha256_canonical_json(b)


def test_canonical_json_is_compact():
    assert canonical_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_different_metadata_yields_different_hash():
    base = {"source_url": "https://example.com/a", "title": "x"}
    other = {"source_url": "https://example.com/b", "title": "x"}
    assert sha256_canonical_json(base) != sha256_canonical_json(other)


def test_bytes32_roundtrip():
    digest = sha256_bytes(b"payload")
    raw = to_bytes32(digest)
    assert len(raw) == 32
    assert from_bytes32(raw) == digest
    assert to_bytes32("0x" + digest) == raw


def test_to_bytes32_rejects_bad_length():
    with pytest.raises(ValueError):
        to_bytes32("deadbeef")
