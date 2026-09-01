"""Blockchain record parsing / verification tests - fully mocked, no network."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blockchain.contract import ChainError  # noqa: E402
from src.blockchain.verifier import BlockchainVerifier, parse_record  # noqa: E402
from src.utils.hashing import sha256_bytes, to_bytes32  # noqa: E402

DATA_HASH = sha256_bytes(b"post-metadata")
IMAGE_HASH = sha256_bytes(b"image-bytes")
ADDR = "0x000000000000000000000000000000000000dEaD"


def test_parse_get_record_tuple():
    raw = (to_bytes32(DATA_HASH), to_bytes32(IMAGE_HASH), "https://example.com/p/1", 1735689600, ADDR)
    rec = parse_record(raw)
    assert rec.data_hash == DATA_HASH
    assert rec.image_hash == IMAGE_HASH
    assert rec.source_url == "https://example.com/p/1"
    assert rec.timestamp == 1735689600
    assert rec.submitter == ADDR


def test_parse_get_latest_by_hash_tuple():
    raw = (to_bytes32(IMAGE_HASH), "https://example.com/p/2", 1735689601, ADDR, 7)
    rec = parse_record(raw, data_hash=DATA_HASH)
    assert rec.record_id == 7
    assert rec.data_hash == DATA_HASH
    assert rec.image_hash == IMAGE_HASH


def test_parse_record_rejects_bad_shape():
    with pytest.raises(ChainError):
        parse_record((1, 2, 3))
    with pytest.raises(ChainError):
        parse_record(None)


def test_compare_verified_and_tampered():
    assert BlockchainVerifier.compare(DATA_HASH, DATA_HASH)
    assert BlockchainVerifier.compare("0x" + DATA_HASH, DATA_HASH.upper())
    assert not BlockchainVerifier.compare(DATA_HASH, IMAGE_HASH)


def test_record_as_dict_roundtrip():
    raw = (to_bytes32(DATA_HASH), to_bytes32(IMAGE_HASH), "https://x.test/1", 42, ADDR)
    d = parse_record(raw).as_dict()
    assert d["data_hash"] == DATA_HASH and d["timestamp"] == 42
