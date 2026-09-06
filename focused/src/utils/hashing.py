"""Deterministic fingerprinting helpers.

Two things are hashed by this project:

1. ``sha256_bytes(image_bytes)`` - the raw bytes of the matched candidate image.
2. ``sha256_canonical_json(metadata)`` - a canonical JSON serialisation of the
   normalised post record (sorted keys, no insignificant whitespace, UTF-8).

The canonical form guarantees that two logically identical records always
produce the same digest, which is what makes on-chain read-back verification
meaningful.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("sha256_bytes expects bytes")
    return hashlib.sha256(bytes(data)).hexdigest()


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, compact separators, UTF-8 safe."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical_json(obj: Any) -> str:
    """Hex SHA-256 of the canonical JSON form of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def to_bytes32(hex_digest: str) -> bytes:
    """Convert a 64-char hex digest to 32 raw bytes (for ``bytes32`` on-chain)."""
    clean = hex_digest[2:] if hex_digest.startswith("0x") else hex_digest
    if len(clean) != 64:
        raise ValueError(f"expected a 64-char sha256 hex digest, got {len(clean)} chars")
    return bytes.fromhex(clean)


def hex_to_bytes32(hex_hash: str) -> bytes:
    """Convert a 64-char hex SHA-256 digest to 32 bytes for Solidity bytes32."""
    clean = hex_hash.removeprefix("0x")
    return bytes.fromhex(clean)


def generate_fingerprint(canonical_string: str) -> str:
    """Compute the SHA-256 hex digest of *canonical_string*."""
    return hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()


def from_bytes32(raw: bytes) -> str:
    """Convert 32 raw bytes read back from chain into a hex digest string."""
    if len(raw) != 32:
        raise ValueError(f"expected 32 bytes, got {len(raw)}")
    return raw.hex()
