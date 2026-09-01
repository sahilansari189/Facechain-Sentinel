"""Merkle tree over SHA-256 hex digests.

A batch of N verified posts is anchored with a single 32-byte root, which makes
on-chain storage cost O(1) per batch instead of O(N) while still allowing any
individual record to be proven later with a log2(N) proof.

Convention (kept deliberately simple and language-agnostic):
  * leaves are the raw 32 bytes of each record's sha256 digest, in order
  * an odd node at any level is promoted by duplicating itself
  * parent = sha256(left || right)
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple


def _to_raw(digest: str) -> bytes:
    clean = digest[2:] if digest.startswith("0x") else digest
    if len(clean) != 64:
        raise ValueError(f"expected a 64-char sha256 hex digest, got {len(clean)} chars")
    return bytes.fromhex(clean)


def _pair(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(left + right).digest()


def merkle_levels(digests: Sequence[str]) -> List[List[bytes]]:
    """All levels of the tree, leaves first, root last."""
    if not digests:
        raise ValueError("cannot build a merkle tree over zero leaves")
    level = [_to_raw(d) for d in digests]
    levels = [level]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(_pair(left, right))
        level = nxt
        levels.append(level)
    return levels


def merkle_root(digests: Sequence[str]) -> str:
    """Hex root of the merkle tree built over ``digests``."""
    return merkle_levels(digests)[-1][0].hex()


def merkle_proof(digests: Sequence[str], index: int) -> List[Tuple[str, str]]:
    """Inclusion proof for ``index`` as a list of ``(side, hex_hash)`` siblings."""
    levels = merkle_levels(digests)
    if not 0 <= index < len(levels[0]):
        raise IndexError(f"leaf index {index} out of range (0..{len(levels[0]) - 1})")
    proof: List[Tuple[str, str]] = []
    idx = index
    for level in levels[:-1]:
        sibling_idx = idx + 1 if idx % 2 == 0 else idx - 1
        sibling = level[sibling_idx] if sibling_idx < len(level) else level[idx]
        proof.append(("right" if idx % 2 == 0 else "left", sibling.hex()))
        idx //= 2
    return proof


def verify_proof(leaf: str, proof: Sequence[Tuple[str, str]], root: str) -> bool:
    """Recompute the root from a leaf + proof and compare it with ``root``."""
    try:
        acc = _to_raw(leaf)
        for side, sibling_hex in proof:
            sibling = _to_raw(sibling_hex)
            acc = _pair(acc, sibling) if side == "right" else _pair(sibling, acc)
    except ValueError:
        return False
    return acc.hex().lower() == (root[2:] if root.startswith("0x") else root).lower()
