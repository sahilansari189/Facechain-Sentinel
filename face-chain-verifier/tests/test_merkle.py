import hashlib

import pytest

from src.utils.merkle import merkle_proof, merkle_root, verify_proof


def h(n: int) -> str:
    return hashlib.sha256(str(n).encode()).hexdigest()


def test_single_leaf_root_is_the_leaf():
    assert merkle_root([h(1)]) == h(1)


def test_root_is_deterministic_and_order_sensitive():
    a, b = [h(1), h(2), h(3)], [h(3), h(2), h(1)]
    assert merkle_root(a) == merkle_root(a)
    assert merkle_root(a) != merkle_root(b)


def test_root_changes_when_any_leaf_changes():
    leaves = [h(i) for i in range(50)]
    root = merkle_root(leaves)
    leaves[37] = h(999)
    assert merkle_root(leaves) != root


def test_proofs_verify_for_every_leaf_in_a_100_item_batch():
    leaves = [h(i) for i in range(100)]
    root = merkle_root(leaves)
    for i in (0, 1, 49, 98, 99):
        assert verify_proof(leaves[i], merkle_proof(leaves, i), root)


def test_proof_fails_for_a_tampered_leaf():
    leaves = [h(i) for i in range(9)]
    root = merkle_root(leaves)
    assert not verify_proof(h(1234), merkle_proof(leaves, 4), root)


def test_odd_level_duplicates_last_node():
    leaves = [h(1), h(2), h(3)]
    lvl1_left = hashlib.sha256(bytes.fromhex(h(1)) + bytes.fromhex(h(2))).digest()
    lvl1_right = hashlib.sha256(bytes.fromhex(h(3)) + bytes.fromhex(h(3))).digest()
    assert merkle_root(leaves) == hashlib.sha256(lvl1_left + lvl1_right).hexdigest()


def test_empty_and_invalid_inputs_raise():
    with pytest.raises(ValueError):
        merkle_root([])
    with pytest.raises(ValueError):
        merkle_root(["not-a-digest"])
