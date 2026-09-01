"""Face similarity + candidate ranking (pure functions, unit-testable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]. Inputs need not be pre-normalised."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.size == 0 or vb.size == 0:
        raise ValueError("empty embedding")
    if va.shape != vb.shape:
        raise ValueError(f"embedding dimension mismatch: {va.shape} vs {vb.shape}")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        raise ValueError("zero-norm embedding")
    return float(np.dot(va, vb) / (na * nb))


@dataclass
class ScoredCandidate:
    candidate: object                  # search.reverse_search.Candidate
    similarity: float
    faces_found: int = 0
    error: Optional[str] = None
    image_bytes: Optional[bytes] = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None and self.faces_found > 0


def best_similarity(reference: Sequence[float], embeddings: Iterable[Sequence[float]]) -> float:
    """Max cosine similarity of a reference embedding against several faces."""
    best = -1.0
    for emb in embeddings:
        best = max(best, cosine_similarity(reference, emb))
    return best


def rank_candidates(scored: List[ScoredCandidate]) -> List[ScoredCandidate]:
    """Sort by similarity descending; failed candidates always sink to the end."""
    return sorted(scored, key=lambda s: (s.ok, s.similarity), reverse=True)


def select_match(scored: List[ScoredCandidate], threshold: float) -> Optional[ScoredCandidate]:
    """Return the strongest valid candidate at/above the threshold, else None."""
    ranked = rank_candidates(scored)
    for item in ranked:
        if item.ok and item.similarity >= threshold:
            return item
    return None


def similarity_matrix(reference: Sequence[float], embeddings: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorised cosine similarity of one reference against many embeddings.

    Far faster than looping :func:`cosine_similarity` when a batch produces
    thousands of candidate faces.
    """
    if len(embeddings) == 0:
        return np.zeros((0,), dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64).ravel()
    mat = np.asarray(embeddings, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[1] != ref.shape[0]:
        raise ValueError(f"embedding dimension mismatch: {mat.shape[1]} vs {ref.shape[0]}")
    rn = float(np.linalg.norm(ref))
    norms = np.linalg.norm(mat, axis=1)
    if rn == 0.0 or float(norms.min()) == 0.0:
        raise ValueError("zero-norm embedding")
    return (mat @ ref) / (norms * rn)


def best_similarity_fast(reference: Sequence[float], embeddings: Sequence[Sequence[float]]) -> float:
    """Max cosine similarity, computed with a single matrix product."""
    sims = similarity_matrix(reference, embeddings)
    return float(sims.max()) if sims.size else -1.0
