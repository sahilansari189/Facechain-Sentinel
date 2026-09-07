import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.face.matcher import (  # noqa: E402
    ScoredCandidate,
    best_similarity,
    cosine_similarity,
    rank_candidates,
    select_match,
)
from src.search.reverse_search import Candidate, dedupe  # noqa: E402


def test_cosine_identical_vectors():
    v = [0.1, 0.2, 0.3, 0.4]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_and_opposite():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a * 7.5) == pytest.approx(1.0)


def test_cosine_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2, 3], [1, 2])


def test_cosine_zero_norm():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 1])


def test_best_similarity_picks_max():
    ref = [1.0, 0.0]
    assert best_similarity(ref, [[0.0, 1.0], [1.0, 0.1], [-1.0, 0.0]]) == pytest.approx(
        cosine_similarity(ref, [1.0, 0.1])
    )


def _c(url):
    return Candidate(url=url, image_url=url + "/img.jpg", source="example.com")


def test_ranking_orders_by_similarity_and_sinks_failures():
    items = [
        ScoredCandidate(_c("https://a"), 0.34, faces_found=1),
        ScoredCandidate(_c("https://b"), -1.0, error="no face"),
        ScoredCandidate(_c("https://c"), 0.76, faces_found=2),
        ScoredCandidate(_c("https://d"), 0.48, faces_found=1),
    ]
    ranked = rank_candidates(items)
    assert [r.candidate.url for r in ranked[:3]] == ["https://c", "https://d", "https://a"]
    assert ranked[-1].error == "no face"


def test_select_match_respects_threshold():
    items = [
        ScoredCandidate(_c("https://a"), 0.34, faces_found=1),
        ScoredCandidate(_c("https://c"), 0.76, faces_found=1),
    ]
    assert select_match(items, 0.45).candidate.url == "https://c"
    assert select_match(items, 0.9) is None


def test_select_match_ignores_failed_candidates_even_with_high_score():
    items = [ScoredCandidate(_c("https://x"), 0.99, faces_found=0, error="download failed")]
    assert select_match(items, 0.45) is None


def test_dedupe_removes_duplicate_pages_and_images():
    cands = [
        Candidate(url="https://ex.com/p/1", image_url="https://cdn/1.jpg"),
        Candidate(url="https://ex.com/p/1/", image_url="https://cdn/2.jpg"),
        Candidate(url="https://ex.com/p/2", image_url="https://cdn/1.jpg"),
        Candidate(url="https://ex.com/p/3", image_url="https://cdn/3.jpg"),
    ]
    assert [c.url for c in dedupe(cands)] == ["https://ex.com/p/1", "https://ex.com/p/3"]
