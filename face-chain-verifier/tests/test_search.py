"""Search-layer tests with mocked HTTP - no live API calls."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search.candidate_fetcher import build_post_record, is_safe_url, platform_name  # noqa: E402
from src.search.reverse_search import (  # noqa: E402
    Candidate,
    SearchError,
    SerpApiGoogleLensProvider,
    dedupe,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def test_serpapi_parses_visual_matches():
    payload = {
        "visual_matches": [
            {"link": "https://site.test/a", "title": "A", "source": "site.test", "thumbnail": "https://cdn/a.jpg"},
            {"link": "https://site.test/b", "title": "B", "source": "site.test", "image": "https://cdn/b.jpg"},
        ]
    }
    p = SerpApiGoogleLensProvider(api_key="k")
    with patch("src.search.reverse_search.requests.request", return_value=FakeResponse(payload=payload)):
        results = p.search("https://img.test/q.jpg", limit=10)
    assert [r.url for r in results] == ["https://site.test/a", "https://site.test/b"]
    assert results[1].image_url == "https://cdn/b.jpg"


def test_serpapi_reports_api_error_instead_of_faking_results():
    p = SerpApiGoogleLensProvider(api_key="k")
    with patch(
        "src.search.reverse_search.requests.request",
        return_value=FakeResponse(payload={"error": "no results found"}),
    ):
        with pytest.raises(SearchError):
            p.search("https://img.test/q.jpg")


def test_rate_limit_surfaces_clearly():
    p = SerpApiGoogleLensProvider(api_key="k")
    with patch("src.search.reverse_search.requests.request", return_value=FakeResponse(status_code=429, text="slow down")):
        with pytest.raises(SearchError, match="rate limited"):
            p.search("https://img.test/q.jpg")


def test_missing_api_key_raises():
    with pytest.raises(SearchError):
        SerpApiGoogleLensProvider(api_key="").search("https://img.test/q.jpg")


def test_is_safe_url_blocks_local_and_non_http():
    assert not is_safe_url("http://127.0.0.1/x.jpg")
    assert not is_safe_url("file:///etc/passwd")
    assert not is_safe_url("ftp://example.com/x.jpg")


def test_platform_name_detection():
    assert platform_name("https://www.instagram.com/p/abc/") == "Instagram"
    assert platform_name("https://x.com/user/status/1") == "X"
    assert platform_name("https://blog.example.org/post") == "blog.example.org"


def test_build_post_record_shape():
    cand = Candidate(url="https://www.instagram.com/p/abc/", title="t", image_url="https://cdn/i.jpg")
    rec = build_post_record(cand, {"title": "Real title", "description": "caption"})
    assert rec["platform"] == "Instagram"
    assert rec["title"] == "Real title"
    assert rec["description"] == "caption"
    assert rec["retrieved_at"].endswith("+00:00")


def test_dedupe_collapses_social_url_variants_but_keeps_distinct_posts():
    candidates = [
        Candidate(
            url="https://www.instagram.com/p/ABC123/?utm_source=lens",
            image_url="https://cdn.test/one.jpg",
        ),
        Candidate(
            url="https://instagram.com/p/ABC123/?img_index=1",
            image_url="https://cdn.test/one.jpg",
        ),
        Candidate(
            url="https://www.instagram.com/p/XYZ789/",
            image_url="https://cdn.test/two.jpg",
        ),
    ]

    result = dedupe(candidates)

    assert [candidate.url for candidate in result] == [
        "https://www.instagram.com/p/ABC123/?utm_source=lens",
        "https://www.instagram.com/p/XYZ789/",
    ]
