"""Batch-mode unit tests. No network, no models: the heavy parts are stubbed."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.batch import BatchRunner, _result_from_dict, discover_images, write_reports
from src.face.matcher import best_similarity_fast, cosine_similarity, similarity_matrix
from src.pipeline import STATUS_MATCHED, CandidateCache, ImageResult, PipelineRunner
from src.search.reverse_search import Candidate


# --------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------- #

def test_discover_images_filters_and_recurses(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.jpg").write_bytes(b"1")
    (tmp_path / "b.PNG").write_bytes(b"2")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub" / "c.jpeg").write_bytes(b"3")

    found = discover_images([str(tmp_path)])
    names = sorted(p.name for p in found)
    assert names == ["a.jpg", "b.PNG", "c.jpeg"]

    shallow = discover_images([str(tmp_path)], recursive=False)
    assert "c.jpeg" not in {p.name for p in shallow}


def test_discover_images_deduplicates_paths(tmp_path: Path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"1")
    assert len(discover_images([str(f), str(f), str(tmp_path)])) == 1


# --------------------------------------------------------------------- #
# vectorised matching
# --------------------------------------------------------------------- #

def test_similarity_matrix_matches_scalar_implementation():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=512)
    embs = rng.normal(size=(64, 512))
    fast = similarity_matrix(ref, embs)
    slow = [cosine_similarity(ref, e) for e in embs]
    assert np.allclose(fast, slow, atol=1e-9)
    assert best_similarity_fast(ref, embs) == pytest.approx(max(slow))


def test_similarity_matrix_handles_empty_input():
    assert best_similarity_fast(np.ones(4), []) == -1.0


# --------------------------------------------------------------------- #
# candidate cache
# --------------------------------------------------------------------- #

def test_candidate_cache_computes_each_url_once():
    calls = []

    def compute(url):
        calls.append(url)
        return np.ones((1, 4), dtype=np.float32), b"bytes"

    cache = CandidateCache()
    for _ in range(5):
        cache.get_or_compute("https://x/a.jpg", compute)
    cache.get_or_compute("https://x/b.jpg", compute)
    assert calls == ["https://x/a.jpg", "https://x/b.jpg"]
    assert cache.stats() == {"entries": 2, "hits": 4, "misses": 2}


def test_candidate_cache_remembers_failures_without_raising():
    def boom(url):
        raise RuntimeError("404")

    cache = CandidateCache()
    embeddings, data, err = cache.get_or_compute("https://x/bad.jpg", boom)
    assert embeddings is None and data is None and "404" in err
    cache.get_or_compute("https://x/bad.jpg", boom)
    assert cache.stats()["misses"] == 1


# --------------------------------------------------------------------- #
# pipeline (stubbed detector + provider)
# --------------------------------------------------------------------- #

class FakeFace:
    def __init__(self, emb):
        self.embedding = np.asarray(emb, dtype=np.float32)
        self.bbox = [0, 0, 10, 10]
        self.det_score = 0.9

    @property
    def area(self):
        return 100.0


class FakeDetector:
    def __init__(self, mapping, reference):
        self.mapping = mapping
        self.reference = reference

    def load_image(self, path):
        return path

    def detect(self, image):
        return [FakeFace(self.reference)]

    def detect_bytes(self, data):
        return [FakeFace(self.mapping[data])]

    def primary_face(self, faces):
        return faces[0]


class FakePool:
    size = 1

    def __init__(self, detector):
        self.detector = detector

    def warm(self):
        return self.detector

    class _Lease:
        def __init__(self, d):
            self.d = d

        def __enter__(self):
            return self.d

        def __exit__(self, *a):
            return False

    def lease(self):
        return FakePool._Lease(self.detector)


class FakeProvider:
    name = "fake"

    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0

    def search(self, image_url, limit=20):
        self.calls += 1
        return self.candidates


class FakeConfig:
    match_threshold = 0.45
    max_candidates = 10
    max_image_bytes = 1024
    http_timeout = 5
    image_upload_url = "https://upload.test"
    search_provider = "fake"
    face_model = "buffalo_l"
    face_det_size = 320
    batch_workers = 2
    face_pool_size = 1
    batch_chunk_size = 25


@pytest.fixture
def stubbed(monkeypatch):
    ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    match_emb = np.array([0.99, 0.14, 0.0, 0.0], dtype=np.float32)
    other_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    payloads = {b"match": match_emb, b"other": other_emb}

    candidates = [
        Candidate(url="https://site.test/other", title="Other", image_url="https://img/other.jpg"),
        Candidate(url="https://site.test/post", title="Post", image_url="https://img/match.jpg"),
    ]

    def fake_download(url, max_bytes, timeout=15):
        return b"match" if "match" in url else b"other"

    monkeypatch.setattr("src.pipeline.download_image", fake_download)
    monkeypatch.setattr("src.pipeline.fetch_page_metadata", lambda url, timeout=15: ({"title": "T"}, None))
    monkeypatch.setattr("src.pipeline.upload_image_for_search", lambda p, u, t: "https://upload.test/q.jpg")

    pool = FakePool(FakeDetector(payloads, ref))
    runner = PipelineRunner(FakeConfig(), pool, FakeProvider(candidates), verbose=False)
    return runner


def test_pipeline_matches_and_fingerprints(stubbed, tmp_path):
    img = tmp_path / "in.jpg"
    img.write_bytes(b"input")
    res = stubbed.run(str(img))
    assert res.status == STATUS_MATCHED
    assert res.source_url == "https://site.test/post"
    assert res.similarity > 0.9
    assert len(res.data_hash) == 64 and len(res.image_hash) == 64
    assert res.candidates_found == 2


def test_pipeline_is_deterministic_for_the_same_record(stubbed, tmp_path):
    img = tmp_path / "in.jpg"
    img.write_bytes(b"input")
    a = stubbed.run(str(img))
    b = stubbed.run(str(img))
    a.record["retrieved_at"] = b.record["retrieved_at"] = "2026-01-01T00:00:00+00:00"
    from src.utils.hashing import sha256_canonical_json

    assert sha256_canonical_json(a.record) == sha256_canonical_json(b.record)


def test_pipeline_reports_no_match_below_threshold(stubbed, tmp_path):
    stubbed.cfg.match_threshold = 0.999999
    img = tmp_path / "in.jpg"
    img.write_bytes(b"input")
    res = stubbed.run(str(img))
    assert res.status == "no_match" and not res.data_hash


def test_pipeline_never_raises_when_a_stage_explodes(stubbed, tmp_path, monkeypatch):
    img = tmp_path / "in.jpg"
    img.write_bytes(b"input")
    monkeypatch.setattr("src.pipeline.fetch_page_metadata",
                        lambda url, timeout=15: (_ for _ in ()).throw(RuntimeError("boom")))
    res = stubbed.run(str(img))
    assert res.status == "error" and "boom" in res.error


def test_pipeline_reports_no_face(stubbed, tmp_path, monkeypatch):
    monkeypatch.setattr(stubbed.pool.detector, "detect", lambda image: [])
    img = tmp_path / "in.jpg"
    img.write_bytes(b"input")
    res = stubbed.run(str(img))
    assert res.status == "no_face" and res.error


# --------------------------------------------------------------------- #
# batch runner
# --------------------------------------------------------------------- #

def test_batch_deduplicates_identical_inputs_and_reports_every_file(stubbed, tmp_path, monkeypatch):
    for name in ("a.jpg", "b.jpg", "copy.jpg"):
        (tmp_path / name).write_bytes(b"same-bytes" if name != "b.jpg" else b"other-bytes")
    images = discover_images([str(tmp_path)])

    runner = BatchRunner(FakeConfig(), stubbed.provider, workers=2, checkpoint=tmp_path / "ck.jsonl")
    runner.runner = stubbed
    runner.pool = stubbed.pool

    results = runner.run(images)
    assert len(results) == 3                      # every input file appears in the report
    assert runner.stats.duplicates == 1           # but only two were actually processed
    assert stubbed.provider.calls == 2
    assert all(r.matched for r in results)


def test_batch_checkpoint_supports_resume(stubbed, tmp_path):
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    images = discover_images([str(tmp_path)])
    ck = tmp_path / "ck.jsonl"

    runner = BatchRunner(FakeConfig(), stubbed.provider, workers=2, checkpoint=ck)
    runner.runner, runner.pool = stubbed, stubbed.pool
    runner.run(images)
    assert len(ck.read_text().strip().splitlines()) == 2

    calls_before = stubbed.provider.calls
    runner2 = BatchRunner(FakeConfig(), stubbed.provider, workers=2, checkpoint=ck)
    runner2.runner, runner2.pool = stubbed, stubbed.pool
    results = runner2.run(images, resume=True)
    assert stubbed.provider.calls == calls_before   # nothing re-processed
    assert runner2.stats.skipped_resume == 2
    assert len(results) == 2 and all(r.matched for r in results)


def test_write_reports_emits_json_and_csv(tmp_path):
    results = [
        ImageResult(image_path="a.jpg", status=STATUS_MATCHED, similarity=0.8, data_hash="a" * 64),
        ImageResult(image_path="b.jpg", status="no_match", error="below threshold"),
    ]
    paths = write_reports(results, tmp_path / "out", {"images": 2, "matched": 1})
    payload = json.loads(paths["json"].read_text())
    assert payload["summary"]["matched"] == 1 and len(payload["results"]) == 2
    csv_text = paths["csv"].read_text()
    assert "image_path,status" in csv_text and "no_match" in csv_text


def test_result_roundtrips_through_dict():
    original = ImageResult(image_path="a.jpg", status=STATUS_MATCHED, similarity=0.77,
                           data_hash="b" * 64, source_url="https://x/y")
    clone = _result_from_dict(original.as_dict())
    assert clone.matched and clone.similarity == pytest.approx(0.77)
    assert clone.source_url == original.source_url
