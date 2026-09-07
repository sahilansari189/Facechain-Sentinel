"""High-throughput batch mode: 50-100+ images through the full pipeline.

Design notes (why this is fast):

* **One model load, N sessions.** The InsightFace model is loaded once into a
  small :class:`DetectorPool`; workers check sessions out instead of re-loading
  ~300 MB per image.
* **I/O parallelism.** Reverse-image search, candidate downloads and metadata
  fetches are network-bound, so images are processed on a thread pool. The GIL
  is released during requests and during ONNX inference.
* **Batch-wide candidate cache.** Candidate URLs repeat heavily across a batch;
  each URL is downloaded and embedded exactly once.
* **Input de-duplication.** Byte-identical input images are detected by SHA-256
  and processed once.
* **Vectorised matching.** Similarity is a single matrix product per candidate.
* **One transaction per chunk.** Verified fingerprints are anchored with
  ``storeBatch`` (default 25 per tx), amortising the 21k base gas cost and
  cutting on-chain time/cost by roughly an order of magnitude.
* **Resumable.** Every finished image is appended to a JSONL checkpoint, so
  ``--resume`` skips work already done after an interruption.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .face.detector import DetectorPool
from .pipeline import STATUS_MATCHED, CandidateCache, ImageResult, PipelineRunner
from .utils.hashing import sha256_bytes
from .utils.logging_ui import banner, fail, kv, ok, step, warn

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_images(paths: Iterable[str], recursive: bool = True) -> List[Path]:
    """Expand files, directories and globs into a sorted list of image paths."""
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            found.extend(f for f in it if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.is_file():
            found.append(p)
        else:  # treat as a glob pattern
            base = Path(p.anchor or ".")
            pattern = str(p) if not p.anchor else str(p.relative_to(base))
            found.extend(
                f for f in base.glob(pattern) if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
            )
    uniq = sorted({f.resolve() for f in found})
    return uniq


@dataclass
class BatchStats:
    total: int = 0
    processed: int = 0
    matched: int = 0
    duplicates: int = 0
    skipped_resume: int = 0
    failures: Dict[str, int] = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def note(self, status: str) -> None:
        if status != STATUS_MATCHED:
            self.failures[status] = self.failures.get(status, 0) + 1



class PersistentSearchCache:
    """Disk-backed cache for reverse-image-search candidate lists.

    The key is the SHA-256 digest of the original input image. This cache
    survives process restarts and is shared by all workers in the batch.
    """

    CACHE_VERSION = 2

    def __init__(self, root: Path = Path("cache") / "searches"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def digest_file(path: str | Path) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _path(self, digest: str) -> Path:
        return self.root / f"{digest}.json"

    def get(self, digest: str):
        path = self._path(digest)
        with self._lock:
            if not path.is_file():
                self.misses += 1
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("version") != self.CACHE_VERSION:
                    raise ValueError("stale cache version")
                candidates = payload.get("candidates")
                if not isinstance(candidates, list):
                    raise ValueError("invalid cache payload")
                self.hits += 1
                return candidates
            except Exception:
                # A corrupt cache entry is treated as a miss.
                self.misses += 1
                return None

    def put(self, digest: str, candidates) -> None:
        path = self._path(digest)
        payload = {
            "version": self.CACHE_VERSION,
            "sha256": digest,
            "candidates": candidates,
            "saved_at": time.time(),
        }
        tmp = path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            tmp.replace(path)
            self.writes += 1

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "writes": self.writes,
            }


class PersistentResultCache:
    """Persistent cache for completed ImageResult objects.

    Cache keys include the input bytes, provider, threshold and candidate
    limit, so changing the important matching parameters does not silently
    reuse an incompatible result.
    """

    VERSION = 2

    def __init__(self, root: Path = Path("cache") / "results"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _key(
        self,
        image_path: str | Path,
        provider_name: str,
        threshold: Any,
        max_candidates: Any,
    ) -> str:
        h = hashlib.sha256()
        with Path(image_path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        config = {
            "version": self.VERSION,
            "image_path": str(Path(image_path).resolve()),
            "provider": provider_name,
            "threshold": threshold,
            "max_candidates": max_candidates,
        }
        h.update(
            json.dumps(
                config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(
        self,
        image_path: str | Path,
        provider_name: str,
        threshold: Any,
        max_candidates: Any,
    ):
        key = self._key(image_path, provider_name, threshold, max_candidates)
        path = self._path(key)
        with self._lock:
            if not path.is_file():
                self.misses += 1
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("version") != self.VERSION:
                    raise ValueError("unsupported result-cache version")
                result = payload.get("result")
                if not isinstance(result, dict):
                    raise ValueError("invalid result cache")
                self.hits += 1
                return _result_from_dict(result)
            except Exception:
                self.misses += 1
                return None

    def put(
        self,
        image_path: str | Path,
        provider_name: str,
        threshold: Any,
        max_candidates: Any,
        result: ImageResult,
    ) -> None:
        key = self._key(image_path, provider_name, threshold, max_candidates)
        path = self._path(key)
        payload = {
            "version": self.VERSION,
            "key": key,
            "saved_at": time.time(),
            "result": result.as_dict(),
        }
        tmp = path.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
            self.writes += 1

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "writes": self.writes,
            }


class CachedSearchProvider:
    """Compatibility wrapper around the existing reverse-search provider.

    PipelineRunner uses upload_image() followed by search_image_id() for
    SerpApi. We can therefore cache the expensive search result without
    changing PipelineRunner's public API.
    """

    def __init__(self, provider, cache: PersistentSearchCache):
        self._provider = provider
        self._cache = cache
        self._pending: Dict[str, str] = {}
        self._lock = threading.RLock()

        # Preserve provider identity expected by the CLI/reporting code.
        self.name = getattr(provider, "name", provider.__class__.__name__)

    def __getattr__(self, name):
        return getattr(self._provider, name)

    @staticmethod
    def _cached_id(digest: str) -> str:
        return f"fcv-cache:{digest}"

    def upload_image(self, image_path):
        digest = self._cache.digest_file(image_path)

        # Return a deterministic synthetic image id on cache hit. This avoids
        # the network upload entirely.
        if self._cache._path(digest).is_file():
            with self._lock:
                self._pending[self._cached_id(digest)] = digest
            return self._cached_id(digest)

        image_id = self._provider.upload_image(image_path)
        with self._lock:
            self._pending[image_id] = digest
        return image_id

    def search_image_id(self, image_id, limit=None):
        with self._lock:
            digest = self._pending.get(image_id)

        # A synthetic ID means the complete search result is already cached.
        if image_id.startswith("fcv-cache:") and digest:
            candidates = self._cache.get(digest)
            if candidates is not None:
                return _candidates_from_dicts(candidates)

        if digest:
            cached = self._cache.get(digest)
            if cached is not None:
                return _candidates_from_dicts(cached)

        candidates = self._provider.search_image_id(
            image_id,
            limit=limit,
        )

        if digest:
            self._cache.put(
                digest,
                [_candidate_to_dict(c) for c in candidates],
            )

        return candidates

    def search(self, query_url, limit=None):
        # For URL-based providers there is no reliable local input-image key,
        # so preserve the provider's original behavior.
        return self._provider.search(query_url, limit=limit)


def _candidate_to_dict(candidate) -> Dict[str, Any]:
    """Serialize a Candidate without depending on its exact dataclass fields."""
    if hasattr(candidate, "__dict__"):
        raw = dict(candidate.__dict__)
    else:
        raw = {}
        for name in (
            "url",
            "title",
            "image_url",
            "content_url",
            "thumbnail_url",
            "source",
            "is_exact_match",
            "search_type",
        ):
            if hasattr(candidate, name):
                raw[name] = getattr(candidate, name)

    # JSON cannot encode arbitrary objects. Keep only JSON-safe values.
    out = {}
    for key, value in raw.items():
        try:
            json.dumps(value)
        except TypeError:
            continue
        out[key] = value
    return out


def _candidates_from_dicts(rows):
    """Rebuild provider Candidate objects from cached dictionaries."""
    # Importing the class here avoids changing the module-level import layout.
    from .search.reverse_search import Candidate

    out = []
    for row in rows:
        try:
            out.append(Candidate(**row))
        except TypeError:
            # Be tolerant of future Candidate fields changing.
            allowed = getattr(Candidate, "__dataclass_fields__", {})
            filtered = {k: v for k, v in row.items() if k in allowed}
            out.append(Candidate(**filtered))
    return out

class BatchRunner:
    def __init__(self, cfg, provider, workers: int = 8, pool_size: Optional[int] = None,
                 checkpoint: Optional[Path] = None, keep_image_bytes: bool = True):
        self.cfg = cfg
        self.provider = provider
        self.workers = max(1, int(workers))
        self.pool = DetectorPool(
            size=pool_size or min(self.workers, max(1, (os.cpu_count() or 2) // 2), 4),
            model_name=cfg.face_model,
            det_size=cfg.face_det_size,
        )
        self.cache = CandidateCache(keep_bytes=keep_image_bytes)
        self.search_cache = PersistentSearchCache()
        self.result_cache = PersistentResultCache()
        self.cached_provider = CachedSearchProvider(provider, self.search_cache)
        self.runner = PipelineRunner(
            cfg,
            self.pool,
            self.cached_provider,
            cache=self.cache,
            verbose=False,
        )
        self.checkpoint = checkpoint
        self._ck_lock = threading.Lock()
        self._print_lock = threading.Lock()
        self.stats = BatchStats()

    # ------------------------------------------------------------------ #

    def load_checkpoint(self) -> Dict[str, Dict[str, Any]]:
        done: Dict[str, Dict[str, Any]] = {}
        if not self.checkpoint or not self.checkpoint.is_file():
            return done
        for line in self.checkpoint.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("image_path"):
                done[row["image_path"]] = row
        return done

    def _append_checkpoint(self, result: ImageResult) -> None:
        if not self.checkpoint:
            return
        with self._ck_lock:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with self.checkpoint.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #

    def run(self, images: List[Path], resume: bool = False) -> List[ImageResult]:
        self.stats = BatchStats(total=len(images))
        done = self.load_checkpoint() if resume else {}

        # de-duplicate byte-identical inputs so the same face is not searched twice
        pending: List[Path] = []
        by_digest: Dict[str, Path] = {}
        alias: Dict[str, Path] = {}
        results: List[ImageResult] = []

        for path in images:
            key = str(path)
            if resume and key in done:
                self.stats.skipped_resume += 1
                results.append(_result_from_dict(done[key]))
                continue
            try:
                digest = sha256_bytes(path.read_bytes())
            except OSError as exc:
                results.append(ImageResult(image_path=key, status="error", error=f"cannot read file: {exc}"))
                continue
            if digest in by_digest:
                alias[key] = by_digest[digest]
                self.stats.duplicates += 1
                continue
            by_digest[digest] = path
            pending.append(path)

        banner("batch discovery")
        kv("Images found", len(images))
        kv("Unique to process", len(pending))
        if self.stats.duplicates:
            kv("Duplicate inputs", self.stats.duplicates)
        if self.stats.skipped_resume:
            kv("Skipped (resume)", self.stats.skipped_resume)
        kv("Worker threads", self.workers)
        kv("Face model sessions", self.pool.size)
        kv("Provider", self.provider.name)

        if pending:
            step("Warming up the face model...")
            self.pool.warm()
            ok("Model ready")

        by_path: Dict[str, ImageResult] = {r.image_path: r for r in results}

        provider_name = getattr(self.provider, "name", self.provider.__class__.__name__)
        threshold = getattr(self.cfg, "match_threshold", None)
        max_candidates = getattr(self.cfg, "max_candidates", None)

        def _run_one(path: Path):
            # Result cache is deliberately checked before InsightFace/search.
            # This is the fast path for repeated identical inputs.
            cached = self.result_cache.get(
                path,
                provider_name,
                threshold,
                max_candidates,
            )
            if cached is not None:
                cached.image_path = str(path)
                cached.elapsed = 0.0
                return cached, True

            try:
                res = self.runner.run(str(path))
            except Exception as exc:  # defensive: PipelineRunner.run never raises
                res = ImageResult(
                    image_path=str(path),
                    status="error",
                    error=str(exc),
                )

            # Cache deterministic terminal results. Do not cache transient
            # search/network failures so a later run can retry them.
            if res.status in {"matched", "no_face"}:
                self.result_cache.put(
                    path,
                    provider_name,
                    threshold,
                    max_candidates,
                    res,
                )
            return res, False

        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="fcv") as pool:
            futures = {pool.submit(_run_one, p): p for p in pending}
            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    res, from_result_cache = fut.result()
                except Exception as exc:
                    res = ImageResult(
                        image_path=str(path),
                        status="error",
                        error=str(exc),
                    )
                    from_result_cache = False

                self.stats.processed += 1
                if res.status == STATUS_MATCHED:
                    self.stats.matched += 1
                self.stats.note(res.status)
                by_path[res.image_path] = res
                results.append(res)
                self._append_checkpoint(res)
                self._progress(res)

        # fan duplicates back out so the report covers every input file
        for dup_key, original in alias.items():
            src = by_path.get(str(original))
            if src is None:
                continue
            clone = _result_from_dict(src.as_dict())
            clone.image_path = dup_key
            clone.error = (clone.error + " " if clone.error else "") + "(duplicate of %s)" % original.name
            results.append(clone)

        order = {str(p): i for i, p in enumerate(images)}
        results.sort(key=lambda r: order.get(r.image_path, 1 << 30))
        return results

    def _progress(self, res: ImageResult) -> None:
        n, total = self.stats.processed, max(1, self.stats.total - self.stats.skipped_resume - self.stats.duplicates)
        rate = self.stats.processed / max(1e-6, self.stats.elapsed)
        eta = (total - n) / rate if rate else 0
        icon = "\u2713" if res.status == STATUS_MATCHED else "\u00b7"
        detail = (
            f"sim {res.similarity:.3f} {res.platform}" if res.status == STATUS_MATCHED
            else f"{res.status}{': ' + res.error[:48] if res.error else ''}"
        )
        with self._print_lock:
            print(
                f"  [{n:>3}/{total}] {icon} {Path(res.image_path).name[:32]:<32} "
                f"{detail[:60]:<60} {res.elapsed:5.1f}s  eta {eta/60:4.1f}m"
            )


def _result_from_dict(row: Dict[str, Any]) -> ImageResult:
    return ImageResult(
        image_path=row.get("image_path", ""),
        status=row.get("status", "error"),
        error=row.get("error", ""),
        similarity=float(row.get("similarity", -1.0)),
        candidates_found=int(row.get("candidates_found", 0) or 0),
        candidates_checked=int(row.get("candidates_checked", 0) or 0),
        source_url=row.get("source_url", ""),
        platform=row.get("platform", ""),
        matched_image_url=row.get("matched_image_url", ""),
        record=row.get("record"),
        data_hash=row.get("data_hash", ""),
        image_hash=row.get("image_hash", ""),
        query_image_url=row.get("query_image_url", ""),
        elapsed=float(row.get("elapsed_seconds", 0.0) or 0.0),
    )


# --------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------- #

CSV_COLUMNS = [
    "image_path", "status", "similarity", "platform", "source_url",
    "data_hash", "image_hash", "candidates_found", "candidates_checked",
    "elapsed_seconds", "error",
]


def write_reports(results: List[ImageResult], out_dir: Path, summary: Dict[str, Any]) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "batch_results.json"
    csv_path = out_dir / "batch_results.csv"

    json_path.write_text(
        json.dumps({"summary": summary, "results": [r.as_dict() for r in results]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_dict())
    return {"json": json_path, "csv": csv_path}


def print_summary(stats: BatchStats, results: List[ImageResult], cache_stats: Dict[str, int], search_cache_stats: Optional[Dict[str, int]] = None, result_cache_stats: Optional[Dict[str, int]] = None) -> None:
    banner("batch summary")
    total = len(results)
    matched = sum(1 for r in results if r.matched)
    kv("Images", total)
    kv("Matched", f"{matched} ({(matched / total * 100 if total else 0):.1f}%)")
    kv("Unmatched", total - matched)
    for status, count in sorted(stats.failures.items(), key=lambda kv_: -kv_[1]):
        kv(f"  {status}", count)
    kv("Wall clock", f"{stats.elapsed:.1f}s")
    if stats.processed:
        kv("Throughput", f"{stats.processed / max(1e-6, stats.elapsed) * 60:.1f} images/min")
        kv("Avg per image", f"{stats.elapsed / stats.processed:.2f}s")
    kv("Candidate cache", f"{cache_stats['hits']} hits / {cache_stats['misses']} downloads")
    if search_cache_stats is not None:
        kv(
            "Search cache",
            f"{search_cache_stats['hits']} hits / "
            f"{search_cache_stats['misses']} misses",
        )
        if search_cache_stats["writes"]:
            kv("Search cache writes", search_cache_stats["writes"])
    if cache_stats["hits"]:
        saved = cache_stats["hits"] / max(1, cache_stats["hits"] + cache_stats["misses"]) * 100
        kv("Downloads avoided", f"{saved:.1f}%")
