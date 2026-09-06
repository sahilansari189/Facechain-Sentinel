"""Reusable face identification pipeline.

Pipeline:

    input image
        -> InsightFace / ArcFace
        -> live reverse-image search
        -> candidate image retrieval
        -> multi-face ArcFace verification
        -> ranking
        -> strict threshold decision
        -> metadata extraction
        -> SHA-256 fingerprint

Design goals:

* Never treat "highest score" as a match unless it passes the threshold.
* Candidate image verification is independent from reverse-image-search ranking.
* Candidate downloads/embeddings are cached.
* Candidate verification runs concurrently for speed.
* Every face detected in a candidate image is compared.
* Page metadata is fetched only after a genuine face match.
* Safe failure: one broken candidate never aborts the whole run.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .face.detector import (
    DetectorPool,
    FaceError,
)
from .face.matcher import (
    ScoredCandidate,
    best_similarity_fast,
    rank_candidates,
    select_match,
)
from .search.candidate_fetcher import (
    FetchError,
    build_post_record,
    download_image,
    fetch_page_metadata,
)
from .search.reverse_search import (
    Candidate,
    SearchError,
    upload_image_for_search,
)
from .utils.hashing import (
    canonical_json,
    sha256_bytes,
    sha256_canonical_json,
)


# ============================================================
# Status codes
# ============================================================

STATUS_MATCHED = "matched"
STATUS_NO_FACE = "no_face"
STATUS_NO_CANDIDATES = "no_candidates"
STATUS_NO_MATCH = "no_match"
STATUS_SEARCH_ERROR = "search_error"
STATUS_ERROR = "error"


# ============================================================
# Result model
# ============================================================

@dataclass
class ImageResult:
    """Everything the pipeline learned about one input image."""

    image_path: str
    status: str

    error: str = ""

    similarity: float = -1.0

    candidates_found: int = 0
    candidates_checked: int = 0

    source_url: str = ""
    platform: str = ""

    matched_image_url: str = ""
    matched_image_source: str = ""

    record: Optional[Dict[str, Any]] = None

    data_hash: str = ""
    image_hash: str = ""

    query_image_url: str = ""

    elapsed: float = 0.0

    scores: List[Dict[str, Any]] = field(
        default_factory=list
    )

    @property
    def matched(self) -> bool:
        """True only for an actual verified + fingerprinted match."""
        return (
            self.status == STATUS_MATCHED
            and bool(self.data_hash)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "status": self.status,
            "error": self.error,
            "similarity": round(
                float(self.similarity),
                6,
            ),
            "candidates_found": (
                self.candidates_found
            ),
            "candidates_checked": (
                self.candidates_checked
            ),
            "source_url": self.source_url,
            "platform": self.platform,
            "matched_image_url": (
                self.matched_image_url
            ),
            "matched_image_source": (
                self.matched_image_source
            ),
            "data_hash": self.data_hash,
            "image_hash": self.image_hash,
            "query_image_url": (
                self.query_image_url
            ),
            "elapsed_seconds": round(
                self.elapsed,
                3,
            ),
            "record": self.record,
            "scores": self.scores,
        }


# ============================================================
# Candidate cache
# ============================================================

class CandidateCache:
    """Thread-safe cache for candidate downloads/embeddings.

    Same image URLs frequently appear across reverse-search
    results. Cache them so they are downloaded and embedded
    only once.

    The cache is also safe for concurrent workers.
    """

    def __init__(
        self,
        max_entries: int = 4096,
        keep_bytes: bool = True,
    ):
        self.max_entries = max_entries
        self.keep_bytes = keep_bytes

        self._data: Dict[
            str,
            Tuple[
                Optional[np.ndarray],
                Optional[bytes],
                Optional[str],
            ],
        ] = {}

        self._locks: Dict[
            str,
            threading.Lock,
        ] = {}

        self._guard = threading.Lock()

        self.hits = 0
        self.misses = 0

    def _lock_for(
        self,
        url: str,
    ) -> threading.Lock:

        with self._guard:

            lock = self._locks.get(url)

            if lock is None:

                lock = threading.Lock()

                self._locks[url] = lock

            return lock

    def get_or_compute(
        self,
        url: str,
        compute,
    ):
        # Fast cache check.
        with self._guard:

            if url in self._data:

                self.hits += 1

                return self._data[url]

        # Per-URL lock prevents duplicate work
        # when several workers request the same URL.
        with self._lock_for(url):

            with self._guard:

                if url in self._data:

                    self.hits += 1

                    return self._data[url]

            try:

                embeddings, data = compute(url)

                entry = (
                    embeddings,
                    data
                    if self.keep_bytes
                    else None,
                    None,
                )

            except (
                FetchError,
                FaceError,
            ) as exc:

                entry = (
                    None,
                    None,
                    str(exc),
                )

            except Exception as exc:

                entry = (
                    None,
                    None,
                    f"unexpected error: {exc}",
                )

            with self._guard:

                self.misses += 1

                if (
                    len(self._data)
                    >= self.max_entries
                ):

                    oldest = next(
                        iter(self._data)
                    )

                    self._data.pop(
                        oldest,
                        None,
                    )

                self._data[url] = entry

            return entry

    def stats(self) -> Dict[str, int]:

        with self._guard:

            return {
                "entries": len(
                    self._data
                ),
                "hits": self.hits,
                "misses": self.misses,
            }


# ============================================================
# Candidate evaluation result
# ============================================================

@dataclass
class _CandidateEvaluation:

    index: int
    candidate: Candidate

    scored: ScoredCandidate

    image_url: str = ""
    image_source: str = ""

    page_meta: Dict[str, str] = field(
        default_factory=dict
    )

    metadata_error: str = ""


# ============================================================
# Pipeline
# ============================================================

class PipelineRunner:
    """Runs discovery + face verification for one image."""

    def __init__(
        self,
        cfg,
        pool: DetectorPool,
        provider,
        cache: Optional[CandidateCache] = None,
        verbose: bool = True,
        keep_image_bytes: bool = True,
    ):

        self.cfg = cfg

        self.pool = pool

        self.provider = provider

        self.cache = (
            cache
            or CandidateCache(
                keep_bytes=keep_image_bytes
            )
        )

        self.verbose = verbose

        # Do not use an excessive number of
        # InsightFace sessions.
        configured_workers = getattr(
            cfg,
            "batch_workers",
            8,
        )

        self.max_workers = max(
            1,
            min(
                int(configured_workers),
                8,
            ),
        )

    # ========================================================
    # Logging
    # ========================================================

    def _log(
        self,
        msg: str,
    ) -> None:

        if self.verbose:

            print(msg)

    # ========================================================
    # Candidate embedding
    # ========================================================

    def _candidate_image_sources(
        self,
        candidate: Candidate,
    ) -> List[Tuple[str, str]]:
        """Return all provider-supplied image URLs, without duplicates.

        Lens can return a thumbnail or an intermediate CDN image.  The
        candidate model may also be extended by future providers with
        additional image fields, so collect the common variants here.
        """
        fields = (
            ("image_url", "lens_image"),
            ("content_url", "content_image"),
            ("thumbnail_url", "thumbnail"),
        )

        out: List[Tuple[str, str]] = []
        seen = set()

        for attr, source in fields:
            value = (getattr(candidate, attr, "") or "").strip()
            if value and value not in seen:
                seen.add(value)
                out.append((value, source))

        return out

    def _page_image_fallback(
        self,
        candidate: Candidate,
        attempted_urls: Sequence[str],
    ) -> Optional[Tuple[str, str]]:
        """Get a public og:image only after direct image retrieval fails."""
        page_url = (getattr(candidate, "url", "") or "").strip()
        if not page_url:
            return None

        try:
            meta, _error = fetch_page_metadata(
                page_url,
                self.cfg.http_timeout,
            )
        except Exception:
            return None

        og_image = (meta.get("og_image") or "").strip()
        if not og_image or og_image in attempted_urls:
            return None

        return og_image, "page_og_image"

    def _download_and_embed_image(
        self,
        image_url: str,
    ) -> Tuple[Optional[np.ndarray], Optional[bytes], Optional[str]]:
        """Download one public image and return every detected face embedding.

        This helper deliberately keeps image retrieval and face detection
        together so every image source gets identical verification treatment.
        """
        try:
            data = download_image(
                image_url,
                self.cfg.max_image_bytes,
                self.cfg.http_timeout,
            )

            with self.pool.lease() as detector:
                faces = detector.detect_bytes(data)

            if not faces:
                return None, data, "no face in candidate image"

            embeddings = np.asarray(
                [face.embedding for face in faces],
                dtype=np.float32,
            )

            return embeddings, data, None

        except (FetchError, FaceError) as exc:
            return None, None, str(exc)
        except Exception as exc:
            return None, None, f"unexpected error: {exc}"

    def _try_page_og_image(
        self,
        candidate: Candidate,
        attempted_urls: Sequence[str],
    ) -> Optional[Tuple[str, str]]:
        """Fetch public page metadata and return a new og:image URL."""
        fallback = self._page_image_fallback(candidate, attempted_urls)
        return fallback

    def embed_candidate(
        self,
        candidate: Candidate,
        reference: Optional[np.ndarray] = None,
        retry_weak_match: bool = True,
    ):
        """Download, detect and embed candidate images.

        Verification strategy:
          1. Try the provider's full image first.
          2. If it fails, try public og:image.
          3. If a reference embedding is available, inspect the source
             page's og:image as well and retain the strongest score. This is
             important because Lens can return a low-quality/cropped
             thumbnail while the source page contains the actual
             high-resolution image.
          4. Compare every detected face and keep the strongest source.

        Returns:
            (embeddings, image_bytes, image_url, image_source, error)
        """
        sources = self._candidate_image_sources(candidate)
        errors: List[str] = []
        best = None

        # Fast path: provider image/CDN.
        for image_url, image_source in sources:
            embeddings, data, error = self._download_and_embed_image(image_url)

            if error or embeddings is None or len(embeddings) == 0:
                errors.append(f"{image_source}: {error or 'no usable face'}")
                continue

            score = (
                best_similarity_fast(reference, embeddings)
                if reference is not None
                else -1.0
            )

            best = (score, embeddings, data, image_url, image_source)

            # Do not stop merely because this source crossed the
            # threshold. A different public image from the same candidate
            # (especially the source page's og:image) may produce a stronger
            # verification score. We therefore keep evaluating all available
            # candidate image sources and retain the strongest one.
            #
            # This is important for consistency between single-image and
            # batch execution: a Lens thumbnail such as 0.8264 must not prevent
            # the source page image from reaching 1.0000.

            # If no reference was supplied, the first valid image is enough.
            if reference is None:
                return (
                    embeddings,
                    data,
                    image_url,
                    image_source,
                    None,
                )

        # For weak/unusable direct results, inspect the source page's
        # OpenGraph image.  This is intentionally not done for a strong
        # direct match because it would only add latency.
        attempted = [url for url, _source in sources]
        fallback = self._try_page_og_image(candidate, attempted)

        if fallback:
            image_url, image_source = fallback

            embeddings, data, error = self._download_and_embed_image(image_url)

            if not error and embeddings is not None and len(embeddings):
                score = (
                    best_similarity_fast(reference, embeddings)
                    if reference is not None
                    else -1.0
                )

                if best is None or score > best[0]:
                    best = (
                        score,
                        embeddings,
                        data,
                        image_url,
                        image_source,
                    )
            else:
                errors.append(
                    f"{image_source}: {error or 'no usable face'}"
                )

        if best is not None:
            _score, embeddings, data, image_url, image_source = best
            return (
                embeddings,
                data,
                image_url,
                image_source,
                None,
            )

        return (
            None,
            None,
            "",
            "",
            "; ".join(errors) or "candidate has no usable image URL",
        )

    # ========================================================
    # Reference embedding
    # ========================================================

    def reference_embedding(
        self,
        image_path: str,
        face_index: Optional[int] = None,
    ):

        with self.pool.lease() as detector:

            image = detector.load_image(
                image_path
            )

            faces = detector.detect(
                image
            )

            if not faces:

                raise FaceError(
                    "no face detected in the input image"
                )

            if face_index is not None:

                if not (
                    0
                    <= face_index
                    < len(faces)
                ):

                    raise FaceError(
                        f"face index "
                        f"{face_index} out of range "
                        f"(0..{len(faces)-1})"
                    )

                primary = faces[
                    face_index
                ]

            else:

                primary = (
                    detector.primary_face(
                        faces
                    )
                )

        return primary, faces

    # ========================================================
    # Single candidate scoring
    # ========================================================

    def _score_one(
        self,
        index: int,
        candidate: Candidate,
        reference: np.ndarray,
    ) -> _CandidateEvaluation:

        # ----------------------------------------------------
        # Candidate image URL
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Cached download + embedding
        # ----------------------------------------------------

        (
            embeddings,
            data,
            verified_image_url,
            verified_image_source,
            error,
        ) = self.embed_candidate(
            candidate,
            reference=reference,
            retry_weak_match=True,
        )

        if (
            error
            or embeddings is None
        ):

            scored = ScoredCandidate(
                candidate,
                -1.0,
                faces_found=0,
                error=(
                    error
                    or "no embedding"
                ),
            )

            return _CandidateEvaluation(
                index=index,
                candidate=candidate,
                scored=scored,
            )

        # ----------------------------------------------------
        # Compare reference against EVERY face.
        #
        # best_similarity_fast performs one matrix
        # multiplication instead of a Python loop.
        # ----------------------------------------------------

        similarity = (
            best_similarity_fast(
                reference,
                embeddings,
            )
        )

        scored = ScoredCandidate(
            candidate,
            similarity,
            faces_found=len(
                embeddings
            ),
            image_bytes=data,
        )

        return _CandidateEvaluation(
            index=index,
            candidate=candidate,
            scored=scored,
            image_url=verified_image_url,
            image_source=verified_image_source,
        )

    # ========================================================
    # Parallel candidate scoring
    # ========================================================

    def score_candidates(
        self,
        reference: np.ndarray,
        candidates: Sequence[Candidate],
    ) -> List[ScoredCandidate]:
        """Score candidates concurrently.

        Network downloads dominate this stage, so sequential
        evaluation is unnecessarily slow.

        The actual InsightFace calls remain protected by the
        DetectorPool.
        """

        if not candidates:

            return []

        evaluations: List[
            _CandidateEvaluation
        ] = []

        # Verify Lens exact matches first.  This does not lower the face
        # threshold and therefore cannot turn a weak result into a match.
        # It simply gives exact Lens evidence priority in the verification
        # work queue and improves the chance of finding the genuine result
        # earlier.
        ordered_candidates = sorted(
            enumerate(candidates, 1),
            key=lambda pair: (
                not bool(
                    getattr(
                        pair[1],
                        "is_exact_match",
                        False,
                    )
                ),
                pair[0],
            ),
        )

        workers = min(
            self.max_workers,
            len(ordered_candidates),
        )

        self._log(
            f"  Verifying "
            f"{len(candidates)} candidates "
            f"with {workers} workers..."
        )

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="face-match",
        ) as executor:

            futures = {
                executor.submit(
                    self._score_one,
                    index,
                    candidate,
                    reference,
                ): (
                    index,
                    candidate,
                )
                for index, candidate
                in ordered_candidates
            }

            for future in as_completed(
                futures
            ):

                index, candidate = (
                    futures[future]
                )

                try:

                    evaluation = (
                        future.result()
                    )

                except Exception as exc:

                    scored = ScoredCandidate(
                        candidate,
                        -1.0,
                        faces_found=0,
                        error=(
                            f"unexpected error: "
                            f"{exc}"
                        ),
                    )

                    evaluation = (
                        _CandidateEvaluation(
                            index=index,
                            candidate=candidate,
                            scored=scored,
                        )
                    )

                evaluations.append(
                    evaluation
                )

        # Restore search-result order.
        evaluations.sort(
            key=lambda item: item.index
        )

        return [
            evaluation.scored
            for evaluation in evaluations
        ]

    # ========================================================
    # Candidate scoring with diagnostic information
    # ========================================================

    def evaluate_candidates(
        self,
        reference: np.ndarray,
        candidates: Sequence[Candidate],
    ) -> List[_CandidateEvaluation]:
        """Evaluate candidates and retain image-source metadata."""

        if not candidates:

            return []

        evaluations: List[
            _CandidateEvaluation
        ] = []

        # Verify Lens exact matches first.  This does not lower the face
        # threshold and therefore cannot turn a weak result into a match.
        # It simply gives exact Lens evidence priority in the verification
        # work queue and improves the chance of finding the genuine result
        # earlier.
        ordered_candidates = sorted(
            enumerate(candidates, 1),
            key=lambda pair: (
                not bool(
                    getattr(
                        pair[1],
                        "is_exact_match",
                        False,
                    )
                ),
                pair[0],
            ),
        )

        workers = min(
            self.max_workers,
            len(ordered_candidates),
        )

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="face-match",
        ) as executor:

            futures = {
                executor.submit(
                    self._score_one,
                    index,
                    candidate,
                    reference,
                ): (
                    index,
                    candidate,
                )
                for index, candidate
                in ordered_candidates
            }

            for future in as_completed(
                futures
            ):

                index, candidate = (
                    futures[future]
                )

                try:

                    evaluation = (
                        future.result()
                    )

                except Exception as exc:

                    scored = ScoredCandidate(
                        candidate,
                        -1.0,
                        faces_found=0,
                        error=(
                            f"unexpected error: "
                            f"{exc}"
                        ),
                    )

                    evaluation = (
                        _CandidateEvaluation(
                            index=index,
                            candidate=candidate,
                            scored=scored,
                        )
                    )

                evaluations.append(
                    evaluation
                )

        evaluations.sort(
            key=lambda item: item.index
        )

        return evaluations

    # ========================================================
    # Page metadata
    # ========================================================

    def _fetch_metadata(
        self,
        candidate: Candidate,
    ) -> Tuple[
        Dict[str, str],
        str,
    ]:

        if not candidate.url:

            return {}, ""

        try:

            meta, error = (
                fetch_page_metadata(
                    candidate.url,
                    self.cfg.http_timeout,
                )
            )

            return (
                meta or {},
                error or "",
            )

        except Exception as exc:

            return {}, str(exc)

    # ========================================================
    # Main pipeline
    # ========================================================

    def run(
        self,
        image_path: str,
        image_url: str = "",
        face_index: Optional[int] = None,
    ) -> ImageResult:
        """Run the full discovery pipeline.

        This method deliberately NEVER raises a normal pipeline
        error. The result object records the failure instead.
        """

        started = time.time()

        result = ImageResult(
            image_path=str(image_path),
            status=STATUS_ERROR,
        )

        try:

            # =================================================
            # 1. Reference face
            # =================================================

            try:

                primary, _faces = (
                    self.reference_embedding(
                        image_path,
                        face_index,
                    )
                )

            except FaceError as exc:

                result.status = (
                    STATUS_NO_FACE
                )

                result.error = str(exc)

                return result

            reference = (
                primary.embedding
            )

            # =================================================
            # 2. Live reverse image search
            # =================================================

            try:
                # SerpApi can accept the local image directly through its
                # Image API. This avoids the old 0x0.st/Cloudinary hop for
                # the actual Lens query and gives Lens an image_id.
                if image_url:
                    query_url = image_url
                    result.query_image_url = query_url

                    candidates = self.provider.search(
                        query_url,
                        limit=self.cfg.max_candidates,
                    )

                elif (
                    self.cfg.search_provider.lower() == "serpapi"
                    and hasattr(self.provider, "upload_image")
                    and hasattr(self.provider, "search_image_id")
                ):
                    self._log(
                        "  Uploading image directly to SerpApi Image API..."
                    )

                    image_id = self.provider.upload_image(
                        image_path
                    )

                    # Keep the identifier for diagnostics. It is not a
                    # browser URL and must not be sent to download_image().
                    result.query_image_url = (
                        f"serpapi:image_id:{image_id}"
                    )

                    self._log(
                        "  Performing LIVE Google Lens search..."
                    )

                    candidates = self.provider.search_image_id(
                        image_id,
                        limit=self.cfg.max_candidates,
                    )

                else:
                    query_url = upload_image_for_search(
                        image_path,
                        self.cfg.image_upload_url,
                        self.cfg.http_timeout,
                    )

                    result.query_image_url = query_url

                    candidates = self.provider.search(
                        query_url,
                        limit=self.cfg.max_candidates,
                    )

            except SearchError as exc:

                result.status = (
                    STATUS_SEARCH_ERROR
                )

                result.error = str(exc)

                return result

            result.candidates_found = (
                len(candidates)
            )

            if not candidates:

                result.status = (
                    STATUS_NO_CANDIDATES
                )

                result.error = (
                    "reverse image search "
                    "returned no candidates"
                )

                return result

            # =================================================
            # 3. Fast parallel face verification
            # =================================================

            evaluations = (
                self.evaluate_candidates(
                    reference,
                    candidates,
                )
            )

            scored = [
                evaluation.scored
                for evaluation in evaluations
            ]

            result.candidates_checked = sum(
                1
                for item in scored
                if item.ok
            )

            ranked = rank_candidates(
                scored
            )

            result.scores = []

            for item in ranked:

                result.scores.append(
                    {
                        "url": (
                            item.candidate.url
                        ),
                        "image_url": (
                            item.candidate.image_url
                        ),
                        "similarity": round(
                            float(
                                item.similarity
                            ),
                            6,
                        ),
                        "faces_found": (
                            item.faces_found
                        ),
                        "is_exact_match": bool(
                            getattr(
                                item.candidate,
                                "is_exact_match",
                                False,
                            )
                        ),
                        "search_type": (
                            getattr(
                                item.candidate,
                                "search_type",
                                "",
                            )
                        ),
                        "verified_image_source": (
                            next(
                                (
                                    evaluation.image_source
                                    for evaluation in evaluations
                                    if evaluation.scored is item
                                ),
                                "",
                            )
                        ),
                        "verified_image_url": (
                            next(
                                (
                                    evaluation.image_url
                                    for evaluation in evaluations
                                    if evaluation.scored is item
                                ),
                                "",
                            )
                        ),
                        "error": item.error,
                    }
                )

            # =================================================
            # 4. STRICT MATCH DECISION
            # =================================================

            match = select_match(
                scored,
                self.cfg.match_threshold,
            )

            # -------------------------------------------------
            # IMPORTANT:
            #
            # If 0.294 is the highest score but threshold
            # is 0.45, this is NO MATCH.
            #
            # We do NOT identify the "best candidate" as
            # the person.
            # -------------------------------------------------

            if match is None:

                best = next(
                    (
                        item
                        for item in ranked
                        if item.ok
                    ),
                    None,
                )

                result.status = (
                    STATUS_NO_MATCH
                )

                if best:

                    result.similarity = (
                        float(
                            best.similarity
                        )
                    )

                    result.error = (
                        "no candidate reached "
                        f"the threshold "
                        f"{self.cfg.match_threshold:.2f}"
                    )

                else:

                    result.similarity = (
                        -1.0
                    )

                    result.error = (
                        "no candidate image "
                        "contained a usable face"
                    )

                return result

            # =================================================
            # 5. Genuine match
            # =================================================

            result.similarity = (
                float(match.similarity)
            )

            matched_eval = next(
                (
                    evaluation
                    for evaluation in evaluations
                    if evaluation.scored is match
                ),
                None,
            )

            result.matched_image_url = (
                (
                    matched_eval.image_url
                    if matched_eval
                    else match.candidate.image_url
                )
                or ""
            )

            result.matched_image_source = (
                (
                    matched_eval.image_source
                    if matched_eval
                    else "lens_image"
                )
                or ""
            )

            # =================================================
            # 6. Only now fetch source page metadata
            # =================================================

            page_meta, metadata_error = (
                self._fetch_metadata(
                    match.candidate
                )
            )

            if metadata_error:
                result.status = STATUS_ERROR
                result.error = f"source metadata unavailable: {metadata_error}"
                return result

            record = build_post_record(
                match.candidate,
                page_meta,
            )

            record[
                "similarity"
            ] = round(
                float(match.similarity),
                6,
            )

            record[
                "match_threshold"
            ] = round(
                float(
                    self.cfg.match_threshold
                ),
                6,
            )

            record[
                "search_provider"
            ] = self.cfg.search_provider

            record[
                "lens_exact_match"
            ] = bool(
                getattr(
                    match.candidate,
                    "is_exact_match",
                    False,
                )
            )

            record[
                "lens_search_type"
            ] = getattr(
                match.candidate,
                "search_type",
                "",
            )

            record[
                "verified_image_url"
            ] = result.matched_image_url

            record[
                "verified_image_source"
            ] = result.matched_image_source

            if metadata_error:

                record[
                    "metadata_error"
                ] = metadata_error

            # =================================================
            # 7. Fingerprints
            # =================================================

            data_hash = (
                sha256_canonical_json(
                    record
                )
            )

            image_hash = (
                sha256_bytes(
                    match.image_bytes
                )
                if match.image_bytes
                else ""
            )

            result.record = record

            result.source_url = (
                record[
                    "source_url"
                ]
            )

            result.platform = (
                record.get(
                    "platform",
                    "",
                )
            )

            result.data_hash = (
                data_hash
            )

            result.image_hash = (
                image_hash
            )

            result.status = (
                STATUS_MATCHED
            )

            return result

        except Exception as exc:

            result.status = (
                STATUS_ERROR
            )

            result.error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            return result

        finally:

            result.elapsed = (
                time.time()
                - started
            )


# ============================================================
# Canonical record helper
# ============================================================

def canonical_record_json(
    record: Dict[str, Any],
) -> str:

    return canonical_json(
        record
    )