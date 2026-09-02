"""Face Identification & Blockchain Verification pipeline (CLI entrypoint).

Usage:

    python src/main.py --image examples/input.jpg

Single-image flow:

    input image
        -> InsightFace detection
        -> ArcFace reference embedding
        -> Cloudinary/public image URL
        -> live reverse image search
        -> candidate pages/images
        -> candidate image verification
        -> Lens image + og:image fallback
        -> all-face ArcFace comparison
        -> threshold verification
        -> metadata fingerprint
        -> optional blockchain anchoring

Batch mode remains available through src.batch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from src.batch import (  # noqa: E402
    BatchRunner,
    discover_images,
    print_summary,
    write_reports,
)
from src.blockchain.contract import ChainError  # noqa: E402
from src.blockchain.verifier import (  # noqa: E402
    BlockchainVerifier,
)
from src.face.detector import (  # noqa: E402
    FaceDetector,
    FaceError,
)
from src.face.matcher import (  # noqa: E402
    ScoredCandidate,
    best_similarity,
    rank_candidates,
    select_match,
)
from src.search.candidate_fetcher import (  # noqa: E402
    FetchError,
    build_post_record,
    download_image,
    fetch_page_metadata,
)
from src.search.reverse_search import (  # noqa: E402
    SearchError,
    get_provider,
    upload_image_for_search,
)
from src.utils.config import (  # noqa: E402
    ConfigError,
    load_config,
)
from src.utils.hashing import (  # noqa: E402
    canonical_json,
    sha256_bytes,
    sha256_canonical_json,
)
from src.utils.logging_ui import (  # noqa: E402
    banner,
    fail,
    kv,
    ok,
    step,
    warn,
)


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Face identification + blockchain "
            "verification pipeline"
        )
    )

    p.add_argument(
        "--image",
        help="path to a single input image containing a face",
    )

    p.add_argument(
        "--batch",
        nargs="+",
        metavar="PATH",
        help=(
            "batch mode: one or more image files, "
            "directories or globs"
        ),
    )

    p.add_argument(
        "--image-url",
        default="",
        help=(
            "public URL of the same image "
            "(skips temporary upload)"
        ),
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "cosine similarity threshold "
            "(overrides MATCH_THRESHOLD)"
        ),
    )

    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help=(
            "max candidates to evaluate "
            "(overrides MAX_CANDIDATES)"
        ),
    )

    p.add_argument(
        "--provider",
        default=None,
        help=(
            "reverse image search provider: "
            "serpapi | bing | tineye"
        ),
    )

    p.add_argument(
        "--save-results",
        action="store_true",
        help=(
            "write live search/candidate data "
            "to debug/"
        ),
    )

    p.add_argument(
        "--no-chain",
        action="store_true",
        help=(
            "run discovery and hashing only; "
            "skip blockchain"
        ),
    )

    p.add_argument(
        "--face-index",
        type=int,
        default=None,
        help=(
            "pick a specific input face when "
            "several are detected"
        ),
    )

    # Batch options
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "[batch] parallel worker threads"
        ),
    )

    p.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help=(
            "[batch] concurrent face-model sessions (keep near workers; 1-2 on CPU)"
        ),
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "[batch] records anchored per transaction"
        ),
    )

    p.add_argument(
        "--out",
        default="reports",
        help="[batch] output directory",
    )

    p.add_argument(
        "--resume",
        action="store_true",
        help="[batch] resume checkpoint",
    )

    p.add_argument(
        "--checkpoint",
        default="reports/checkpoint.jsonl",
        help="[batch] checkpoint file",
    )

    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="[batch] do not recurse",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="[batch] maximum images",
    )

    p.add_argument(
        "--fast-batch",
        action="store_true",
        help=(
            "[batch] prefer low-overhead CPU settings: "
            "1 face-model session per worker"
        ),
    )

    args = p.parse_args(argv)

    if not args.image and not args.batch:
        p.error(
            "provide --image PATH for a single run, "
            "or --batch PATH... for batch mode"
        )

    return args


# ============================================================
# Configuration overrides
# ============================================================

def _apply_overrides(cfg, args):
    if args.threshold is not None:
        cfg.match_threshold = args.threshold

    if args.max_candidates is not None:
        cfg.max_candidates = args.max_candidates

    if args.provider:
        cfg.search_provider = args.provider.lower()

    if args.workers is not None:
        cfg.batch_workers = args.workers

    if args.pool_size is not None:
        cfg.face_pool_size = args.pool_size

    if args.chunk_size is not None:
        cfg.batch_chunk_size = args.chunk_size

    if (
        getattr(args, "fast_batch", False)
        and args.pool_size is None
    ):
        # On CPU, multiple sessions per worker can multiply model
        # initialization and memory pressure without improving throughput.
        cfg.face_pool_size = max(1, cfg.batch_workers)

    return cfg


# ============================================================
# Debug output
# ============================================================

def _save_debug(debug: dict) -> None:
    """Write debug information to debug/latest.json."""

    debug_dir = Path("debug")
    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = debug_dir / "latest.json"

    output.write_text(
        json.dumps(
            debug,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"\n  Debug report saved to: {output}"
    )



# ============================================================
# Search upload optimization
# ============================================================

def _search_local_with_serpapi(provider, image_path, limit):
    """Use SerpApi's direct image upload when supported.

    This avoids the extra Cloudinary/public-URL upload used by the
    legacy path. Falls back to the normal URL-based search if the
    provider does not expose the direct local-file API.
    """
    search_local_file = getattr(provider, "search_local_file", None)
    if callable(search_local_file):
        return search_local_file(
            image_path,
            limit=limit,
        )
    return None


def _is_google_goto_url(url: str) -> bool:
    """Return True when Google gives us a /goto intermediary URL."""
    if not url:
        return False

    try:
        parsed = urlparse(url)
    except Exception:
        return False

    return (
        parsed.netloc.lower().endswith("google.com")
        and parsed.path.lower().rstrip("/").endswith("/goto")
    )


def _extract_goto_target(url: str) -> str:
    """Extract a directly embedded target URL when available."""
    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
    except Exception:
        return ""

    for key in ("url", "q", "target"):
        values = query.get(key) or []
        if values:
            target = unquote(values[0]).strip()
            if target.startswith(("http://", "https://")):
                return target

    return ""


def _resolve_page_url(url: str, timeout: float) -> str:
    """Resolve a Google /goto URL to its real source page when possible."""
    if not url or not _is_google_goto_url(url):
        return url

    embedded = _extract_goto_target(url)
    if embedded:
        return embedded

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/139.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=max(2.0, min(float(timeout), 10.0)),
            allow_redirects=True,
            stream=True,
        )
        resolved = response.url.strip()
        response.close()

        if resolved and not _is_google_goto_url(resolved):
            return resolved
    except requests.RequestException:
        pass

    return url


def _candidate_page_url(candidate, timeout: float) -> str:
    """Prefer a real source/canonical page over Google's /goto URL."""
    url = getattr(candidate, "url", "") or ""
    resolved = _resolve_page_url(url, timeout)

    metadata = getattr(candidate, "page_metadata", None) or {}
    canonical = (
        metadata.get("canonical_url")
        or metadata.get("source_url")
        or ""
    )

    if canonical and not _is_google_goto_url(canonical):
        return canonical

    return resolved



def _source_image_url(page_meta, page_url: str) -> str:
    """Return a real source-page image URL when metadata exposes one.

    SerpApi/Lens image URLs are temporary proxy/cache URLs and should not be
    treated as the authoritative image URL.  The candidate fetcher already
    extracts og:image when it can; keep the selection centralized here.
    """
    meta = page_meta or {}
    for key in (
        "og_image",
        "twitter_image",
        "image_url",
        "image",
    ):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            value = value.strip()
            if not _is_google_goto_url(value):
                return value
    return ""


# ============================================================
# Main
# ============================================================

def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = _apply_overrides(
        load_config(),
        args,
    )

    if args.batch:
        return run_batch(
            args,
            cfg,
        )

    return run_single(
        args,
        cfg,
    )


# ============================================================
# Single-image pipeline
# ============================================================

def run_single(args, cfg) -> int:

    debug = {}

    # ========================================================
    # FACE IDENTIFICATION
    # ========================================================

    banner("face identification")

    try:

        step(
            f"Loading image: {args.image}"
        )

        detector = FaceDetector(
            model_name=cfg.face_model,
            det_size=cfg.face_det_size,
        )

        image = FaceDetector.load_image(
            args.image
        )

        kv(
            "Image size",
            f"{image.shape[1]}x{image.shape[0]}",
        )

        step(
            "Running detector (InsightFace / ArcFace)..."
        )

        faces = detector.detect(
            image
        )

        if not faces:

            fail(
                "No face detected in the input image. "
                "Try a clearer, front-facing photo."
            )

            return 2

        ok(
            f"Faces detected: {len(faces)}"
        )

        for i, face in enumerate(faces):

            kv(
                f"  face[{i}]",
                face.summary(),
            )

        if (
            len(faces) > 1
            and args.face_index is None
        ):

            warn(
                "Multiple faces detected - "
                "using the largest one. "
                "Override with --face-index N."
            )

        if args.face_index is not None:

            if not (
                0
                <= args.face_index
                < len(faces)
            ):

                fail(
                    f"--face-index {args.face_index} "
                    f"out of range "
                    f"(0..{len(faces)-1})"
                )

                return 2

            primary = faces[
                args.face_index
            ]

        else:

            primary = detector.primary_face(
                faces
            )

        reference = primary.embedding

        ok(
            "Embedding generated"
        )

        kv(
            "Selected face",
            primary.summary(),
        )

        kv(
            "Embedding dims",
            reference.shape[0],
        )

    except FaceError as exc:

        fail(str(exc))

        return 2

    # ========================================================
    # WEB SEARCH
    # ========================================================

    banner("web search")

    try:

        cfg.require_search()

        image_url = args.image_url

        provider = get_provider(
            cfg.search_provider,
            cfg,
        )

        step(
            "Performing LIVE reverse image search "
            f"(provider: {provider.name})..."
        )

        # SerpApi can accept the local image directly. This removes
        # the extra Cloudinary/public-URL upload from the hot path.
        candidates = None

        if (
            not image_url
            and provider.name == "serpapi"
        ):
            try:
                candidates = _search_local_with_serpapi(
                    provider,
                    args.image,
                    cfg.max_candidates,
                )
                if candidates is not None:
                    kv(
                        "Search upload",
                        "SerpApi direct local upload",
                    )
            except SearchError:
                # Preserve the existing public-URL fallback.
                candidates = None

        if candidates is None:
            if not image_url:
                step(
                    "Publishing input image temporarily "
                    f"via {cfg.image_upload_url} ..."
                )

                image_url = upload_image_for_search(
                    args.image,
                    cfg.image_upload_url,
                    cfg.http_timeout,
                )

            kv(
                "Query image URL",
                image_url,
            )

            candidates = provider.search(
                image_url,
                limit=cfg.max_candidates,
            )

        if not candidates:

            fail(
                "Search returned zero candidates. "
                "Try another image or a different provider."
            )

            return 3

        ok(
            f"Found {len(candidates)} unique candidates"
        )

        for i, candidate in enumerate(
            candidates[:10],
            1,
        ):

            kv(
                f"  [{i}] "
                f"{candidate.source or 'unknown'}",
                (
                    candidate.title
                    or candidate.url
                )[:70],
            )

        debug[
            "query_image_url"
        ] = image_url

        debug[
            "provider"
        ] = provider.name

        debug[
            "candidates"
        ] = [
            candidate.as_dict()
            for candidate in candidates
        ]

    except (
        SearchError,
        ConfigError,
    ) as exc:

        fail(str(exc))

        return 3

    # ========================================================
    # FACE MATCHING
    # ========================================================

    banner("face matching")

    scored = []

    for i, candidate in enumerate(
        candidates,
        1,
    ):

        label = (
            f"Candidate {i:>2} "
            f"({candidate.source or 'unknown'})"
        )

        print()
        print(
            f"  {label}"
        )

        display_page_url = _resolve_page_url(
            candidate.url,
            cfg.http_timeout,
        )

        print(
            f"    Page URL : "
            f"{display_page_url or '(none)'}"
        )

        print(
            f"    Lens URL : "
            f"{candidate.image_url or '(none)'}"
        )

        best_sim = -1.0
        best_data = None
        best_faces = 0
        best_url = ""
        best_source = ""

        errors = []

        # ----------------------------------------------------
        # 1. Lens/SerpApi image
        # ----------------------------------------------------

        if candidate.image_url:

            print(
                "    Trying Lens image..."
            )

            try:

                data = download_image(
                    candidate.image_url,
                    cfg.max_image_bytes,
                    cfg.http_timeout,
                )

                print(
                    f"    Downloaded "
                    f"{len(data):,} bytes"
                )

                candidate_faces = (
                    detector.detect_bytes(
                        data
                    )
                )

                if not candidate_faces:

                    errors.append(
                        "Lens image: no face"
                    )

                    print(
                        "    Lens image -> "
                        "no face detected"
                    )

                else:

                    sim = best_similarity(
                        reference,
                        [
                            face.embedding
                            for face
                            in candidate_faces
                        ],
                    )

                    print(
                        f"    Lens image -> "
                        f"{sim:.4f} "
                        f"({len(candidate_faces)} face(s))"
                    )

                    if sim > best_sim:

                        best_sim = sim
                        best_data = data
                        best_faces = len(
                            candidate_faces
                        )
                        best_url = (
                            candidate.image_url
                        )
                        best_source = (
                            "lens_image"
                        )

            except (
                FetchError,
                FaceError,
            ) as exc:

                errors.append(
                    f"Lens image: {exc}"
                )

                print(
                    f"    Lens image -> "
                    f"unavailable ({exc})"
                )

        else:

            errors.append(
                "Lens image: missing image URL"
            )

            print(
                "    Lens image -> "
                "missing URL"
            )

        # ----------------------------------------------------
        # 2. Source page metadata / og:image
        # ----------------------------------------------------

        page_meta = {}
        metadata_error = None

        if candidate.url:

            print(
                "    Fetching source page metadata..."
            )

            try:

                page_meta, metadata_error = (
                    fetch_page_metadata(
                        display_page_url,
                        cfg.http_timeout,
                    )
                )

            except Exception as exc:

                metadata_error = str(exc)

            if metadata_error:

                print(
                    f"    Metadata -> "
                    f"unavailable ({metadata_error})"
                )

                errors.append(
                    f"metadata: {metadata_error}"
                )

            og_image = (
                page_meta.get(
                    "og_image",
                    "",
                )
                or ""
            )

            if og_image:

                print(
                    f"    og:image : "
                    f"{og_image}"
                )

                # Don't download the same URL twice.
                if (
                    og_image
                    == candidate.image_url
                ):

                    print(
                        "    og:image is the same "
                        "as Lens image; skipping duplicate"
                    )

                else:

                    print(
                        "    Trying page og:image..."
                    )

                    try:

                        og_data = download_image(
                            og_image,
                            cfg.max_image_bytes,
                            cfg.http_timeout,
                        )

                        print(
                            f"    Downloaded "
                            f"{len(og_data):,} bytes"
                        )

                        og_faces = (
                            detector.detect_bytes(
                                og_data
                            )
                        )

                        if not og_faces:

                            errors.append(
                                "og:image: no face"
                            )

                            print(
                                "    og:image -> "
                                "no face detected"
                            )

                        else:

                            og_sim = best_similarity(
                                reference,
                                [
                                    face.embedding
                                    for face
                                    in og_faces
                                ],
                            )

                            print(
                                f"    og:image -> "
                                f"{og_sim:.4f} "
                                f"({len(og_faces)} face(s))"
                            )

                            # Prefer a verified image fetched from the source page
                            # over SerpApi's temporary Lens proxy.  It is acceptable
                            # for its score to be lower than the proxy as long as it
                            # independently clears the configured match threshold.
                            if og_sim >= cfg.match_threshold:

                                best_sim = og_sim
                                best_data = og_data
                                best_faces = len(
                                    og_faces
                                )
                                best_url = og_image
                                best_source = (
                                    "og_image"
                                )

                    except (
                        FetchError,
                        FaceError,
                    ) as exc:

                        errors.append(
                            f"og:image: {exc}"
                        )

                        print(
                            f"    og:image -> "
                            f"unavailable ({exc})"
                        )

            else:

                print(
                    "    og:image : "
                    "not available"
                )

        # ----------------------------------------------------
        # Candidate result
        # ----------------------------------------------------

        # Prefer the actual source-page image over SerpApi's Lens proxy.
        source_image_url = _source_image_url(
            page_meta,
            display_page_url,
        )
        if source_image_url:
            print(
                f"    Source image: {source_image_url}"
            )

        if best_sim < 0:

            error_text = (
                "; ".join(errors)
                if errors
                else "no usable candidate image"
            )

            print(
                f"    RESULT -> unavailable "
                f"({error_text})"
            )

            scored.append(
                ScoredCandidate(
                    candidate,
                    -1.0,
                    faces_found=0,
                    error=error_text,
                )
            )

            continue

        flag = ""

        if best_sim >= cfg.match_threshold:

            flag = "  ← MATCH"

        print(
            f"    RESULT -> "
            f"{best_sim:.4f} "
            f"via {best_source}"
            f"{flag}"
        )

        item = ScoredCandidate(
            candidate,
            best_sim,
            faces_found=best_faces,
            image_bytes=best_data,
        )

        # Attach diagnostic information without changing
        # matcher.py's dataclass.
        item.matched_image_url = best_url
        item.matched_image_source = best_source
        item.page_metadata = page_meta

        scored.append(
            item
        )

    # ========================================================
    # RANKING
    # ========================================================

    ranked = rank_candidates(
        scored
    )

    print()
    print(
        "============================================================"
    )
    print(
        "RANKED FACE MATCH RESULTS"
    )
    print(
        "============================================================"
    )

    for rank, item in enumerate(
        ranked,
        1,
    ):

        platform = (
            item.candidate.source
            or "unknown"
        )

        if item.ok:

            image_source = getattr(
                item,
                "matched_image_source",
                "unknown",
            )

            image_url = getattr(
                item,
                "matched_image_url",
                "",
            )

            print(
                f"  #{rank:<2} "
                f"{platform:<14} "
                f"similarity="
                f"{item.similarity:.4f} "
                f"faces="
                f"{item.faces_found} "
                f"via {image_source}"
            )

            page_url = _candidate_page_url(
                item.candidate,
                cfg.http_timeout,
            )

            print(
                f"      Page : "
                f"{page_url}"
            )

            print(
                f"      Image: "
                f"{image_url}"
            )

        else:

            print(
                f"  #{rank:<2} "
                f"{platform:<14} "
                f"unavailable: "
                f"{item.error or 'unknown error'}"
            )

    # ========================================================
    # Save score diagnostics
    # ========================================================

    debug["scores"] = []

    for item in ranked:

        debug["scores"].append(
            {
                "url": _candidate_page_url(
                    item.candidate,
                    cfg.http_timeout,
                ),
                "image_url": item.candidate.image_url,
                "matched_image_url": getattr(
                    item,
                    "matched_image_url",
                    "",
                ),
                "matched_image_source": getattr(
                    item,
                    "matched_image_source",
                    "",
                ),
                "similarity": float(
                    item.similarity
                ),
                "faces_found": item.faces_found,
                "error": item.error,
            }
        )

    # ========================================================
    # MATCH SELECTION
    # ========================================================

    match = select_match(
        scored,
        cfg.match_threshold,
    )

    if match is None:

        best = next(
            (
                item
                for item in ranked
                if item.ok
            ),
            None,
        )

        if best:

            fail(
                f"No candidate reached the threshold "
                f"{cfg.match_threshold:.2f} "
                f"(best was "
                f"{best.similarity:.3f})"
            )

            print()
            print(
                "  Best candidate:"
            )

            kv(
                "Platform",
                best.candidate.source
                or "unknown",
            )

            kv(
                "Similarity",
                f"{best.similarity:.4f}",
            )

            kv(
                "Page",
                _candidate_page_url(
                    best.candidate,
                    cfg.http_timeout,
                ),
            )

            kv(
                "Image source",
                getattr(
                    best,
                    "matched_image_source",
                    "",
                ),
            )

            kv(
                "Image URL",
                getattr(
                    best,
                    "matched_image_url",
                    "",
                ),
            )

        else:

            fail(
                f"No candidate reached the "
                f"threshold "
                f"{cfg.match_threshold:.2f}"
            )

        if args.save_results:

            _save_debug(
                debug
            )

        return 4

    # ========================================================
    # VERIFIED MATCH
    # ========================================================

    print()

    ok(
        "MATCH FOUND"
    )

    kv(
        "Similarity",
        f"{match.similarity:.4f} "
        f"(threshold "
        f"{cfg.match_threshold:.2f})",
    )

    kv(
        "Platform",
        match.candidate.source
        or "unknown",
    )

    match_page_url = _candidate_page_url(
        match.candidate,
        cfg.http_timeout,
    )

    kv(
        "URL",
        match_page_url,
    )

    kv(
        "Verified image",
        getattr(
            match,
            "matched_image_url",
            "",
        ),
    )

    kv(
        "Image source",
        getattr(
            match,
            "matched_image_source",
            "",
        ),
    )

    # ========================================================
    # MATCHING POST EXTRACTION
    # ========================================================

    banner(
        "matching post extraction"
    )

    # Fetch metadata from the resolved source page, not a Google
    # /goto intermediary URL returned by Lens.
    page_meta, err = fetch_page_metadata(
        match_page_url,
        cfg.http_timeout,
    )

    if err:

        warn(
            f"Public metadata unavailable: {err}"
        )

    record = build_post_record(
        match.candidate,
        page_meta,
    )

    # Prefer the real source/canonical page over Google's /goto URL.
    if match_page_url and not _is_google_goto_url(match_page_url):
        record["source_url"] = match_page_url
        record["canonical_url"] = (
            page_meta.get("canonical_url")
            or match_page_url
        )

    # Face verification metadata.
    record[
        "similarity"
    ] = round(
        float(match.similarity),
        6,
    )

    record[
        "match_threshold"
    ] = round(
        float(cfg.match_threshold),
        6,
    )

    record[
        "search_provider"
    ] = cfg.search_provider

    verified_image_url = getattr(
        match,
        "matched_image_url",
        "",
    )

    verified_image_source = getattr(
        match,
        "matched_image_source",
        "",
    )

    if verified_image_url:

        # Preserve the actual matched image URL instead of exposing
        # SerpApi's temporary image proxy URL as the source image.
        record[
            "image_url"
        ] = verified_image_url

        record[
            "verified_image_url"
        ] = verified_image_url

    if verified_image_source:

        record[
            "verified_image_source"
        ] = verified_image_source

    for key, value in record.items():

        text = str(value)

        if len(text) > 80:

            text = (
                text[:80]
                + "..."
            )

        kv(
            key,
            text,
        )

    debug[
        "post_record"
    ] = record

    # ========================================================
    # DATA FINGERPRINTING
    # ========================================================

    banner(
        "data fingerprinting"
    )

    image_hash = (
        sha256_bytes(
            match.image_bytes
        )
        if match.image_bytes
        else ""
    )

    data_hash = (
        sha256_canonical_json(
            record
        )
    )

    step(
        "Hashing canonical post metadata JSON "
        "(sorted keys, compact separators)"
    )

    print(
        "\n  SHA-256 (metadata):\n"
        f"  {data_hash}\n"
    )

    if image_hash:

        print(
            "  SHA-256 (matched image bytes):\n"
            f"  {image_hash}\n"
        )

    debug[
        "data_hash"
    ] = data_hash

    debug[
        "image_hash"
    ] = image_hash

    debug[
        "canonical_json"
    ] = canonical_json(
        record
    )

    if args.save_results:

        _save_debug(
            debug
        )

    # ========================================================
    # BLOCKCHAIN
    # ========================================================

    if args.no_chain:

        warn(
            "--no-chain set: skipping blockchain "
            "submission and verification."
        )

        return 0

    banner(
        "blockchain"
    )

    try:

        cfg.require_chain(
            need_key=True
        )

        verifier = BlockchainVerifier(
            cfg
        )

        kv(
            "Network chainId",
            verifier.w3.eth.chain_id,
        )

        kv(
            "Contract",
            cfg.contract_address,
        )

        step(
            "Submitting transaction..."
        )

        tx = verifier.store(
            data_hash,
            image_hash,
            record["source_url"],
        )

        ok(
            "Transaction mined"
        )

        kv(
            "Transaction",
            tx["tx_hash"],
        )

        kv(
            "Block",
            tx["block_number"],
        )

        kv(
            "Gas used",
            tx["gas_used"],
        )

        kv(
            "Explorer",
            tx["explorer_url"],
        )

    except (
        ChainError,
        ConfigError,
    ) as exc:

        fail(str(exc))

        return 5

    # ========================================================
    # BLOCKCHAIN VERIFICATION
    # ========================================================

    banner(
        "blockchain verification"
    )

    try:

        step(
            "Reading the record back from chain..."
        )

        onchain = verifier.read_back(
            data_hash,
            tx.get("record_id"),
        )

        recomputed = (
            sha256_canonical_json(
                record
            )
        )

        kv(
            "Network",
            f"chainId "
            f"{verifier.w3.eth.chain_id}",
        )

        print(
            f"\n  Stored hash:\n"
            f"  {onchain.data_hash}\n"
        )

        print(
            f"\n  Recomputed hash:\n"
            f"  {recomputed}\n"
        )

        kv(
            "Source URL on-chain",
            onchain.source_url,
        )

        kv(
            "Timestamp on-chain",
            onchain.timestamp,
        )

        kv(
            "Submitter",
            onchain.submitter,
        )

        if BlockchainVerifier.compare(
            recomputed,
            onchain.data_hash,
        ):

            print(
                "\n  Result:\n"
                "  ✅ VERIFIED - "
                "DATA INTEGRITY CONFIRMED\n"
            )

            return 0

        print(
            "\n  Result:\n"
            "  ❌ VERIFICATION FAILED - "
            "DATA HAS CHANGED\n"
        )

        return 6

    except ChainError as exc:

        fail(str(exc))

        return 6


# ============================================================
# Batch mode
# ============================================================

def run_batch(args, cfg) -> int:
    """Run the existing batch pipeline."""

    try:

        cfg.require_search()

        paths = discover_images(
            args.batch,
            recursive=not args.no_recursive,
            
        )
        if args.limit and args.limit > 0:
            paths = paths[:args.limit]


        if not paths:

            fail(
                "No input images found."
            )

            return 2

        banner(
            "batch face identification"
        )

        kv(
            "Images",
            len(paths),
        )

        kv(
            "Workers",
            cfg.batch_workers,
        )

        kv(
            "Face pool",
            cfg.face_pool_size,
        )

        # Create the same reverse-image-search provider used by
        # the single-image pipeline, then pass it to BatchRunner.
        provider = get_provider(
            cfg.search_provider,
            cfg,
        )

        step(
            f"Using reverse image search provider: {provider.name}"
        )

        runner = BatchRunner(
            cfg=cfg,
            provider=provider,
            workers=cfg.batch_workers,
            pool_size=cfg.face_pool_size,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        )

        results = runner.run(
            paths,
            resume=args.resume,
        )

        print_summary(
            runner.stats,
            results,
            runner.cache.stats(),
        )

        summary = {
            "images": len(results),
            "matched": sum(1 for r in results if getattr(r, "matched", False)),
            "processed": runner.stats.processed,
            "duplicates": runner.stats.duplicates,
            "skipped_resume": runner.stats.skipped_resume,
            "failures": dict(runner.stats.failures),
            "elapsed_seconds": round(runner.stats.elapsed, 3),
            "provider": provider.name,
        }

        report_paths = write_reports(
            results,
            Path(args.out),
            summary,
        )

        kv(
            "JSON report",
            report_paths["json"],
        )
        kv(
            "CSV report",
            report_paths["csv"],
        )

        if args.no_chain:

            warn(
                "--no-chain set: blockchain anchoring skipped."
            )

            return 0

        # Batch blockchain anchoring.
        try:

            cfg.require_chain(
                need_key=True
            )

            verifier = (
                BlockchainVerifier(cfg)
            )

            chunk_size = max(
                1,
                cfg.batch_chunk_size,
            )

            successful = [
                r
                for r in results
                if getattr(
                    r,
                    "matched",
                    False,
                )
            ]

            if not successful:

                warn(
                    "No verified matches to anchor."
                )

                return 0

            banner(
                "batch blockchain anchoring"
            )

            # Use the existing BlockchainVerifier API.  The current
            # verifier module exposes `store()` for individual records;
            # do not import or instantiate a non-existent BatchBlockchainVerifier.
            # We still process records in configured chunks so the batch
            # operation remains resumable/reportable without breaking the
            # existing contract interface.
            anchored = 0
            for start in range(
                0,
                len(successful),
                chunk_size,
            ):
                chunk = successful[start:start + chunk_size]
                step(
                    "Anchoring "
                    f"{len(chunk)} records..."
                )

                for result in chunk:
                    data_hash = getattr(result, "data_hash", "") or ""
                    image_hash = getattr(result, "image_hash", "") or ""
                    source_url = getattr(result, "source_url", "") or ""

                    if not data_hash:
                        warn(
                            f"Skipping {Path(result.image_path).name}: "
                            "missing data hash"
                        )
                        continue

                    tx = verifier.store(
                        data_hash,
                        image_hash,
                        source_url,
                    )
                    anchored += 1
                    kv(
                        Path(result.image_path).name,
                        f"{tx['tx_hash']} "
                        f"(block {tx['block_number']})",
                    )

                ok(
                    f"Anchored {len(chunk)} records "
                    f"({anchored}/{len(successful)} total)"
                )

            kv("Records anchored", anchored)
            return 0

        except (
            ChainError,
            ConfigError,
        ) as exc:

            fail(str(exc))

            return 5

    except (
        SearchError,
        ConfigError,
    ) as exc:

        fail(str(exc))

        return 3


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )