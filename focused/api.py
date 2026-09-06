from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MAIN_PY = BASE_DIR / "src" / "main.py"

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_FILE_SIZE = 20 * 1024 * 1024
PROCESS_TIMEOUT = 180
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="FaceChain Sentinel API",
    version="1.0.0",
    description=(
        "API bridge for the FaceChain Sentinel face "
        "identification and verification pipeline."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_origin_regex=r"https://([a-z0-9-]+\.)*vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BACKGROUND JOB MANAGEMENT
# ============================================================

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def create_job(
    job_id: str,
    filename: str,
) -> None:
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "Queued",
            "progress": 0,
            "filename": filename,
            "success": None,
            "result": None,
            "message": None,
            "attempts": 0,
            "stdout": "",
            "stderr": "",
            "created_at": time.time(),
            "completed_at": None,
        }


def update_job(
    job_id: str,
    **updates: Any,
) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def get_job(
    job_id: str,
) -> dict[str, Any] | None:
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return None

        return dict(job)


# ============================================================
# PROGRESS DETECTION
# ============================================================

def update_progress_from_output(
    job_id: str,
    output: str,
) -> None:
    """
    Translate CLI output into approximate frontend progress.

    The existing CLI does not expose native progress values,
    so these stages are inferred from its output.
    """

    checks = [
        (
            "FACE IDENTIFICATION",
            "Face identification",
            10,
        ),
        (
            "Loading image:",
            "Loading image",
            15,
        ),
        (
            "Running detector",
            "Detecting face",
            20,
        ),
        (
            "Faces detected:",
            "Face detected",
            25,
        ),
        (
            "Embedding generated",
            "Generating face embedding",
            30,
        ),
        (
            "Selected face",
            "Selecting face",
            35,
        ),
        (
            "WEB SEARCH",
            "Preparing web search",
            40,
        ),
        (
            "Performing LIVE reverse image search",
            "Searching the web",
            50,
        ),
        (
            "Search upload",
            "Uploading image for search",
            48,
        ),
        (
            "Found ",
            "Reverse search completed",
            60,
        ),
        (
            "FACE MATCHING",
            "Matching candidate faces",
            65,
        ),
        (
            "Candidate",
            "Comparing candidates",
            70,
        ),
        (
            "MATCH FOUND",
            "Face match found",
            82,
        ),
        (
            "MATCHING POST EXTRACTION",
            "Extracting source metadata",
            86,
        ),
        (
            "DATA FINGERPRINTING",
            "Generating data fingerprints",
            90,
        ),
        (
            "SHA-256 (metadata)",
            "Generating SHA-256 fingerprint",
            93,
        ),
        (
            "BLOCKCHAIN",
            "Blockchain verification",
            96,
        ),
        (
            "Transaction mined",
            "Blockchain transaction confirmed",
            98,
        ),
        (
            "VERIFIED - DATA INTEGRITY CONFIRMED",
            "Verification complete",
            100,
        ),
        (
            "BLOCKCHAIN EVIDENCE VERIFIED",
            "Verification complete",
            100,
        ),
    ]

    latest: tuple[str, int] | None = None

    for pattern, stage, progress in checks:
        if pattern.lower() in output.lower():
            latest = (stage, progress)

    if latest is not None:
        stage, progress = latest

        update_job(
            job_id,
            stage=stage,
            progress=progress,
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_upload(
    filename: str | None,
    content_type: str | None,
) -> str:

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image type: {extension}. "
                f"Allowed types: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    if content_type and not content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    return extension


# ============================================================
# OUTPUT HELPERS
# ============================================================

def decode_output(
    value: Any,
) -> str:

    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


def is_transient_search_error(
    stdout: str,
    stderr: str,
) -> bool:

    combined = (
        f"{stdout}\n{stderr}"
    ).lower()

    patterns = [
        "google lens hasn't returned any results",
        "google lens has not returned any results",
        "no results for this query",
        "search_error",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "timed out",
        "timeout",
    ]

    return any(
        pattern in combined
        for pattern in patterns
    )


# ============================================================
# OUTPUT PARSER
# ============================================================

def parse_pipeline_output(
    output: str,
) -> dict[str, Any]:
    """
    Convert the existing CLI output into structured JSON.

    This parses the existing FaceChain Sentinel CLI output
    without modifying the face-identification pipeline.
    """

    result: dict[str, Any] = {
        "face": {
            "detected": False,
            "count": 0,
            "confidence": None,
            "embedding_dimensions": None,
        },
        "match": {
            "found": False,
            "similarity": None,
            "threshold": None,
            "platform": None,
            "source_url": None,
            "verified_image_url": None,
            "image_source": None,
        },
        "candidates": [],
        "record": {},
        "fingerprints": {
            "metadata_sha256": None,
            "image_sha256": None,
        },
        "blockchain": {
            "anchored": False,
            "verified": False,
            "transaction": None,
            "block": None,
            "explorer": None,
            "evidence_id": None,
        },
    }

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_match = re.search(
        r"Faces detected:\s*(\d+)",
        output,
        re.IGNORECASE,
    )

    if face_match:
        count = int(
            face_match.group(1)
        )

        result["face"]["detected"] = count > 0
        result["face"]["count"] = count

    face_details = re.search(
        r"face\[0\].*?"
        r"score=([0-9.]+).*?"
        r"dim=(\d+)",
        output,
        re.IGNORECASE,
    )

    if face_details:
        result["face"]["confidence"] = float(
            face_details.group(1)
        )

        result["face"]["embedding_dimensions"] = int(
            face_details.group(2)
        )

    embedding_match = re.search(
        r"Embedding dims\s+(\d+)",
        output,
        re.IGNORECASE,
    )

    if embedding_match:
        result["face"]["embedding_dimensions"] = int(
            embedding_match.group(1)
        )

    # --------------------------------------------------------
    # MAIN MATCH
    # --------------------------------------------------------

    similarity_match = re.search(
        r"Similarity\s+([0-9.]+)\s+"
        r"\(threshold\s+([0-9.]+)\)",
        output,
        re.IGNORECASE,
    )

    if similarity_match:
        result["match"]["found"] = True

        result["match"]["similarity"] = float(
            similarity_match.group(1)
        )

        result["match"]["threshold"] = float(
            similarity_match.group(2)
        )

    platform_match = re.search(
        r"Platform\s+(.+)",
        output,
        re.IGNORECASE,
    )

    if platform_match:
        result["match"]["platform"] = (
            platform_match.group(1).strip()
        )

    url_match = re.search(
        r"^\s+URL\s+(.+)$",
        output,
        re.MULTILINE | re.IGNORECASE,
    )

    if url_match:
        result["match"]["source_url"] = (
            url_match.group(1).strip()
        )

    verified_image_match = re.search(
        r"Verified image\s+(.+)",
        output,
        re.IGNORECASE,
    )

    if verified_image_match:
        result["match"]["verified_image_url"] = (
            verified_image_match.group(1).strip()
        )

    image_source_match = re.search(
        r"Image source\s+(.+)",
        output,
        re.IGNORECASE,
    )

    if image_source_match:
        result["match"]["image_source"] = (
            image_source_match.group(1).strip()
        )

    # --------------------------------------------------------
    # RANKED CANDIDATES
    # --------------------------------------------------------

    candidate_pattern = re.compile(
        r"#(\d+)\s+"
        r"(.+?)\s+"
        r"similarity=([0-9.]+)\s+"
        r"faces=(\d+)\s+"
        r"via\s+([^\s]+)",
        re.IGNORECASE,
    )

    for match in candidate_pattern.finditer(
        output
    ):
        result["candidates"].append(
            {
                "rank": int(match.group(1)),
                "platform": match.group(2).strip(),
                "similarity": float(match.group(3)),
                "faces": int(match.group(4)),
                "source": match.group(5),
            }
        )

    # --------------------------------------------------------
    # MATCHING POST EXTRACTION
    # --------------------------------------------------------

    extraction_fields = {
        "source_url": r"source_url\s+(.+)",
        "platform": r"platform\s+(.+)",
        "title": r"title\s+(.+)",
        "description": r"description\s+(.+)",
        "image_url": r"image_url\s+(.+)",
        "site_name": r"site_name\s+(.+)",
        "published_time": r"published_time\s*(.*)",
        "author": r"author\s*(.*)",
        "canonical_url": r"canonical_url\s+(.+)",
        "retrieved_at": r"retrieved_at\s+(.+)",
        "is_exact_match": r"is_exact_match\s+(.+)",
        "search_type": r"search_type\s+(.+)",
        "similarity": r"similarity\s+([0-9.]+)",
        "match_threshold": r"match_threshold\s+([0-9.]+)",
        "search_provider": r"search_provider\s+(.+)",
        "verified_image_url": r"verified_image_url\s+(.+)",
        "verified_image_source": (
            r"verified_image_source\s+(.+)"
        ),
    }

    for field, pattern in extraction_fields.items():

        field_match = re.search(
            pattern,
            output,
            re.IGNORECASE,
        )

        if not field_match:
            continue

        value = field_match.group(1).strip()

        if field in {
            "similarity",
            "match_threshold",
        }:
            try:
                value = float(value)
            except ValueError:
                pass

        elif field == "is_exact_match":
            value = value.lower() == "true"

        result["record"][field] = value

    # --------------------------------------------------------
    # SHA-256 FINGERPRINTS
    # --------------------------------------------------------

    metadata_hash = re.search(
        r"SHA-256 \(metadata\):\s*"
        r"([a-fA-F0-9]{64})",
        output,
        re.IGNORECASE,
    )

    if metadata_hash:
        result["fingerprints"][
            "metadata_sha256"
        ] = metadata_hash.group(1)

    image_hash = re.search(
        r"SHA-256 \(matched image bytes\):\s*"
        r"([a-fA-F0-9]{64})",
        output,
        re.IGNORECASE,
    )

    if image_hash:
        result["fingerprints"][
            "image_sha256"
        ] = image_hash.group(1)

    # --------------------------------------------------------
    # BLOCKCHAIN
    # --------------------------------------------------------

    if (
        "Transaction mined" in output
        or "Transaction          0x" in output
    ):
        result["blockchain"]["anchored"] = True

    if (
        "VERIFIED - DATA INTEGRITY CONFIRMED" in output
        or "VERIFIED - DATA + IMAGE INTEGRITY CONFIRMED" in output
    ):
        result["blockchain"]["verified"] = True

    transaction_match = re.search(
        r"Transaction\s+(0x[a-fA-F0-9]+)",
        output,
    )

    if transaction_match:
        result["blockchain"][
            "transaction"
        ] = transaction_match.group(1)

    block_match = re.search(
        r"Block\s+(\d+)",
        output,
    )

    if block_match:
        result["blockchain"]["block"] = int(
            block_match.group(1)
        )

    explorer_match = re.search(
        r"Explorer\s+(https?://\S+)",
        output,
    )

    if explorer_match:
        result["blockchain"][
            "explorer"
        ] = explorer_match.group(1)

    evidence_match = re.search(
        r"Evidence bundle\s+(.+)", output, re.IGNORECASE
    )
    if evidence_match:
        result["blockchain"]["evidence_id"] = Path(
            evidence_match.group(1).strip()
        ).name

    # --------------------------------------------------------
    # OVERALL STATUS
    # --------------------------------------------------------

    if result["match"]["found"]:
        result["status"] = "matched"

    elif result["face"]["detected"]:
        result["status"] = "no_match"

    else:
        result["status"] = "no_face"

    return result


# ============================================================
# BACKGROUND VERIFICATION WORKER
# ============================================================

def run_verification_job(
    job_id: str,
    temp_path: Path,
    filename: str,
    context: str = "",
    handle: str = "",
    platform: str = "",
    threshold: float | None = None,
    provider: str = "",
    no_chain: bool = False,
) -> None:
    """
    Run the existing FaceChain Sentinel CLI in the background.
    """

    env = os.environ.copy()

    # Force UTF-8 so Windows does not fail on ✓ characters.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable,
        str(MAIN_PY),
        "--image",
        str(temp_path),
    ]

    if context:
        command.extend(["--context", context])
    if handle:
        command.extend(["--handle", handle])
    if platform:
        command.extend(["--platform", platform])
    if threshold is not None:
        command.extend(["--threshold", str(threshold)])
    if provider:
        command.extend(["--provider", provider])
    if no_chain:
        command.append("--no-chain")

    stdout = ""
    stderr = ""
    process = None
    attempts_used = 0

    try:

        update_job(
            job_id,
            status="processing",
            stage="Starting pipeline",
            progress=5,
        )

        # ====================================================
        # RETRY LOOP
        # ====================================================

        for attempt in range(
            1,
            MAX_ATTEMPTS + 1,
        ):

            attempts_used = attempt

            update_job(
                job_id,
                attempts=attempt,
                stage=(
                    "Retrying pipeline"
                    if attempt > 1
                    else "Starting pipeline"
                ),
            )

            try:

                process = subprocess.Popen(
                    command,
                    cwd=str(BASE_DIR),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                output_lines: list[str] = []

                start_time = time.time()

                # =================================================
                # STREAM CLI OUTPUT
                # =================================================

                while True:

                    if process.stdout is None:
                        break

                    line = process.stdout.readline()

                    if line:

                        output_lines.append(line)

                        stdout = "".join(
                            output_lines
                        )

                        update_progress_from_output(
                            job_id,
                            stdout,
                        )

                        update_job(
                            job_id,
                            stdout=stdout[-20000:],
                        )

                    if process.poll() is not None:
                        break

                    if (
                        time.time() - start_time
                        > PROCESS_TIMEOUT
                    ):

                        process.kill()

                        update_job(
                            job_id,
                            status="timeout",
                            stage="Pipeline timeout",
                            progress=100,
                            success=False,
                            attempts=attempts_used,
                            message=(
                                f"Pipeline exceeded "
                                f"{PROCESS_TIMEOUT}s timeout."
                            ),
                            stdout=stdout[-20000:],
                            completed_at=time.time(),
                        )

                        return

                process.wait()

                stdout = "".join(
                    output_lines
                )

                update_job(
                    job_id,
                    stdout=stdout[-20000:],
                )

                # =================================================
                # SUCCESS
                # =================================================

                if process.returncode == 0:
                    break

                # =================================================
                # NON-TRANSIENT FAILURE
                # =================================================

                if not is_transient_search_error(
                    stdout,
                    "",
                ):
                    break

                # =================================================
                # RETRY
                # =================================================

                if attempt < MAX_ATTEMPTS:

                    update_job(
                        job_id,
                        stage=(
                            "Temporary search failure — "
                            "retrying"
                        ),
                    )

                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )

            except Exception as exc:

                stderr = str(exc)

                if attempt >= MAX_ATTEMPTS:

                    update_job(
                        job_id,
                        status="error",
                        stage="Pipeline error",
                        progress=100,
                        success=False,
                        attempts=attempts_used,
                        message=str(exc),
                        stderr=stderr,
                        completed_at=time.time(),
                    )

                    return

        # ========================================================
        # PROCESS DID NOT START
        # ========================================================

        if process is None:

            update_job(
                job_id,
                status="error",
                stage="Pipeline failed to start",
                progress=100,
                success=False,
                attempts=attempts_used,
                message="Pipeline did not start.",
                completed_at=time.time(),
            )

            return

        # ========================================================
        # PIPELINE FAILED
        # ========================================================

        if process.returncode != 0:

            parsed = parse_pipeline_output(
                stdout
            )

            update_job(
                job_id,
                status="error",
                stage="Pipeline failed",
                progress=100,
                success=False,
                attempts=attempts_used,
                result=parsed,
                message=(
                    "Face verification pipeline failed."
                ),
                completed_at=time.time(),
            )

            return

        # ========================================================
        # SUCCESSFUL PIPELINE
        # ========================================================

        update_job(
            job_id,
            stage="Processing final result",
            progress=99,
        )

        parsed = parse_pipeline_output(
            stdout
        )

        update_job(
            job_id,
            status="completed",
            stage="Verification complete",
            progress=100,
            success=True,
            attempts=attempts_used,
            result=parsed,
            message=(
                "Verification completed successfully."
            ),
            completed_at=time.time(),
        )

    except Exception as exc:

        update_job(
            job_id,
            status="error",
            stage="Unexpected error",
            progress=100,
            success=False,
            attempts=attempts_used,
            message=(
                f"Verification failed: {exc}"
            ),
            stderr=str(exc),
            completed_at=time.time(),
        )

    finally:

        # ========================================================
        # CLEANUP TEMP IMAGE
        # ========================================================

        try:
            temp_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


# ============================================================
# INDEPENDENT EVIDENCE RE-VERIFICATION
# ============================================================

EVIDENCE_ROOT = (BASE_DIR / "evidence").resolve()


def _safe_evidence_path(evidence_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", evidence_id):
        raise HTTPException(status_code=400, detail="Invalid evidence ID.")
    path = (EVIDENCE_ROOT / evidence_id).resolve()
    if EVIDENCE_ROOT not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid evidence path.")
    return path


@app.post("/api/reverify")
def reverify_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    evidence_id = str(payload.get("evidence_id", "")).strip()
    if not evidence_id:
        raise HTTPException(status_code=400, detail="evidence_id is required.")

    bundle = _safe_evidence_path(evidence_id)
    if not (bundle / "metadata.json").exists():
        raise HTTPException(status_code=404, detail="Evidence bundle not found.")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable,
        str(MAIN_PY),
        "--reverify",
        str(bundle),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Evidence re-verification timed out.") from exc

    output = completed.stdout or ""
    if completed.returncode not in {0, 6}:
        raise HTTPException(
            status_code=500,
            detail=(completed.stderr or output or "Evidence re-verification failed."),
        )

    # Parse ONLY the explicit verification-status lines.
    # Do not use the raw "Metadata hash" / "Image hash" labels above,
    # because the CLI prints saved, on-chain, and recomputed values first.
    def status_after(label: str) -> bool:
        pattern = rf"^\s*{re.escape(label)}\s+(✓ MATCH|✗ MISMATCH)\s*$"
        match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
        return bool(match and match.group(1).strip().upper() == "✓ MATCH")

    metadata_hash_match = status_after("Metadata hash")
    image_hash_match = status_after("Image hash")
    source_url_match = status_after("Source URL")

    # A successful subprocess is not the same thing as successful
    # evidence verification. All three independent checks must pass.
    verified = (
        completed.returncode == 0
        and metadata_hash_match
        and image_hash_match
        and source_url_match
    )

    return {
        "success": verified,
        "verified": verified,
        "evidence_id": evidence_id,
        "metadata_hash_match": metadata_hash_match,
        "image_hash_match": image_hash_match,
        "source_url_match": source_url_match,
        "output": output,
        "status": "verified" if verified else "tampered",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health() -> dict[str, Any]:

    return {
        "status": "ok",
        "service": "facechain-sentinel",
        "version": "1.0.0",
        "pipeline": str(MAIN_PY),
        "pipeline_exists": MAIN_PY.exists(),
        "active_jobs": len(jobs),
    }


# ============================================================
# START VERIFICATION
# ============================================================

@app.post(
    "/api/verify",
    status_code=202,
)
async def verify_image(
    file: UploadFile = File(...),
    context: str = Form(default=""),
    handle: str = Form(default=""),
    platform: str = Form(default=""),
    threshold: float | None = Form(default=None),
    provider: str = Form(default=""),
    no_chain: bool = Form(default=False),
) -> dict[str, Any]:

    extension = validate_upload(
        file.filename,
        file.content_type,
    )

    data = await file.read()

    if not data:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(data) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum allowed size is 20 MB."
            ),
        )

    if threshold is not None and not 0 <= threshold <= 1:
        raise HTTPException(
            status_code=400,
            detail="Threshold must be between 0 and 1.",
        )

    if provider and provider not in {"serpapi", "bing", "tineye"}:
        raise HTTPException(
            status_code=400,
            detail="Provider must be serpapi, bing, or tineye.",
        )

    if platform and platform not in {"instagram", "twitter", "linkedin"}:
        raise HTTPException(
            status_code=400,
            detail="Platform must be instagram, twitter, or linkedin.",
        )

    if not MAIN_PY.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                f"Pipeline entrypoint not found: "
                f"{MAIN_PY}"
            ),
        )

    job_id = uuid.uuid4().hex

    temp_path: Path | None = None

    try:

        # ========================================================
        # CREATE TEMPORARY IMAGE
        # ========================================================

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            prefix=f"facechain_{job_id}_",
            delete=False,
        ) as temp_file:

            temp_file.write(data)

            temp_path = Path(
                temp_file.name
            )

        # ========================================================
        # REGISTER JOB
        # ========================================================

        create_job(
            job_id,
            file.filename or "uploaded_image",
        )

        # ========================================================
        # START BACKGROUND WORKER
        # ========================================================

        worker = threading.Thread(
            target=run_verification_job,
            args=(
                job_id,
                temp_path,
                file.filename
                or "uploaded_image",
                context.strip(),
                handle.strip(),
                platform.strip(),
                threshold,
                provider.strip(),
                no_chain,
            ),
            daemon=True,
            name=f"verification-{job_id}",
        )

        worker.start()

        # Worker now owns the temp file.
        temp_path = None

        # ========================================================
        # IMMEDIATE RESPONSE
        # ========================================================

        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": (
                "Verification job started."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:

        if temp_path is not None:

            try:
                temp_path.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to start verification: "
                f"{exc}"
            ),
        ) from exc


# ============================================================
# GET VERIFICATION STATUS
# ============================================================

@app.get(
    "/api/verify/{job_id}"
)
def verification_status(
    job_id: str,
) -> dict[str, Any]:

    job = get_job(job_id)

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Verification job not found.",
        )

    response: dict[str, Any] = {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "filename": job["filename"],
        "success": job["success"],
        "attempts": job["attempts"],
        "message": job["message"],
    }

    if job["status"] == "completed":

        response["result"] = job["result"]

    elif job["status"] == "error":

        response["result"] = job["result"]

    elif job["status"] == "timeout":

        response["result"] = None

    return response


# ============================================================
# DELETE VERIFICATION JOB
# ============================================================

@app.delete(
    "/api/verify/{job_id}"
)
def delete_verification_job(
    job_id: str,
) -> dict[str, Any]:

    with jobs_lock:

        if job_id not in jobs:

            raise HTTPException(
                status_code=404,
                detail="Verification job not found.",
            )

        del jobs[job_id]

    return {
        "success": True,
        "job_id": job_id,
        "message": "Job removed.",
    }
