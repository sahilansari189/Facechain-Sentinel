"""Face detection + ArcFace embedding generation using InsightFace."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


class FaceError(RuntimeError):
    """Raised for any recoverable face-pipeline problem."""


@dataclass
class DetectedFace:
    bbox: Sequence[float]          # x1, y1, x2, y2
    det_score: float
    embedding: np.ndarray          # normalised ArcFace embedding
    age: Optional[int] = None
    sex: Optional[str] = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    def summary(self) -> str:
        x1, y1, x2, y2 = (int(v) for v in self.bbox)
        return f"bbox=({x1},{y1})-({x2},{y2}) score={self.det_score:.3f} dim={self.embedding.shape[0]}"


def _normalise(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise FaceError("degenerate (zero-norm) embedding")
    return vec / norm


class FaceDetector:
    """Thin wrapper around ``insightface.app.FaceAnalysis``.

    The heavy model is loaded lazily so importing this module (e.g. in unit
    tests) stays cheap and dependency-free.
    """

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640, providers: Optional[list] = None):
        self.model_name = model_name
        self.det_size = det_size
        self.providers = providers or ["CPUExecutionProvider"]
        self._app = None

    def _ensure_model(self):
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:  # pragma: no cover
                raise FaceError(
                    "insightface is not installed. Run: pip install -r requirements.txt"
                ) from exc
            app = FaceAnalysis(name=self.model_name, providers=self.providers)
            app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
            self._app = app
        return self._app

    # ------------------------------------------------------------------ #

    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        """Decode raw image bytes into a BGR numpy array."""
        import cv2

        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise FaceError("invalid or unsupported image data")
        return img

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise FaceError(f"cannot read image file: {exc}") from exc
        if not data:
            raise FaceError("image file is empty")
        return FaceDetector.decode_image(data)

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        app = self._ensure_model()
        faces = app.get(image)
        out: List[DetectedFace] = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                emb = getattr(f, "embedding", None)
            if emb is None:
                continue
            out.append(
                DetectedFace(
                    bbox=[float(v) for v in f.bbox],
                    det_score=float(getattr(f, "det_score", 0.0)),
                    embedding=_normalise(emb),
                    age=getattr(f, "age", None),
                    sex=getattr(f, "sex", None),
                )
            )
        return out

    def detect_bytes(self, data: bytes) -> List[DetectedFace]:
        return self.detect(self.decode_image(data))

    def primary_face(self, faces: List[DetectedFace]) -> DetectedFace:
        """Pick the largest, highest-confidence face; raise if none."""
        if not faces:
            raise FaceError("no face detected in the image")
        return max(faces, key=lambda f: (f.area, f.det_score))


class DetectorPool:
    """A small pool of ready InsightFace sessions for multi-threaded batches.

    ONNXRuntime sessions are not safe to share across threads, so each worker
    checks out its own detector. The pool is capped (``size``) because every
    instance costs ~300 MB of RAM; the first instance is created eagerly so a
    batch fails fast if the model cannot be loaded at all.
    """

    def __init__(self, size: int = 2, model_name: str = "buffalo_l", det_size: int = 640,
                 providers: Optional[list] = None):
        import queue
        import threading

        self.size = max(1, int(size))
        self._model_name = model_name
        self._det_size = det_size
        self._providers = providers
        self._pool: "queue.LifoQueue" = queue.LifoQueue()
        self._created = 0
        self._lock = threading.Lock()

    def _new(self) -> "FaceDetector":
        det = FaceDetector(self._model_name, self._det_size, self._providers)
        det._ensure_model()
        return det

    def warm(self) -> "FaceDetector":
        """Create (and return to the pool) the first detector, loading the model."""
        det = self.acquire()
        self.release(det)
        return det

    def acquire(self) -> "FaceDetector":
        import queue

        try:
            return self._pool.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self.size:
                self._created += 1
                make_new = True
            else:
                make_new = False
        if make_new:
            try:
                return self._new()
            except Exception:
                with self._lock:
                    self._created -= 1
                raise
        return self._pool.get()

    def release(self, detector: "FaceDetector") -> None:
        self._pool.put(detector)

    @contextmanager
    def lease(self):
        det = self.acquire()
        try:
            yield det
        finally:
            self.release(det)
