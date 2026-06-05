"""MediaPipe Face backend using the Tasks API (mediapipe >= 0.10)."""

from __future__ import annotations

import os
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

os.environ["GLOG_MINLOGLEVEL"] = "2"

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_CACHE = Path.home() / ".cache" / "minecraft_cv" / "face_landmarker.task"


def _ensure_model() -> Path:
    """Download the FaceLandmarker model bundle to a local cache if not present."""
    if _MODEL_CACHE.exists():
        return _MODEL_CACHE
    _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mediapipe] Downloading face_landmarker model to {_MODEL_CACHE} …")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    except Exception:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(_MODEL_URL, context=ctx) as resp, open(_MODEL_CACHE, "wb") as f:
        f.write(resp.read())
    return _MODEL_CACHE


class MediaPipeFaceTracker:
    """Face tracker backed by MediaPipe Tasks FaceLandmarker (CPU-only)."""

    def __init__(
        self,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for the face tracking backend.") from exc

        model_path = _ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._detector: Any = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp_image_format = mp.ImageFormat.SRGB
        self._mp_image_cls = mp.Image
        self._start_ns = time.monotonic_ns()

    def _timestamp_ms(self) -> int:
        return (time.monotonic_ns() - self._start_ns) // 1_000_000

    def detect(self, rgb_frame: np.ndarray) -> tuple[np.ndarray | None, dict[str, float] | None]:
        """Run FaceLandmarker on an RGB frame.

        Returns:
            A tuple of (landmarks_array, blendshapes_dict).
            landmarks_array is an array of shape (478, 3).
            blendshapes_dict maps blendshape names to their scores [0, 1].
            Returns (None, None) if no face is detected.
        """
        mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb_frame)
        result = self._detector.detect_for_video(mp_image, self._timestamp_ms())

        if not result.face_landmarks:
            return None, None

        # Extract landmarks
        landmarks = result.face_landmarks[0]
        coords = np.empty((len(landmarks), 3), dtype=np.float32)
        for j, lm in enumerate(landmarks):
            coords[j] = (lm.x, lm.y, lm.z)

        # Extract blendshapes
        blendshapes = {}
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                blendshapes[cat.category_name] = cat.score

        return coords, blendshapes

    def close(self) -> None:
        """Close the FaceLandmarker and free its resources."""
        detector = getattr(self, "_detector", None)
        if detector is not None:
            detector.close()
            self._detector = None
