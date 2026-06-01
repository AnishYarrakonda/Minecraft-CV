"""MediaPipe Hands backend using the Tasks API (mediapipe >= 0.10).

Wraps ``mediapipe.tasks.python.vision.HandLandmarker`` in VIDEO running mode.
MediaPipe runs on CPU only (no MPS/CUDA path); do not attempt to move it to a GPU device.
Imported lazily so the package imports without MediaPipe present (tests use synthetic
landmarks, never this backend).
"""

from __future__ import annotations

import os
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from minecraft_cv.tracking.tracker import HandResult, HandTracker

_NUM_LANDMARKS = 21
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_CACHE = Path.home() / ".cache" / "minecraft_cv" / "hand_landmarker.task"


def _ensure_model() -> Path:
    """Download the HandLandmarker model bundle to a local cache if not present."""
    if _MODEL_CACHE.exists():
        return _MODEL_CACHE
    _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mediapipe] Downloading hand_landmarker model to {_MODEL_CACHE} …")
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(_MODEL_URL, context=ctx) as resp, open(_MODEL_CACHE, "wb") as f:
        f.write(resp.read())
    return _MODEL_CACHE


class MediaPipeHandTracker(HandTracker):
    """Hand tracker backed by MediaPipe Tasks HandLandmarker (CPU-only).

    Uses VIDEO running mode so MediaPipe tracks landmarks across frames rather than
    running full detection on every frame — significantly faster on subsequent frames.
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the MediaPipe HandLandmarker.

        Args:
            max_num_hands: Maximum hands to detect (1-2 for this controller).
            min_detection_confidence: Minimum confidence to start a track.
            min_tracking_confidence: Minimum confidence to keep tracking across frames.

        Raises:
            RuntimeError: If MediaPipe is not importable.
        """
        try:
            # Must be set before the C extension initialises glog — too late at module level.
            os.environ.setdefault("GLOG_MINLOGLEVEL", "2")
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "mediapipe is required for the default tracking backend. Install it with "
                "'pip install mediapipe', or select a different backend."
            ) from exc

        model_path = _ensure_model()
        # Force CPU delegate: mediapipe ≥0.10.14 defaults to GPU on Apple Silicon,
        # which tries to create a Metal/GL context on the worker thread — fatal on
        # macOS when Qt already owns the main-thread GL context (SIGTRAP).
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector: Any = mp_vision.HandLandmarker.create_from_options(options)
        self._mp_image_format = mp.ImageFormat.SRGB
        self._mp_image_cls = mp.Image
        self._start_ns = time.monotonic_ns()

    def _timestamp_ms(self) -> int:
        """Monotonically increasing milliseconds since tracker construction."""
        return (time.monotonic_ns() - self._start_ns) // 1_000_000

    def detect(self, rgb_frame: np.ndarray) -> list[HandResult]:
        """Run HandLandmarker on an RGB frame and return per-hand landmark results.

        Args:
            rgb_frame: ``(H, W, 3)`` ``uint8`` RGB frame. The BGR->RGB conversion is the
                pipeline's responsibility (done once per frame, before this call).

        Returns:
            Zero to ``max_num_hands`` :class:`HandResult` objects.
        """
        mp_image = self._mp_image_cls(
            image_format=self._mp_image_format, data=rgb_frame
        )
        result = self._detector.detect_for_video(mp_image, self._timestamp_ms())
        if not result.hand_landmarks:
            return []
        out: list[HandResult] = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            # Protobuf -> ndarray. The per-landmark read is unavoidable (21 fields), but all
            # downstream math (distances, joystick) stays vectorized on this array.
            coords = np.empty((_NUM_LANDMARKS, 3), dtype=np.float32)
            for j, lm in enumerate(hand_landmarks):
                coords[j] = (lm.x, lm.y, lm.z)
            label, score = "Unknown", 1.0
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0]
                label = cat.category_name  # "Left" or "Right"
                score = float(cat.score)
            out.append(HandResult(landmarks=coords, handedness=label, score=score))
        return out

    def close(self) -> None:
        """Close the HandLandmarker and free its resources."""
        detector = getattr(self, "_detector", None)
        if detector is not None:
            detector.close()
            self._detector = None
