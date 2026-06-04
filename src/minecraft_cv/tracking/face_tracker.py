"""MediaPipe FaceLandmarker tracking backend."""

from __future__ import annotations

import logging
from typing import Any

import mediapipe as mp
import numpy as np

# MediaPipe aliases for conciseness
mp_face_landmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

logger = logging.getLogger(__name__)


class FaceResult:
    """Wrapper around MediaPipe FaceLandmarker result."""

    def __init__(
        self,
        blendshapes: dict[str, float] | None = None,
        landmarks: np.ndarray | None = None,
    ) -> None:
        self.blendshapes = blendshapes or {}
        self.landmarks = landmarks  # (478, 3) float array if present


class FaceTracker:
    """Real-time face blendshape tracker using MediaPipe."""

    def __init__(
        self,
        model_path: str = "models/face_landmarker.task",
        device: str = "cpu",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the MediaPipe FaceLandmarker."""
        # Use CPU delegate to prevent GL context collisions with PySide6/Qt
        delegate = mp.tasks.BaseOptions.Delegate.CPU
        if device.lower() in ("mps", "cuda"):
            logger.warning("FaceTracker ignoring %r device; forcing CPU to prevent Qt GL conflict.", device)

        self._base_options = BaseOptions(
            model_asset_path=model_path,
            delegate=delegate,
        )

        options = FaceLandmarkerOptions(
            base_options=self._base_options,
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
        )

        self._landmarker = mp_face_landmarker.create_from_options(options)
        logger.info("FaceLandmarker initialized from %s", model_path)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> FaceResult:
        """Process a single frame and return face blendshapes/landmarks.

        Args:
            rgb_frame: (H, W, 3) uint8 array in RGB color space.
            timestamp_ms: Monotonically increasing frame timestamp in milliseconds.

        Returns:
            FaceResult containing blendshapes dict and optional landmarks array.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_blendshapes:
            return FaceResult()

        # result.face_blendshapes is a list of lists (one per face) of Category objects.
        # We requested num_faces=1, so we take the first face.
        categories = result.face_blendshapes[0]
        blendshapes = {cat.category_name: cat.score for cat in categories}
        
        # Optional: extract 3D landmarks if needed for HUD rendering later
        landmarks_array = None
        if result.face_landmarks:
            raw_lms = result.face_landmarks[0]
            landmarks_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in raw_lms], dtype=np.float32
            )
            
        return FaceResult(blendshapes=blendshapes, landmarks=landmarks_array)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "_landmarker"):
            self._landmarker.close()

    def __enter__(self) -> FaceTracker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
