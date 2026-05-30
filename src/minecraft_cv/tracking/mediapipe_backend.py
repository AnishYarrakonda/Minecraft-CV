"""MediaPipe Hands backend.

Wraps ``mediapipe.solutions.hands.Hands``. MediaPipe runs on CPU only (no MPS/CUDA path);
do not attempt to move it to a GPU device. Imported lazily so the package imports without
MediaPipe present (tests use synthetic landmarks, never this backend).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from minecraft_cv.tracking.tracker import HandResult, HandTracker

_NUM_LANDMARKS = 21


class MediaPipeHandTracker(HandTracker):
    """Hand tracker backed by MediaPipe Hands (CPU-only)."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the MediaPipe Hands graph.

        Args:
            max_num_hands: Maximum hands to detect (1-2 for this controller).
            min_detection_confidence: Minimum confidence to start a track.
            min_tracking_confidence: Minimum confidence to keep tracking across frames.

        Raises:
            RuntimeError: If MediaPipe is not importable.
        """
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "mediapipe is required for the default tracking backend. Install it with "
                "'pip install mediapipe' (CPU-only), or select a different backend."
            ) from exc
        self._mp: Any = mp
        self._hands: Any = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, rgb_frame: np.ndarray) -> list[HandResult]:
        """Run MediaPipe on an RGB frame and return per-hand landmark results.

        Args:
            rgb_frame: ``(H, W, 3)`` ``uint8`` RGB frame. MediaPipe requires RGB; the
                BGR->RGB conversion is the pipeline's responsibility (done once per frame).

        Returns:
            Zero to ``max_num_hands`` :class:`HandResult` objects.
        """
        results = self._hands.process(rgb_frame)
        hands = getattr(results, "multi_hand_landmarks", None)
        if not hands:
            return []
        handedness = getattr(results, "multi_handedness", None)
        out: list[HandResult] = []
        for i, hand_landmarks in enumerate(hands):
            # Protobuf -> ndarray. The per-landmark read is unavoidable (21 fields), but all
            # downstream math (distances, joystick) stays vectorized on this array.
            coords = np.empty((_NUM_LANDMARKS, 3), dtype=np.float32)
            for j, lm in enumerate(hand_landmarks.landmark):
                coords[j] = (lm.x, lm.y, lm.z)
            label, score = "Unknown", 1.0
            if handedness is not None and i < len(handedness):
                classification = handedness[i].classification[0]
                label = classification.label
                score = float(classification.score)
            out.append(HandResult(landmarks=coords, handedness=label, score=score))
        return out

    def close(self) -> None:
        """Close the MediaPipe graph and free its resources."""
        hands = getattr(self, "_hands", None)
        if hands is not None:
            hands.close()
            self._hands = None
