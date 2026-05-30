"""Hand-tracker abstraction: one ABC, swappable backends.

MediaPipe is the only component that does hand tracking; any PyTorch/YOLO tracker lives
behind this same ABC as an alternative backend, swappable without touching the pipeline.
A tracker takes an **RGB** frame and returns 0-2 :class:`HandResult` objects, each with 21
3D landmarks normalized to ``[0, 1]`` in frame space (``z`` is relative depth).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class HandResult(NamedTuple):
    """One detected hand.

    Attributes:
        landmarks: ``(21, 3)`` float32 array of ``(x, y, z)``. ``x``/``y`` are normalized to
            ``[0, 1]`` relative to frame width/height; ``z`` is depth relative to the wrist
            in the same scale as ``x`` (negative = closer to camera).
        handedness: ``"Left"`` or ``"Right"`` as reported by the tracker.
        score: Detection/handedness confidence in ``[0, 1]``.
    """

    landmarks: np.ndarray
    handedness: str
    score: float = 1.0


class HandTracker(ABC):
    """Abstract hand tracker. Construct via :meth:`create`."""

    @abstractmethod
    def detect(self, rgb_frame: np.ndarray) -> list[HandResult]:
        """Detect hands in an RGB frame.

        Args:
            rgb_frame: ``(H, W, 3)`` ``uint8`` RGB image (already BGR->RGB converted).

        Returns:
            Zero to two :class:`HandResult` objects.
        """

    def landmarks(self, rgb_frame: np.ndarray) -> np.ndarray | None:
        """Convenience: the first detected hand's landmarks, or ``None`` if no hand."""
        results = self.detect(rgb_frame)
        return results[0].landmarks if results else None

    def close(self) -> None:
        """Release backend resources. Default no-op; override if needed."""

    @staticmethod
    def create(backend: str = "mediapipe", device: str = "auto") -> HandTracker:
        """Factory for swappable tracker backends.

        Args:
            backend: ``"mediapipe"`` (default) or ``"yolo"``.
            device: ``"auto"``/``"mps"``/``"cuda"``/``"cpu"``. MediaPipe ignores this
                (CPU-only); the YOLO backend honors it.

        Returns:
            A concrete :class:`HandTracker`.

        Raises:
            ValueError: For an unknown backend name.
            NotImplementedError: For the not-yet-implemented YOLO backend.
        """
        if backend == "mediapipe":
            from minecraft_cv.tracking.mediapipe_backend import MediaPipeHandTracker

            return MediaPipeHandTracker()
        if backend == "yolo":
            raise NotImplementedError(
                "The YOLO/PyTorch backend is a V2 alternative; it is not implemented yet."
            )
        raise ValueError(f"Unknown tracking backend: {backend!r}")
