"""Frame sources: live AVFoundation camera and offline clip playback.

OpenCV's role here is strictly frame ingestion (see ``.claude/rules/tech-stack.md``). Frames
come out as BGR ``uint8`` NumPy arrays; conversion to RGB happens once, downstream, in the
pipeline. ``cv2`` is imported lazily inside the source classes so the package imports cleanly
without OpenCV present (tests inject a fake :class:`FrameSource`).

macOS footguns handled here:
  * Force the AVFoundation backend explicitly.
  * ``CAP_PROP_BUFFERSIZE = 1`` so a naive read returns the newest frame, not a stale one.
  * Camera-permission black frames: a missing Camera grant yields all-black frames with no
    exception. We grab one frame at startup and raise a clear error if it is uniform.
  * Continuity Camera: a device-enumeration helper lets the user pin the right index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import numpy as np

# Below this per-frame pixel standard deviation, a frame is treated as "blank" (a black or
# uniform frame from a missing Camera permission), not real imagery.
_BLANK_STD_THRESHOLD = 1.0
_LIVE_READ_RETRY_ATTEMPTS = 30
_LIVE_READ_RETRY_SLEEP_S = 0.01


class FrameSource(ABC):
    """A source of BGR ``uint8`` frames."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the latest BGR ``uint8`` frame, or ``None`` if exhausted/unavailable."""

    @abstractmethod
    def release(self) -> None:
        """Release any underlying device/file handle. Idempotent."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames per second for this source."""


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "OpenCV (opencv-python) is required for camera/clip capture. Install it, or use "
            "an injected FrameSource (tests do this)."
        ) from exc
    return cv2


def enumerate_devices(max_index: int = 5) -> list[int]:
    """Return the camera indices that successfully open (to disambiguate Continuity Camera).

    Args:
        max_index: Highest device index to probe (inclusive lower bound is 0).

    Returns:
        Sorted list of indices that opened. Pin the desired one in ``config.yaml``.
    """
    cv2 = _import_cv2()
    available: list[int] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        try:
            if cap.isOpened():
                available.append(idx)
        finally:
            cap.release()
    return available


class AVFoundationSource(FrameSource):
    """Live macOS camera via ``cv2.VideoCapture`` with the AVFoundation backend."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        """Open the camera and verify it yields real (non-blank) frames.

        Args:
            index: Camera device index. macOS may assign a Continuity Camera to 0; use
                :func:`enumerate_devices` to find the right one.
            width: Requested capture width in pixels.
            height: Requested capture height in pixels.
            fps: Requested capture frames per second.

        Raises:
            RuntimeError: If the device cannot be opened.
            PermissionError: If the first frame is blank (likely a missing Camera grant).
        """
        self._cv2 = _import_cv2()
        self._fps = float(fps)
        self._index = index
        cap = self._cv2.VideoCapture(index, self._cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index} (AVFoundation). This could be an "
                f"incorrect device index (check enumerate_devices()) or missing Camera "
                f"permissions (check System Settings -> Privacy & Security -> Camera)."
            )
        # Newest-frame-wins + explicit format (defaults are often low-FPS/high-res).
        cap.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(self._cv2.CAP_PROP_FPS, fps)
        self._cap = cap
        self._assert_not_blank()

    def _assert_not_blank(self) -> None:
        import sys
        import time

        # Some cameras need a few frames to warm up or auto-expose and may return blank/black
        # frames initially. Try reading up to 100 frames before concluding permission is missing.
        # This gives the user up to 5 seconds to approve the macOS camera permission prompt.
        max_attempts = 100
        for i in range(max_attempts):
            ok, frame = self._cap.read()
            if ok and frame is not None and float(np.asarray(frame).std()) >= _BLANK_STD_THRESHOLD:
                if i >= 10:
                    print(" [done]", file=sys.stderr)
                return
            if i == 10:
                print(
                    f"[mcv-run] Waiting for camera {self._index} to warm up or for "
                    "macOS permission prompt...",
                    end="",
                    flush=True,
                    file=sys.stderr,
                )
            time.sleep(0.05)  # give the sensor/driver a brief moment to warm up

        if max_attempts > 10:
            print(" [failed]", file=sys.stderr)
        self.release()
        raise PermissionError(
            f"Camera index {self._index} opened but returned a blank/black frame. This is "
            "almost always a missing Camera permission. Grant it in System Settings -> "
            "Privacy & Security -> Camera to your terminal app, then restart it."
        )

    def read(self) -> np.ndarray | None:
        """Read one BGR frame from the live camera, or ``None`` on sustained failure."""
        import time

        for _ in range(_LIVE_READ_RETRY_ATTEMPTS):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return cast(np.ndarray, frame)
            time.sleep(_LIVE_READ_RETRY_SLEEP_S)
        return None

    def release(self) -> None:
        """Release the underlying ``cv2.VideoCapture`` object."""
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()
            self._cap = None  # type: ignore[assignment]

    @property
    def fps(self) -> float:
        """Configured camera frame rate in frames per second."""
        return self._fps


class ClipSource(FrameSource):
    """Offline playback of a recorded ``.mp4`` / ``.mov`` (or a still image).

    Deterministic and CI-friendly: the canonical way to run the pipeline reproducibly.
    """

    def __init__(self, path: str | Path) -> None:
        """Open a clip or image file.

        Args:
            path: Path to a video (``.mp4``/``.mov``/...) or image (``.png``/``.jpg``/...).

        Raises:
            FileNotFoundError: If the file cannot be opened.
        """
        self._cv2 = _import_cv2()
        self._path = Path(path)
        self._is_image = self._path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        self._image: np.ndarray | None = None
        self._cap: Any = None
        if self._is_image:
            img = self._cv2.imread(str(self._path))
            if img is None:
                raise FileNotFoundError(f"Could not read image: {self._path}")
            self._image = img
            self._fps = 1.0
        else:
            cap = self._cv2.VideoCapture(str(self._path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open clip: {self._path}")
            self._cap = cap
            self._fps = float(cap.get(self._cv2.CAP_PROP_FPS)) or 30.0

    def read(self) -> np.ndarray | None:
        """Read the next BGR frame from the clip/image, or ``None`` at EOF."""
        if self._is_image:
            img, self._image = self._image, None  # yield once, then exhausted
            return img
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        """Release the underlying clip reader."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        """Clip frame rate in frames per second."""
        return self._fps
