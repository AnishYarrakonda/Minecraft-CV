"""Shared pytest fixtures: synthetic landmarks, settings, and a null emitter.

All gesture/joystick/pipeline tests are pure and deterministic — no camera, no MediaPipe,
no OS input. Synthetic ``(21, 3)`` landmark arrays let us drive the Schmitt triggers with
exact, controllable normalized distances.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pytest

from minecraft_cv.config import Settings

if TYPE_CHECKING:
    from minecraft_cv.input.emitter import NullEmitter
    from minecraft_cv.tracking.tracker import HandResult

# Landmark indices mirrored from minecraft_cv.gestures.pinch (kept local to the test fixture
# so a regression in the source constants is caught rather than masked).
_WRIST = 0
_THUMB_TIP = 4
_MIDDLE_MCP = 9
_TIP_INDEX = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
_DIRS = {
    "index": np.array([1.0, 0.0, 0.0]),
    "middle": np.array([0.0, 1.0, 0.0]),
    "ring": np.array([0.0, 0.0, 1.0]),
    "pinky": np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
}


def _build_landmarks(
    distances: Mapping[str, float],
    scale: float = 0.2,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Construct landmarks whose normalized thumb-pinch distances match ``distances``.

    Hand scale (wrist->middle-MCP span) equals ``scale``, so each fingertip placed at
    ``d * scale`` from the thumb yields a normalized distance of exactly ``d``. ``offset``
    translates the whole hand (moving the wrist/MCP anchor) without changing any normalized
    distance — used to drive the spatial joystick.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[_WRIST] = (0.0, 0.0, 0.0)
    lm[_MIDDLE_MCP] = (0.0, scale, 0.0)  # hand span = scale
    thumb = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    lm[_THUMB_TIP] = thumb

    full: dict[str, float] = {f: 1.0 for f in _TIP_INDEX}  # unspecified fingers: far/released
    full.update(distances)
    for finger, d in full.items():
        direction = _DIRS[finger].astype(np.float32)
        lm[_TIP_INDEX[finger]] = thumb + direction * (d * scale)
    return lm + np.asarray(offset, dtype=np.float32)


@pytest.fixture
def make_landmarks() -> Callable[..., np.ndarray]:
    """Return a builder: ``make_landmarks({"index": 0.2}, scale=0.2) -> (21, 3) array``."""
    return _build_landmarks


@pytest.fixture
def make_hand_result() -> Callable[..., "HandResult"]:
    """Return a builder for tracker HandResults from landmarks + handedness."""
    from minecraft_cv.tracking.tracker import HandResult

    def _make(landmarks: np.ndarray, handedness: str) -> HandResult:
        return HandResult(landmarks=landmarks, handedness=handedness, score=1.0)

    return _make


@pytest.fixture
def null_emitter() -> "NullEmitter":
    """A fresh recording NullEmitter (no OS input)."""
    from minecraft_cv.input.emitter import NullEmitter

    return NullEmitter()


@pytest.fixture
def default_settings() -> Settings:
    """Default Settings (input disabled — NullEmitter), MVP gesture map."""
    return Settings()
