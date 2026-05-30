"""Finger extension detection from MediaPipe hand landmarks.

Detects which fingers are extended (straightened) vs curled (closed into fist)
from the 21-landmark hand skeleton. Used by the left-hand gesture system where
the default pose is a closed fist and gestures are triggered by extending
specific finger combinations.

Extension is measured as a continuous ratio (tip-distance / pip-distance from
wrist), suitable for feeding into Schmitt triggers downstream. The thumb uses
a separate lateral-distance metric since it extends sideways rather than
straightening.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20


@dataclass(frozen=True)
class FingerState:
    """Per-frame finger extension state.

    Each field is a continuous ratio: higher = more extended. These raw values
    are fed into Schmitt triggers for hysteresis-based engage/release detection.

    Attributes:
        thumb_ext: Normalized lateral distance of thumb tip from palm (thumb tip
            to index MCP, divided by hand scale). Higher = thumb sticking out more.
        index_ext: Extension ratio for the index finger (wrist-to-tip / wrist-to-pip).
            Values > ~1.2 indicate extension; < ~1.0 indicate curled.
        middle_ext: Extension ratio for the middle finger.
        ring_ext: Extension ratio for the ring finger.
        pinky_ext: Extension ratio for the pinky finger.
    """

    thumb_ext: float
    index_ext: float
    middle_ext: float
    ring_ext: float
    pinky_ext: float


def _dist(landmarks: np.ndarray, a: int, b: int) -> float:
    """Euclidean distance between two landmarks."""
    return float(np.linalg.norm(landmarks[a] - landmarks[b])) or 1e-9


def finger_extensions(landmarks: np.ndarray) -> FingerState:
    """Compute per-finger extension ratios from hand landmarks.

    Args:
        landmarks: ``(21, 3)`` float array of MediaPipe hand keypoints.
            ``x``/``y`` normalized to ``[0, 1]``; ``z`` is relative depth.

    Returns:
        A :class:`FingerState` with continuous extension ratios for each finger.
    """
    hand_scale = _dist(landmarks, WRIST, MIDDLE_MCP) or 1e-6

    # Thumb: lateral distance from thumb tip to index MCP, normalized by hand scale.
    thumb = min(2.0, max(0.0, _dist(landmarks, THUMB_TIP, INDEX_MCP) / hand_scale))

    # For each finger: ratio of (wrist→tip distance) / (wrist→pip distance).
    # Extended finger: tip is farther from wrist than PIP → ratio > 1.
    # Curled finger: tip is closer to wrist than PIP → ratio < 1.
    index = _dist(landmarks, WRIST, INDEX_TIP) / (_dist(landmarks, WRIST, INDEX_PIP) or 1e-9)
    middle = _dist(landmarks, WRIST, MIDDLE_TIP) / (_dist(landmarks, WRIST, MIDDLE_PIP) or 1e-9)
    ring = _dist(landmarks, WRIST, RING_TIP) / (_dist(landmarks, WRIST, RING_PIP) or 1e-9)
    pinky = _dist(landmarks, WRIST, PINKY_TIP) / (_dist(landmarks, WRIST, PINKY_PIP) or 1e-9)

    return FingerState(
        thumb_ext=thumb,
        index_ext=index,
        middle_ext=middle,
        ring_ext=ring,
        pinky_ext=pinky,
    )


__all__ = ["FingerState", "finger_extensions"]
