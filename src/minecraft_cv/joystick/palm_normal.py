"""Palm-normal joystick math for calibrated dual-thumbstick controls.

The palm-normal mode treats each hand as a small plane. A live calibration stores the
resting palm normal for each hand, then gameplay uses ``x``/``y`` deviation from that
normal as the joystick signal. This avoids depending on whole-hand translation or the
noisier MediaPipe depth axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


def palm_normal(landmarks: np.ndarray) -> np.ndarray:
    """Return a unit normal for the palm plane.

    The plane is spanned by the index-to-pinky MCP line and the wrist-to-middle-MCP line.
    The resulting normal is forced into a stable ``z >= 0`` hemisphere so left/right hands
    and small rotations do not randomly flip signs.
    """
    wrist = landmarks[WRIST]
    across = landmarks[PINKY_MCP] - landmarks[INDEX_MCP]
    up = landmarks[MIDDLE_MCP] - wrist
    normal = np.cross(across, up)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    normal = np.asarray(normal / norm, dtype=np.float64)
    if normal[2] < 0.0:
        normal = -normal
    return normal


def palm_normal_xy(landmarks: np.ndarray) -> np.ndarray:
    """Return the screen-space ``(x, y)`` components of :func:`palm_normal`."""
    normal = palm_normal(landmarks)
    return np.asarray([normal[0], normal[1]], dtype=np.float64)


class PalmNormalJoystick:
    """Per-axis linear deadzone joystick over calibrated palm-normal ``x``/``y``."""

    def __init__(
        self,
        neutral: Sequence[float],
        deadzone: float,
        sensitivity: float | Sequence[float],
        max_output: float = 1.0,
        smoothing: float = 0.0,
    ) -> None:
        """Construct the joystick from calibrated values."""
        self._configured_neutral = np.asarray(neutral, dtype=np.float64)[:2].copy()
        self._neutral = self._configured_neutral.copy()
        self.deadzone = float(deadzone)
        sens = np.asarray(sensitivity, dtype=np.float64)
        if sens.ndim == 0:
            sens = np.asarray([float(sens), float(sens)], dtype=np.float64)
        self.sensitivity = sens[:2].copy()
        self.max_output = float(max_output)
        self.smoothing = float(smoothing)
        self._filtered: np.ndarray | None = None

    @property
    def neutral(self) -> np.ndarray:
        """Current calibrated neutral ``(x, y)`` palm-normal vector."""
        return self._neutral

    def recenter(self, new_neutral: np.ndarray) -> None:
        """Set a new neutral explicitly and clear filter history."""
        self._neutral = np.asarray(new_neutral, dtype=np.float64)[:2].copy()
        self._filtered = None

    def reset_neutral(self) -> None:
        """Restore the configured neutral and clear filter history."""
        self._neutral = self._configured_neutral.copy()
        self._filtered = None

    def update(self, signal: np.ndarray) -> np.ndarray:
        """Map a palm-normal ``(x, y)`` sample to per-axis linear output."""
        sig = np.asarray(signal, dtype=np.float64)[:2]
        smoothed = self._smooth(sig)
        delta = smoothed - self._neutral
        magnitude = np.maximum(np.abs(delta) - self.deadzone, 0.0) * self.sensitivity
        magnitude = np.minimum(magnitude, self.max_output)
        return np.asarray(np.sign(delta) * magnitude, dtype=np.float64)

    def _smooth(self, signal: np.ndarray) -> np.ndarray:
        """EMA-smooth a palm-normal sample."""
        if self._filtered is None:
            filtered = signal.copy()
        else:
            alpha = 1.0 - self.smoothing
            filtered = alpha * signal + self.smoothing * self._filtered
        self._filtered = np.asarray(filtered, dtype=np.float64)
        return self._filtered

    def zero(self) -> np.ndarray:
        """Return a zero output vector."""
        return np.zeros(2, dtype=np.float64)


__all__ = ["PalmNormalJoystick", "palm_normal", "palm_normal_xy"]
