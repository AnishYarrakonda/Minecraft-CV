"""Anchored wrist-rotation joystick math for the dual-thumbstick control mode.

The live controller uses the hand as a small virtual thumbstick: the wrist stays anchored
while the palm rotates around it. MediaPipe exposes this as the vector from the wrist to the
palm's MCP plane. We store that vector at neutral and use subsequent ``x``/``z`` deviation
as the joystick signal:

* ``x`` deviation -> left/right movement or mouse yaw.
* ``z`` deviation -> forward/back movement or mouse pitch.

All values are normalized by hand scale, so moving the whole hand across the camera frame does
not affect the output; only the palm/wrist angle relative to neutral matters.
"""

from __future__ import annotations

import numpy as np

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

_PALM_MCP_INDICES = np.array([INDEX_MCP, MIDDLE_MCP, PINKY_MCP], dtype=np.intp)


def palm_vector(landmarks: np.ndarray) -> np.ndarray:
    """Return the normalized wrist->palm-center vector in MediaPipe landmark units.

    Args:
        landmarks: ``(21, 3)`` array of MediaPipe hand landmarks. ``x``/``y`` are normalized
            frame coordinates; ``z`` is MediaPipe relative depth where more negative is closer
            to the camera.

    Returns:
        A ``(3,)`` vector from wrist landmark ``0`` to the mean of MCP landmarks ``5``, ``9``,
        and ``17``, divided by ``norm(landmark[9] - landmark[0])``.
    """
    wrist = landmarks[WRIST]
    scale = float(np.linalg.norm(landmarks[MIDDLE_MCP] - wrist)) or 1e-6
    center = landmarks[_PALM_MCP_INDICES].mean(axis=0)
    return np.asarray((center - wrist) / scale, dtype=np.float64)


def palm_xz(landmarks: np.ndarray) -> np.ndarray:
    """Return the ``(x, z)`` components of :func:`palm_vector`.

    Args:
        landmarks: ``(21, 3)`` MediaPipe hand landmarks.

    Returns:
        A ``(2,)`` array where index ``0`` is horizontal palm deviation and index ``1`` is
        depth deviation. More negative ``z`` means the palm rotated closer to the camera.
    """
    vec = palm_vector(landmarks)
    return np.asarray([vec[0], vec[2]], dtype=np.float64)


class WristRotationJoystick:
    """Per-axis linear deadzone joystick over anchored wrist-rotation ``x``/``z`` signal."""

    def __init__(
        self,
        deadzone_radius: float,
        sensitivity: float,
        max_output: float = 1.0,
        smoothing: float = 0.0,
    ) -> None:
        """Construct the joystick.

        Args:
            deadzone_radius: Per-axis neutral threshold around the stored neutral vector.
            sensitivity: Linear gain applied to travel beyond the deadzone edge.
            max_output: Clamp for each axis's output magnitude.
            smoothing: EMA factor on the incoming ``x``/``z`` palm vector (``0`` disables it).
        """
        self.deadzone_radius = float(deadzone_radius)
        self.sensitivity = float(sensitivity)
        self.max_output = float(max_output)
        self.smoothing = float(smoothing)
        self._neutral: np.ndarray | None = None
        self._filtered: np.ndarray | None = None

    @property
    def neutral(self) -> np.ndarray | None:
        """Current neutral ``(x, z)`` palm vector, or ``None`` until seeded."""
        return self._neutral

    def recenter(self, new_neutral: np.ndarray) -> None:
        """Set the neutral ``(x, z)`` palm vector explicitly."""
        self._neutral = np.asarray(new_neutral, dtype=np.float64)[:2].copy()
        self._filtered = None

    def reset_neutral(self) -> None:
        """Forget neutral/filter state so the next update becomes the new rest pose."""
        self._neutral = None
        self._filtered = None

    def update(self, signal: np.ndarray) -> np.ndarray:
        """Map a palm ``(x, z)`` signal to a per-axis linear joystick output.

        Args:
            signal: Current normalized palm ``(x, z)`` vector.

        Returns:
            ``(2,)`` output. Each axis is zero inside the deadzone and otherwise scales
            linearly as ``sign(delta) * min((abs(delta) - deadzone) * sensitivity, max)``.
        """
        sig = np.asarray(signal, dtype=np.float64)[:2]
        smoothed = self._smooth(sig)
        if self._neutral is None:
            self._neutral = smoothed.copy()
            return self.zero()

        delta = smoothed - self._neutral
        magnitude = np.maximum(np.abs(delta) - self.deadzone_radius, 0.0) * self.sensitivity
        magnitude = np.minimum(magnitude, self.max_output)
        return np.asarray(np.sign(delta) * magnitude, dtype=np.float64)

    def _smooth(self, signal: np.ndarray) -> np.ndarray:
        """EMA-smooth a normalized palm-vector sample."""
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


__all__ = ["WristRotationJoystick", "palm_vector", "palm_xz"]
