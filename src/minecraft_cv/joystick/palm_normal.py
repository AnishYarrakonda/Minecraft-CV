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
    """Per-axis linear deadzone joystick over calibrated palm-normal ``x``/``y``.

    Supports optional asymmetric per-direction sensitivity via ``sensitivity_neg``.
    When provided, negative delta on an axis uses the corresponding negative gain
    instead of the positive one — this allows the geometrically-smaller "back" reach
    to be amplified independently so it matches the effective travel of "forward".
    """

    def __init__(
        self,
        neutral: Sequence[float] | None,
        deadzone: float,
        sensitivity: float | Sequence[float],
        max_output: float = 1.0,
        smoothing: float = 0.0,
        sensitivity_neg: Sequence[float] | None = None,
    ) -> None:
        """Construct the joystick from calibrated values.

        ``neutral=None`` is reserved for dry-run preview. The first visible hand sample
        becomes a temporary neutral so the debug overlay can launch before calibration.

        Args:
            neutral: Calibrated ``(x, y)`` neutral palm-normal vector, or ``None`` for
                uncalibrated dry-run preview.
            deadzone: Per-axis linear deadzone half-width.
            sensitivity: Positive-direction per-axis gain, scalar or ``(x, y)`` pair.
            max_output: Output saturation magnitude (same units as the joystick output).
            smoothing: EMA smoothing factor in ``[0, 1)``. ``0`` disables smoothing.
            sensitivity_neg: Optional negative-direction per-axis gain ``(x, y)``.
                When ``None`` (default), the positive ``sensitivity`` is used for both
                directions (symmetric — identical to the original behavior).
        """
        self._configured_neutral = (
            None if neutral is None else np.asarray(neutral, dtype=np.float64)[:2].copy()
        )
        self._neutral = (
            None if self._configured_neutral is None else self._configured_neutral.copy()
        )
        self.deadzone = deadzone
        sens = np.asarray(sensitivity, dtype=np.float64)
        if sens.ndim == 0:
            sens = np.asarray([sens.item(), sens.item()], dtype=np.float64)
        self.sensitivity = sens[:2].copy()
        self.max_output = max_output
        self.smoothing = smoothing
        self._filtered: np.ndarray | None = None

        # Asymmetric negative gain (None = symmetric, falls back to self.sensitivity)
        if sensitivity_neg is None:
            self.sensitivity_neg: np.ndarray | None = None
        else:
            sn = np.asarray(sensitivity_neg, dtype=np.float64)
            if sn.ndim == 0:
                sn = np.asarray([sn.item(), sn.item()], dtype=np.float64)
            self.sensitivity_neg = sn[:2].copy()

    @property
    def neutral(self) -> np.ndarray:
        """Current calibrated neutral ``(x, y)`` palm-normal vector."""
        if self._neutral is None:
            return np.zeros(2, dtype=np.float64)
        return self._neutral

    def recenter(self, new_neutral: np.ndarray) -> None:
        """Set a new neutral explicitly and clear filter history."""
        self._neutral = np.asarray(new_neutral, dtype=np.float64)[:2].copy()
        self._filtered = None

    def reset_neutral(self) -> None:
        """Restore the configured neutral and clear filter history."""
        self._neutral = (
            None if self._configured_neutral is None else self._configured_neutral.copy()
        )
        self._filtered = None

    def update(self, signal: np.ndarray) -> np.ndarray:
        """Map a palm-normal ``(x, y)`` sample to per-axis linear output.

        If ``sensitivity_neg`` is set, the gain for each axis is chosen by the sign of
        the delta: positive delta uses ``self.sensitivity``; negative delta uses
        ``self.sensitivity_neg``.  This allows separate amplification of the
        geometrically-smaller "back" / "left" reach.
        """
        sig = np.asarray(signal, dtype=np.float64)[:2]
        smoothed = self._smooth(sig)
        if self._neutral is None:
            self._neutral = smoothed.copy()
            return self.zero()
        delta = smoothed - self._neutral
        # Choose per-axis gain by sign of delta (asymmetric mode) or use uniform gain.
        if self.sensitivity_neg is not None:
            gain = np.where(delta >= 0, self.sensitivity, self.sensitivity_neg)
        else:
            gain = self.sensitivity
        magnitude = np.maximum(np.abs(delta) - self.deadzone, 0.0) * gain
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
