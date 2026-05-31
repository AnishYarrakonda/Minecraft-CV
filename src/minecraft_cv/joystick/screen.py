"""Screen-space joystick signals and deadzone processing."""

from __future__ import annotations

import numpy as np

THUMB_TIP = 4


def screen_mcp_centroid(landmarks: np.ndarray) -> np.ndarray:
    """Return the (x, y) centroid of the palm MCP joints (index, middle, ring, pinky).

    This acts as a stable tracking point that feels natural to move across the screen.
    Uses landmarks 5, 9, 13, 17.
    """
    mcps = landmarks[[5, 9, 13, 17], :2]
    return np.mean(mcps, axis=0)


def screen_thumb_tip(landmarks: np.ndarray) -> np.ndarray:
    """Return the right-hand cursor/look signal from the thumb tip."""
    return np.asarray(landmarks[THUMB_TIP, :2], dtype=np.float64)


class ScreenJoystick:
    """Absolute screen-space joystick with zero calibration.

    The first time this joystick receives a signal (if `fixed_neutral` is None),
    it permanently anchors its `_neutral` center to that exact `(x, y)` position.
    If `fixed_neutral` is provided, the anchor is permanently set to those coordinates.
    Moving the hand away from the origin generates continuous output.
    """

    def __init__(
        self,
        deadzone: float,
        sensitivity: float,
        smoothing: float,
        fixed_neutral: tuple[float, float] | None = None,
    ) -> None:
        """Create a screen-space joystick with optional fixed neutral coordinates."""
        self.deadzone = deadzone
        self.sensitivity_val = sensitivity
        self.smoothing = smoothing
        self._is_fixed = fixed_neutral is not None
        self._neutral: np.ndarray | None = (
            np.array(fixed_neutral, dtype=np.float64) if fixed_neutral else None
        )
        self._filtered: np.ndarray | None = None

    @property
    def neutral(self) -> np.ndarray | None:
        """The anchored center point of the joystick."""
        return self._neutral

    @property
    def sensitivity(self) -> np.ndarray:
        """The gain multiplier as a 2D array (for compatibility with Pipeline)."""
        return np.array([self.sensitivity_val, self.sensitivity_val], dtype=np.float64)

    def reset_neutral(self) -> None:
        """Forget the origin (if dynamic). The next `update()` will lock a new origin."""
        if not self._is_fixed:
            self._neutral = None
        self._filtered = None

    def recenter_at(self, signal: np.ndarray) -> None:
        """Dynamically override the anchor to a new position, even if it was fixed."""
        self._neutral = signal.copy()
        self._filtered = signal.copy()

    def zero(self) -> np.ndarray:
        """Return a zeroed output vector."""
        return np.zeros(2, dtype=np.float64)

    def update(self, signal: np.ndarray) -> np.ndarray:
        """Process a raw `(x, y)` position and return normalized `(dx, dy)` joystick output."""
        if self._neutral is None:
            # Lock the origin on the first frame!
            self._neutral = signal.copy()

        if self._filtered is None:
            self._filtered = signal.copy()
        else:
            self._filtered = (
                self.smoothing * self._filtered + (1.0 - self.smoothing) * signal
            )

        # Distance from origin
        delta = self._filtered - self._neutral
        dist = np.linalg.norm(delta)

        if dist <= self.deadzone:
            return self.zero()

        # Magnitude-scaling deadzone (smooth transition out of the deadzone)
        scale = (dist - self.deadzone) / dist
        out = delta * scale * self.sensitivity_val

        # Clip max output magnitude to 1.0 (circular bounds)
        out_mag = np.linalg.norm(out)
        if out_mag > 1.0:
            out = out / out_mag

        return out
