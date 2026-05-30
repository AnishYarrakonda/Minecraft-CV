"""Spatial-joystick math: spherical deadzone + exponential acceleration.

Each hand's anchor landmark (wrist=0 or middle-MCP=9, never the bbox center — pinching
shifts the bbox and corrupts the vector) is compared against a stored neutral position to
produce a continuous 2D output:

    NEUTRAL: pos inside deadzone sphere  -> (0, 0)
    ACTIVE:  pos outside deadzone sphere -> direction * f(distance - deadzone)

The deadzone is a **sphere** (Euclidean norm), not a box, so diagonal directions are not
biased. An exponential acceleration curve maps large physical movements to fast in-game
motion without forcing the user to travel far — this mitigates Gorilla Arm syndrome.

All positions are in **normalized frame coordinates** ([0, 1] in x/y), matching MediaPipe
landmark space. Output units are abstract joystick units; the input layer scales them to
WASD taps (left hand) or relative mouse-look pixels (right hand).
"""

from __future__ import annotations

import numpy as np

# Anchor name -> landmark index (see .claude/rules/gestures.md / tech-stack.md).
ANCHOR_INDEX: dict[str, int] = {"wrist": 0, "middle_mcp": 9}


def anchor_xy(landmarks: np.ndarray, anchor: str) -> np.ndarray:
    """Extract the 2D anchor position (x, y) from a landmark array.

    Args:
        landmarks: ``(21, 3)`` normalized landmark array.
        anchor: ``"wrist"`` or ``"middle_mcp"``.

    Returns:
        A ``(2,)`` float array of the anchor's normalized ``(x, y)`` position.

    Raises:
        KeyError: If ``anchor`` is not a known anchor name.
    """
    return np.asarray(landmarks[ANCHOR_INDEX[anchor]][:2], dtype=np.float64)


class DeadzoneJoystick:
    """Continuous joystick with a spherical deadzone and exponential acceleration.

    The neutral position auto-calibrates on the first sample after construction or
    :meth:`reset_neutral` — this is the recenter/drift macro: when both hands leave and
    re-enter the frame, the new entry coordinates become the fresh ``(0, 0)`` neutral with
    no button press required.
    """

    def __init__(
        self,
        deadzone_radius: float,
        sensitivity: float,
        accel_exponent: float,
        max_output: float = 1.0,
    ) -> None:
        """Construct a joystick.

        Args:
            deadzone_radius: Sphere radius (normalized units) inside which output is zero.
            sensitivity: Linear gain applied to displacement beyond the deadzone edge.
            accel_exponent: Exponent of the acceleration curve (>1 = superlinear). Output is
                continuous (=0) at the deadzone boundary regardless of exponent.
            max_output: Per-call clamp on the output magnitude.
        """
        self.deadzone_radius = float(deadzone_radius)
        self.sensitivity = float(sensitivity)
        self.accel_exponent = float(accel_exponent)
        self.max_output = float(max_output)
        self._neutral: np.ndarray | None = None

    @property
    def neutral(self) -> np.ndarray | None:
        """The current neutral position (``None`` until the first sample)."""
        return self._neutral

    def recenter(self, new_neutral: np.ndarray) -> None:
        """Set the neutral position explicitly (e.g. on inventory-mode exit)."""
        self._neutral = np.asarray(new_neutral, dtype=np.float64)[:2].copy()

    def reset_neutral(self) -> None:
        """Forget the neutral so the next :meth:`update` recalibrates (recenter macro)."""
        self._neutral = None

    def update(self, position: np.ndarray) -> np.ndarray:
        """Map an anchor position to a joystick output vector.

        Args:
            position: ``(2,)`` normalized anchor position ``(x, y)`` for this frame. (A
                longer array is accepted; only the first two components are used.)

        Returns:
            A ``(2,)`` float output vector. ``(0, 0)`` inside the deadzone sphere; otherwise
            ``direction * clamp(((distance - deadzone) * sensitivity) ** accel_exponent)``.
            Continuous at the sphere boundary (no step discontinuity).
        """
        pos = np.asarray(position, dtype=np.float64)[:2]
        if self._neutral is None:
            # First sample after (re)centering becomes the neutral; output is zero.
            self._neutral = pos.copy()
            return np.zeros(2, dtype=np.float64)

        delta = pos - self._neutral
        distance = float(np.linalg.norm(delta))
        if distance <= self.deadzone_radius:
            return np.zeros(2, dtype=np.float64)

        direction = delta / distance
        excess = (distance - self.deadzone_radius) * self.sensitivity
        magnitude = excess**self.accel_exponent
        magnitude = min(magnitude, self.max_output)
        return direction * magnitude

    def zero(self) -> np.ndarray:
        """Return a zero output vector (used when the hand is absent)."""
        return np.zeros(2, dtype=np.float64)
