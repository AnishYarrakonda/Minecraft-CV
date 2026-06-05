"""Wrist-tilt based WASD joystick for the left hand.

Maps the angle of the wrist-to-middle-MCP vector to discrete WASD directions.
The hand can be in any position — only the tilt direction matters. This is
orthogonal to finger pinch gestures, allowing simultaneous WASD + jump + sneak.
"""
from __future__ import annotations

import numpy as np

WRIST = 0
MIDDLE_MCP = 9


def wrist_tilt_vector(landmarks: np.ndarray) -> np.ndarray:
    """Extract the 2D wrist→middle-MCP direction vector.
    
    Args:
        landmarks: (21, 3) float array of hand keypoints, x/y in [0,1].
    
    Returns:
        2D direction vector (dx, dy) from wrist to middle MCP, NOT normalized.
    """
    wrist = landmarks[WRIST, :2]
    mcp = landmarks[MIDDLE_MCP, :2]
    return np.asarray(mcp - wrist, dtype=np.float64)


class WristTiltJoystick:
    """Wrist-tilt based joystick.
    
    Measures the 2D vector from wrist to MCP relative to a calibrated neutral
    resting vector. This gives 2 degrees of freedom (tilt left/right, and
    foreshorten/extend up/down) while being immune to overall arm translation.
    """

    def __init__(
        self,
        deadzone_deg: float = 0.05,  # Using length units since it's a vector diff
        sensitivity: float = 5.0,
        smoothing: float = 0.3,
    ) -> None:
        self.deadzone_val = deadzone_deg
        self._sensitivity_val = sensitivity
        self.smoothing = smoothing
        self._neutral_vec: np.ndarray | None = None
        self._filtered_vec: np.ndarray | None = None
        self._is_fixed = False

    @property
    def deadzone(self) -> float:
        return self.deadzone_val

    @property
    def neutral(self) -> np.ndarray | None:
        return self._neutral_vec

    @property
    def sensitivity(self) -> np.ndarray:
        return np.array([self._sensitivity_val, self._sensitivity_val], dtype=np.float64)

    def reset_neutral(self) -> None:
        """Forget the neutral vector. Next update() seeds a new one."""
        if not self._is_fixed:
            self._neutral_vec = None
        self._filtered_vec = None

    def recenter_at(self, signal: np.ndarray) -> None:
        """Override the neutral to the current tilt direction."""
        self._neutral_vec = signal.copy()
        self._filtered_vec = signal.copy()

    def zero(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float64)

    def update(self, signal: np.ndarray) -> np.ndarray:
        """Process a raw wrist-tilt vector and return normalized WASD output.
        
        Args:
            signal: 2D wrist→MCP direction vector from wrist_tilt_vector().
        """
        if self._neutral_vec is None:
            self._neutral_vec = signal.copy()
            self._filtered_vec = signal.copy()
            return self.zero()

        if self._filtered_vec is None:
            self._filtered_vec = signal.copy()
        else:
            self._filtered_vec = (
                self.smoothing * self._filtered_vec + (1.0 - self.smoothing) * signal
            )

        # 2D deviation from neutral vector
        delta = self._filtered_vec - self._neutral_vec
        dist = np.linalg.norm(delta)

        if dist <= self.deadzone_val:
            return self.zero()

        scale = (dist - self.deadzone_val) / dist
        out = delta * scale * self._sensitivity_val

        mag = np.linalg.norm(out)
        if mag > 1.0:
            out = out / mag
        return out
