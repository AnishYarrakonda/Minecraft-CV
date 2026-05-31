"""Angular cardinal-zone key selection and magnitude-based acceleration curve.

Two pure, stateless helpers used by the movement pipeline to convert a continuous
2D joystick output into discrete WASD key presses and to apply an exponential ease-in
curve that keeps small movements precise while large movements saturate quickly.

All vectors are in **joystick output space** (abstract units, not normalized frame
coordinates) and are processed on the CPU with no per-frame heap allocation beyond the
two-element return arrays.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

# ---------------------------------------------------------------------------
# Internal constants (not exported; callers use the parameter API)
# ---------------------------------------------------------------------------

# Cardinal angle centres, in degrees.  Matches the axis convention documented below.
_CARDINALS: dict[str, float] = {
    "right": 0.0,
    "forward": 90.0,
    "left": 180.0,
    "back": -90.0,
}


def _angular_dist(theta: float, centre: float) -> float:
    """Shortest signed-magnitude arc between two angles (degrees).

    Handles the ±180° wraparound so e.g. dist(179°, −179°) == 2°, not 358°.

    Args:
        theta: Query angle in degrees.
        centre: Reference angle in degrees.

    Returns:
        Unsigned angular distance in ``[0, 180]`` degrees.
    """
    return abs(((theta - centre + 180.0) % 360.0) - 180.0)


def cardinal_keys(
    output: np.ndarray,
    half_width_deg: float,
    bindings: Mapping[str, str],
) -> set[str]:
    """Convert a 2D joystick output vector into the set of WASD keys to press.

    Axis convention (right-hand, image-plane):

    - ``+output[0]`` → ``bindings["right"]``
    - ``−output[0]`` → ``bindings["left"]``
    - ``+output[1]`` → ``bindings["forward"]``
    - ``−output[1]`` → ``bindings["back"]``

    Each cardinal fires when the angle to its centre is within
    ``(90.0 − half_width_deg)`` degrees.  Two cardinals fire simultaneously in the
    diagonal bands between pure zones, producing combined movement (e.g. W+A for
    forward-left).

    Threshold semantics:

    - ``half_width_deg=35`` → 55° firing radius; pure zones ±35°; 20° diagonal bands.
    - ``half_width_deg=45`` → 45° firing radius; no overlaps → no diagonals ever.
    - ``half_width_deg=0``  → 90° firing radius → any nonzero component fires.

    Args:
        output: ``(2,)`` joystick vector ``(x, y)`` in abstract joystick units.
            Longer arrays are accepted; only the first two components are used.
        half_width_deg: Half-angle of the *pure zone* in degrees.  A larger value
            widens the pure zone and narrows the diagonal bands. Range: ``[0, 90)``.
        bindings: Map of cardinal names to key strings.  Must contain keys
            ``"right"``, ``"left"``, ``"forward"``, ``"back"``.

    Returns:
        The set of key strings (from ``bindings``) that should be held this frame.
        Empty set when the vector is zero or near-zero.
    """
    vec = np.asarray(output, dtype=np.float64)
    mag = float(np.linalg.norm(vec[:2]))
    if mag <= 0.0:
        return set()

    theta = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
    firing_radius = 90.0 - half_width_deg

    pressed: set[str] = set()
    for direction, centre in _CARDINALS.items():
        if _angular_dist(theta, centre) <= firing_radius:
            pressed.add(bindings[direction])
    return pressed


def accel_curve(
    vec: np.ndarray,
    exponent: float,
    max_output: float,
) -> np.ndarray:
    """Apply a magnitude-based, direction-preserving exponential ease-in curve.

    Maps a 2D input vector so that small physical displacements produce precise (small)
    outputs and large displacements saturate at ``max_output``.  Direction is preserved
    exactly; only the magnitude is reshaped.

    The curve is:

    .. code-block:: text

        u      = clamp(|v| / max_output, 0, 1)   # normalize magnitude to [0, 1]
        shaped = (u ** exponent) * max_output     # ease-in stretch
        output = (v / |v|) * shaped               # restore direction

    Properties:

    - Continuous at the origin (output → 0 as |v| → 0).
    - Saturates at ``max_output`` once ``|v| >= max_output``.
    - ``exponent > 1`` → gentle near zero, fast near saturation (mitigates Gorilla Arm).
    - ``exponent == 1`` → linear pass-through (still clamped at ``max_output``).

    Args:
        vec: ``(2,)`` or longer input vector in abstract joystick units. Only the first
            two components are used.
        exponent: Ease-in exponent. Must be ``> 0``; values ``> 1`` apply the
            Gorilla-Arm-mitigating acceleration curve.
        max_output: Saturation magnitude in the same units as ``vec``. Input magnitudes
            at or above this value map to exactly ``max_output`` in the output.

    Returns:
        A ``(2,)`` float64 array with the shaped magnitude and the original direction.
        Returns ``np.zeros(2)`` for a zero-magnitude input.
    """
    vec2 = np.asarray(vec, dtype=np.float64)[:2]
    mag = float(np.linalg.norm(vec2))
    if mag <= 0.0:
        return np.zeros(2, dtype=np.float64)
    u = min(mag / max_output, 1.0)
    shaped = (u**exponent) * max_output
    return np.asarray((vec2 / mag) * shaped, dtype=np.float64)


__all__ = ["accel_curve", "cardinal_keys"]
