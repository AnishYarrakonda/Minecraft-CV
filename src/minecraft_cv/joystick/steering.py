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

def octant_keys(
    output: np.ndarray,
    bindings: Mapping[str, str],
) -> set[str]:
    """Convert a 2D joystick output vector into one of 8 strict radial slices.

    The 360-degree space is partitioned into 8 equal 45-degree pizza slices:
    W, WD, D, SD, S, SA, A, WA.

    Axis convention (image-plane):
    - ``+x`` → Right
    - ``+y`` → Down (Back)
    - ``-y`` → Up (Forward)

    Args:
        output: ``(2,)`` joystick vector ``(x, y)`` in abstract joystick units.
        bindings: Map of cardinal names to key strings. Must contain keys
            ``"right"``, ``"left"``, ``"forward"``, ``"back"``.

    Returns:
        The set of key strings to hold this frame. Empty set if vector is zero.
    """
    vec = np.asarray(output, dtype=np.float64)
    if np.linalg.norm(vec[:2]) <= 0.0:
        return set()

    theta = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
    octant = round(theta / 45.0) % 8

    # 0: Right (D)
    # 1: Back-Right (SD)
    # 2: Back (S)
    # 3: Back-Left (SA)
    # 4: Left (A)
    # 5: Forward-Left (WA)
    # 6: Forward (W)
    # 7: Forward-Right (WD)
    octant_map = [
        {"right"},
        {"back", "right"},
        {"back"},
        {"back", "left"},
        {"left"},
        {"forward", "left"},
        {"forward"},
        {"forward", "right"},
    ]
    return {bindings[k] for k in octant_map[octant]}


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


__all__ = ["accel_curve", "octant_keys"]
