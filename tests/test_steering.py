"""Pure unit tests for the steering helpers: octant_keys and accel_curve.

No camera, no MediaPipe, no OS input — all deterministic math.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from minecraft_cv.joystick.steering import accel_curve, octant_keys

_BINDINGS = {"right": "d", "left": "a", "forward": "w", "back": "s"}


# ---------------------------------------------------------------------------
# octant_keys
# ---------------------------------------------------------------------------


def test_zero_vector_returns_empty() -> None:
    assert octant_keys(np.zeros(2), _BINDINGS) == set()


def test_pure_right_fires_only_d() -> None:
    # +x is Right (0 degrees)
    out = np.array([1.0, 0.0])
    assert octant_keys(out, _BINDINGS) == {"d"}


def test_pure_back_fires_only_s() -> None:
    # +y is Down/Back (+90 degrees in image coords)
    out = np.array([0.0, 1.0])
    assert octant_keys(out, _BINDINGS) == {"s"}


def test_pure_left_fires_only_a() -> None:
    # -x is Left (180 degrees)
    out = np.array([-1.0, 0.0])
    assert octant_keys(out, _BINDINGS) == {"a"}


def test_pure_forward_fires_only_w() -> None:
    # -y is Up/Forward (-90 degrees)
    out = np.array([0.0, -1.0])
    assert octant_keys(out, _BINDINGS) == {"w"}


def test_diagonals_fire_two_keys() -> None:
    # 45 degrees: Back-Right
    out_sd = np.array([1.0, 1.0])
    assert octant_keys(out_sd, _BINDINGS) == {"s", "d"}

    # 135 degrees: Back-Left
    out_sa = np.array([-1.0, 1.0])
    assert octant_keys(out_sa, _BINDINGS) == {"s", "a"}

    # -135 degrees: Forward-Left
    out_wa = np.array([-1.0, -1.0])
    assert octant_keys(out_wa, _BINDINGS) == {"w", "a"}

    # -45 degrees: Forward-Right
    out_wd = np.array([1.0, -1.0])
    assert octant_keys(out_wd, _BINDINGS) == {"w", "d"}


def test_octant_boundaries() -> None:
    """Test angles right on the slice boundaries (multiples of 22.5 deg).
    
    math.atan2(-0.41421356, 1.0) is approx -22.5 deg.
    """
    # 22.4 degrees -> still Right
    ang1 = math.radians(22.4)
    assert octant_keys(np.array([math.cos(ang1), math.sin(ang1)]), _BINDINGS) == {"d"}
    
    # 22.6 degrees -> crosses into Back-Right
    ang2 = math.radians(22.6)
    assert octant_keys(np.array([math.cos(ang2), math.sin(ang2)]), _BINDINGS) == {"s", "d"}


def test_accepts_longer_arrays() -> None:
    """Arrays longer than 2 elements should work (only first two used)."""
    out = np.array([0.0, -0.5, 999.0])
    assert octant_keys(out, _BINDINGS) == {"w"}


# ---------------------------------------------------------------------------
# accel_curve
# ---------------------------------------------------------------------------


def test_zero_input_returns_zeros() -> None:
    assert np.allclose(accel_curve(np.zeros(2), exponent=2.0, max_output=1.0), 0.0)


def test_direction_preserved() -> None:
    """Output direction must match input direction exactly."""
    vec = np.array([0.3, 0.4])
    out = accel_curve(vec, exponent=2.0, max_output=1.0)
    unit_in = vec / np.linalg.norm(vec)
    unit_out = out / np.linalg.norm(out)
    assert np.allclose(unit_in, unit_out, atol=1e-9)


def test_saturates_at_max_output() -> None:
    vec = np.array([2.0, 0.0])
    out = accel_curve(vec, exponent=1.5, max_output=1.0)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-9)


def test_exponent_gt_1_shrinks_small_inputs() -> None:
    """With exponent>1, a small input produces a proportionally smaller output than linear."""
    small = np.array([0.1, 0.0])
    linear = accel_curve(small, exponent=1.0, max_output=1.0)
    curved = accel_curve(small, exponent=2.0, max_output=1.0)
    assert np.linalg.norm(curved) < np.linalg.norm(linear)


def test_monotonic_magnitude() -> None:
    """Larger input magnitude -> larger output magnitude."""
    mags = [accel_curve(np.array([v, 0.0]), 2.0, 1.0)[0] for v in [0.1, 0.3, 0.6, 0.9]]
    assert all(mags[i] < mags[i + 1] for i in range(len(mags) - 1))


def test_exponent_1_is_linear_passthrough() -> None:
    vec = np.array([0.4, 0.0])
    out = accel_curve(vec, exponent=1.0, max_output=1.0)
    assert np.allclose(out, vec, atol=1e-9)


def test_continuous_at_zero() -> None:
    """Output approach 0 as input magnitude -> 0."""
    for eps in [1e-3, 1e-4, 1e-5]:
        out = accel_curve(np.array([eps, 0.0]), 2.0, 1.0)
        assert float(out[0]) < eps
