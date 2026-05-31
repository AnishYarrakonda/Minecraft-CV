"""Pure unit tests for the steering helpers: cardinal_keys and accel_curve.

No camera, no MediaPipe, no OS input — all deterministic math.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from minecraft_cv.joystick.steering import accel_curve, cardinal_keys

_BINDINGS = {"right": "d", "left": "a", "forward": "w", "back": "s"}


# ---------------------------------------------------------------------------
# cardinal_keys
# ---------------------------------------------------------------------------


def test_zero_vector_returns_empty() -> None:
    assert cardinal_keys(np.zeros(2), 35.0, _BINDINGS) == set()


def test_pure_forward_fires_only_w() -> None:
    out = np.array([0.0, 0.5])
    assert cardinal_keys(out, 35.0, _BINDINGS) == {"w"}


def test_pure_back_fires_only_s() -> None:
    out = np.array([0.0, -0.5])
    assert cardinal_keys(out, 35.0, _BINDINGS) == {"s"}


def test_pure_right_fires_only_d() -> None:
    out = np.array([0.5, 0.0])
    assert cardinal_keys(out, 35.0, _BINDINGS) == {"d"}


def test_pure_left_fires_only_a() -> None:
    out = np.array([-0.5, 0.0])
    assert cardinal_keys(out, 35.0, _BINDINGS) == {"a"}


def test_anti_drift_mostly_forward_slight_sideways() -> None:
    """The key anti-drift case: small x deviation on mostly-forward tilt -> only W."""
    # Angle ≈ 80° from +x axis -> forward zone (90°). With half_width=35, firing radius=55°.
    # Angular distance from 80° to 90° = 10° which is ≤ 55°. Also check D doesn't fire:
    # distance from 80° to 0° = 80° which is > 55°.
    out = np.array([0.1, 0.6])  # clearly more forward than sideways
    result = cardinal_keys(out, 35.0, _BINDINGS)
    assert "w" in result
    assert "d" not in result


def test_diagonal_near_45_fires_both_w_and_d() -> None:
    """At exactly 45° with half_width=35 both W and D should fire (diagonal band)."""
    # 45° is equidistant from 0° (right) and 90° (forward).
    # firing_radius = 90 - 35 = 55°. dist(45°, 0°) = 45° ≤ 55°; dist(45°, 90°) = 45° ≤ 55°.
    out = np.array([0.5, 0.5])
    result = cardinal_keys(out, 35.0, _BINDINGS)
    assert "w" in result
    assert "d" in result


def test_half_width_45_no_diagonals() -> None:
    """half_width=45 means firing_radius=45°; at 45° exactly, dist equals threshold -> no overlap."""
    # dist(45°, 0°) = 45°. With firing_radius=45, only cardinals within exactly 45° fire.
    # Strict inequality (<=) at exactly 45° fires, so let's use an angle slightly past 45°.
    # We test a clearly-diagonal input to confirm that half_width=45 suppresses double-fire.
    # At 67.5° from +x: dist(67.5°, 0°)=67.5°>45; dist(67.5°, 90°)=22.5°<=45.
    angle = math.radians(67.5)
    out = np.array([math.cos(angle), math.sin(angle)])
    result = cardinal_keys(out, 45.0, _BINDINGS)
    assert "w" in result
    assert "d" not in result


def test_half_width_0_any_nonzero_component_fires() -> None:
    """half_width=0 -> firing_radius=90 -> any nonzero vector fires at least one key."""
    # A purely rightward vector should fire D.
    assert "d" in cardinal_keys(np.array([0.1, 0.0]), 0.0, _BINDINGS)
    # A purely forward vector should fire W.
    assert "w" in cardinal_keys(np.array([0.0, 0.1]), 0.0, _BINDINGS)


def test_back_fires_for_negative_y() -> None:
    """Negative y output (back) should fire S, not W."""
    out = np.array([0.0, -0.3])
    result = cardinal_keys(out, 35.0, _BINDINGS)
    assert "s" in result
    assert "w" not in result


def test_accepts_longer_arrays() -> None:
    """Arrays longer than 2 elements should work (only first two used)."""
    out = np.array([0.0, 0.5, 999.0])
    assert cardinal_keys(out, 35.0, _BINDINGS) == {"w"}


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
    """Output should approach 0 as input magnitude -> 0."""
    for eps in [1e-3, 1e-4, 1e-5]:
        out = accel_curve(np.array([eps, 0.0]), 2.0, 1.0)
        assert float(out[0]) < eps  # exponent 2 -> output smaller than input
