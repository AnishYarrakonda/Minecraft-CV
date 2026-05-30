"""Tests for the DeadzoneJoystick (sphere deadzone, accel curve, recenter)."""

from __future__ import annotations

import numpy as np
import pytest

from minecraft_cv.joystick.deadzone import ANCHOR_INDEX, DeadzoneJoystick, anchor_xy

NEUTRAL = np.array([0.5, 0.5])


def _joystick(
    deadzone: float = 0.05, sensitivity: float = 2.0, exponent: float = 2.0, max_out: float = 1.0
) -> DeadzoneJoystick:
    j = DeadzoneJoystick(deadzone, sensitivity, exponent, max_out)
    j.recenter(NEUTRAL)
    return j


def test_first_sample_sets_neutral_and_outputs_zero() -> None:
    j = DeadzoneJoystick(0.05, 2.0, 2.0)
    out = j.update(np.array([0.42, 0.61]))
    assert np.allclose(out, 0.0)
    assert j.neutral is not None
    assert np.allclose(j.neutral, [0.42, 0.61])


def test_inside_deadzone_is_zero() -> None:
    j = _joystick()
    assert np.allclose(j.update(NEUTRAL + np.array([0.03, 0.0])), 0.0)
    assert np.allclose(j.update(NEUTRAL), 0.0)


def test_boundary_is_continuous_zero() -> None:
    j = _joystick(deadzone=0.05)
    # Exactly at the sphere boundary -> zero (continuous, no step discontinuity).
    assert np.allclose(j.update(NEUTRAL + np.array([0.05, 0.0])), 0.0)
    # Just outside -> small but non-zero, and continuous (tiny) thanks to the accel curve.
    out = j.update(NEUTRAL + np.array([0.06, 0.0]))
    mag = float(np.linalg.norm(out))
    assert 0.0 < mag < 0.01


def test_outside_deadzone_proportional_direction() -> None:
    j = _joystick()
    out = j.update(NEUTRAL + np.array([0.30, 0.0]))
    assert out[0] > 0.0
    assert out[1] == pytest.approx(0.0, abs=1e-9)


def test_sphere_not_box_diagonal_equals_cardinal() -> None:
    j_card = _joystick()
    j_diag = _joystick()
    d = 0.30
    cardinal = j_card.update(NEUTRAL + np.array([d, 0.0]))
    diagonal = j_diag.update(NEUTRAL + np.array([d / np.sqrt(2), d / np.sqrt(2)]))
    # Same Euclidean distance -> same output magnitude (deadzone is a sphere).
    assert np.linalg.norm(cardinal) == pytest.approx(np.linalg.norm(diagonal), abs=1e-9)


def test_acceleration_curve_is_monotonic() -> None:
    j = _joystick()
    mags = [
        float(np.linalg.norm(j.update(NEUTRAL + np.array([dx, 0.0]))))
        for dx in [0.06, 0.10, 0.20, 0.35, 0.50]
    ]
    assert mags == sorted(mags)
    assert mags[0] < mags[-1]


def test_output_clamped_to_max() -> None:
    j = _joystick(max_out=1.0)
    out = j.update(NEUTRAL + np.array([0.9, 0.0]))
    assert float(np.linalg.norm(out)) == pytest.approx(1.0, abs=1e-9)


def test_recenter_changes_neutral() -> None:
    j = _joystick()
    new = np.array([0.2, 0.8])
    j.recenter(new)
    assert np.allclose(j.neutral, new)
    assert np.allclose(j.update(new), 0.0)


def test_reset_neutral_recalibrates() -> None:
    j = _joystick()
    j.reset_neutral()
    assert j.neutral is None
    # Next sample becomes the new neutral.
    j.update(np.array([0.1, 0.1]))
    assert np.allclose(j.neutral, [0.1, 0.1])


def test_anchor_xy_extracts_configured_landmark() -> None:
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[ANCHOR_INDEX["wrist"]] = (0.1, 0.2, 0.3)
    lm[ANCHOR_INDEX["middle_mcp"]] = (0.4, 0.5, 0.6)
    assert np.allclose(anchor_xy(lm, "wrist"), [0.1, 0.2])
    assert np.allclose(anchor_xy(lm, "middle_mcp"), [0.4, 0.5])
