"""Tests for the adaptive (dynamic) deadzone calibration on the spatial joystick.

Pure and deterministic: synthetic resting samples with controlled jitter drive the
calibration; no camera involved.
"""

from __future__ import annotations

import numpy as np

from minecraft_cv.joystick.deadzone import DeadzoneJoystick


def _joy(**kw: object) -> DeadzoneJoystick:
    base = dict(deadzone_radius=0.05, sensitivity=2.0, accel_exponent=2.0, max_output=1.0)
    base.update(kw)
    return DeadzoneJoystick(**base)  # type: ignore[arg-type]


def test_static_behavior_unchanged_when_disabled() -> None:
    """With dynamic off, the effective radius equals the configured base immediately."""
    joy = _joy(dynamic=False)
    joy.update(np.array([0.5, 0.5]))  # seeds neutral
    assert joy.calibrating is False
    assert joy.effective_deadzone_radius == 0.05


def test_calibration_holds_output_zero_then_grows_radius() -> None:
    """During the calibration window output is zero; afterwards the radius reflects jitter."""
    rng = np.random.default_rng(1)
    joy = _joy(dynamic=True, calibration_frames=100, dynamic_margin=2.0)
    center = np.array([0.5, 0.5])
    jitter_sigma = 0.02
    for _ in range(100):
        sample = center + rng.normal(0.0, jitter_sigma, size=2)
        out = joy.update(sample)
        assert np.allclose(out, 0.0)  # output suppressed throughout calibration
    assert joy.calibrating is False
    # The effective radius grew above the base by ~margin * (resting jitter radius).
    assert joy.effective_deadzone_radius > 0.05
    # Sanity bound: jitter radius is on the order of sigma, so growth is modest.
    assert joy.effective_deadzone_radius < 0.05 + 2.0 * 0.15


def test_calibrated_radius_suppresses_resting_but_passes_real_moves() -> None:
    rng = np.random.default_rng(2)
    joy = _joy(dynamic=True, calibration_frames=80, dynamic_margin=2.0)
    center = np.array([0.5, 0.5])
    for _ in range(80):
        joy.update(center + rng.normal(0.0, 0.02, size=2))
    radius = joy.effective_deadzone_radius
    # A sample within the resting band stays zero.
    assert np.allclose(joy.update(center + np.array([0.01, 0.0])), 0.0)
    # A clearly deliberate move well beyond the calibrated radius produces output.
    out = joy.update(center + np.array([radius + 0.3, 0.0]))
    assert np.linalg.norm(out) > 0.0


def test_reset_neutral_restarts_calibration() -> None:
    joy = _joy(dynamic=True, calibration_frames=10, dynamic_margin=1.0)
    center = np.array([0.5, 0.5])
    for _ in range(10):
        joy.update(center)
    assert joy.calibrating is False
    joy.reset_neutral()
    assert joy.calibrating is True
    assert joy.effective_deadzone_radius == 0.05  # reset to base floor
