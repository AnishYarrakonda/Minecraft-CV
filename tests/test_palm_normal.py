"""Tests for calibrated palm-normal virtual thumbsticks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from minecraft_cv.joystick.palm_normal import (
    PalmNormalJoystick,
    palm_normal,
    palm_normal_xy,
)


def test_palm_normal_translation_invariant(
    make_palm_normal_landmarks: Callable[..., np.ndarray],
) -> None:
    lm_a = make_palm_normal_landmarks(normal_xy=(0.2, -0.2), offset=(0.1, 0.2, 0.3))
    lm_b = make_palm_normal_landmarks(normal_xy=(0.2, -0.2), offset=(0.7, 0.1, -0.4))
    assert np.allclose(palm_normal(lm_a), palm_normal(lm_b))
    assert np.allclose(palm_normal_xy(lm_a), palm_normal_xy(lm_b))


def test_palm_normal_is_unit_and_sign_stable(
    make_palm_normal_landmarks: Callable[..., np.ndarray],
) -> None:
    lm = make_palm_normal_landmarks(normal_xy=(0.25, 0.15))
    normal = palm_normal(lm)
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert normal[0] == pytest.approx(0.25, abs=1e-5)
    assert normal[1] == pytest.approx(0.15, abs=1e-5)
    assert normal[2] > 0.0


def test_calibrated_neutral_outputs_zero() -> None:
    joystick = PalmNormalJoystick(neutral=(0.1, -0.1), deadzone=0.05, sensitivity=(2.0, 2.0))
    assert np.allclose(joystick.update(np.array([0.1, -0.1])), 0.0)


def test_uncalibrated_preview_uses_first_sample_as_temporary_neutral() -> None:
    joystick = PalmNormalJoystick(neutral=None, deadzone=0.05, sensitivity=(2.0, 2.0))
    assert np.allclose(joystick.update(np.array([0.4, -0.2])), 0.0)

    out = joystick.update(np.array([0.55, -0.2]))
    assert out[0] == pytest.approx(0.2)
    assert out[1] == pytest.approx(0.0)


def test_axis_deadzone_scaling_and_clamp() -> None:
    joystick = PalmNormalJoystick(
        neutral=(0.0, 0.0), deadzone=0.05, sensitivity=(2.0, 3.0), max_output=0.5
    )
    assert np.allclose(joystick.update(np.array([0.03, -0.03])), 0.0)
    out = joystick.update(np.array([0.10, -0.20]))
    assert out[0] == pytest.approx(0.10)
    assert out[1] == pytest.approx(-0.45)
    assert np.allclose(joystick.update(np.array([1.0, -1.0])), [0.5, -0.5])


# ---------------------------------------------------------------------------
# Asymmetric gain (sensitivity_neg)
# ---------------------------------------------------------------------------


def test_sensitivity_neg_none_is_symmetric() -> None:
    """With sensitivity_neg=None the behavior is identical to the original symmetric path."""
    j_sym = PalmNormalJoystick(neutral=(0.0, 0.0), deadzone=0.0, sensitivity=(3.0, 3.0))
    j_asym = PalmNormalJoystick(
        neutral=(0.0, 0.0), deadzone=0.0, sensitivity=(3.0, 3.0), sensitivity_neg=None
    )
    sig = np.array([0.2, -0.2])
    assert np.allclose(j_sym.update(sig), j_asym.update(sig))


def test_negative_delta_uses_sensitivity_neg() -> None:
    """Negative delta on y should use sensitivity_neg, not sensitivity."""
    j = PalmNormalJoystick(
        neutral=(0.0, 0.0),
        deadzone=0.0,
        sensitivity=(2.0, 2.0),
        sensitivity_neg=(2.0, 8.0),  # y-negative gets 4× more gain
    )
    # Positive y: uses sensitivity[1] = 2.0
    pos_out = j.update(np.array([0.0, 0.1]))
    j.reset_neutral()
    # Negative y: uses sensitivity_neg[1] = 8.0
    neg_out = j.update(np.array([0.0, -0.1]))
    assert pos_out[1] == pytest.approx(0.2)
    assert neg_out[1] == pytest.approx(-0.8)


def test_positive_delta_still_uses_positive_sensitivity() -> None:
    """Positive delta should use sensitivity (not sensitivity_neg), even when neg is set."""
    j = PalmNormalJoystick(
        neutral=(0.0, 0.0),
        deadzone=0.0,
        sensitivity=(5.0, 5.0),
        sensitivity_neg=(1.0, 1.0),
    )
    out = j.update(np.array([0.1, 0.0]))
    assert out[0] == pytest.approx(0.5)  # 0.1 * 5.0


def test_large_back_gain_gives_comparable_travel() -> None:
    """A small back tilt with a large back gain should produce output comparable to a larger forward tilt."""
    j = PalmNormalJoystick(
        neutral=(0.0, 0.0),
        deadzone=0.05,
        sensitivity=(4.0, 4.0),       # forward reach ~0.3, gain ~4
        sensitivity_neg=(4.0, 12.0),  # back reach ~0.1, gain ~12 -> same effective travel
        max_output=1.0,
    )
    # Large forward tilt: (0.3 - 0.05) * 4 = 1.0 (clamped)
    fwd_out = j.update(np.array([0.0, 0.3]))
    j.reset_neutral()
    # Small back tilt: (0.1 - 0.05) * 12 = 0.6 (meaningful output despite tiny reach)
    back_out = j.update(np.array([0.0, -0.1]))
    assert fwd_out[1] == pytest.approx(1.0)
    assert abs(back_out[1]) == pytest.approx(0.6)


def test_sensitivity_neg_max_output_clamped() -> None:
    """Even with a boosted back gain the output must not exceed max_output."""
    j = PalmNormalJoystick(
        neutral=(0.0, 0.0),
        deadzone=0.0,
        sensitivity=(2.0, 2.0),
        sensitivity_neg=(2.0, 100.0),
        max_output=0.5,
    )
    out = j.update(np.array([0.0, -0.3]))
    assert out[1] == pytest.approx(-0.5)
