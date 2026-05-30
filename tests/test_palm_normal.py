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


def test_axis_deadzone_scaling_and_clamp() -> None:
    joystick = PalmNormalJoystick(
        neutral=(0.0, 0.0), deadzone=0.05, sensitivity=(2.0, 3.0), max_output=0.5
    )
    assert np.allclose(joystick.update(np.array([0.03, -0.03])), 0.0)
    out = joystick.update(np.array([0.10, -0.20]))
    assert out[0] == pytest.approx(0.10)
    assert out[1] == pytest.approx(-0.45)
    assert np.allclose(joystick.update(np.array([1.0, -1.0])), [0.5, -0.5])
