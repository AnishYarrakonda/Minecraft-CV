"""Tests for the wrist-rotation virtual thumbstick."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from minecraft_cv.joystick.wrist_rotation import (
    WristRotationJoystick,
    palm_vector,
    palm_xz,
)


def test_palm_xz_translation_invariant(
    make_wrist_rotation_landmarks: Callable[..., np.ndarray],
) -> None:
    lm_a = make_wrist_rotation_landmarks(palm_x=0.2, palm_z=-0.2, offset=(0.1, 0.2, 0.3))
    lm_b = make_wrist_rotation_landmarks(palm_x=0.2, palm_z=-0.2, offset=(0.7, 0.1, -0.4))
    assert np.allclose(palm_vector(lm_a), palm_vector(lm_b))
    assert np.allclose(palm_xz(lm_a), palm_xz(lm_b))


def test_first_sample_sets_neutral(
    make_wrist_rotation_landmarks: Callable[..., np.ndarray],
) -> None:
    joystick = WristRotationJoystick(0.05, 2.0)
    neutral = palm_xz(make_wrist_rotation_landmarks())
    assert np.allclose(joystick.update(neutral), 0.0)
    assert joystick.neutral is not None


def test_axis_deadzone_and_linear_scaling() -> None:
    joystick = WristRotationJoystick(0.05, 2.0, max_output=1.0)
    assert np.allclose(joystick.update(np.array([0.0, 0.0])), 0.0)
    assert np.allclose(joystick.update(np.array([0.03, -0.03])), 0.0)
    out = joystick.update(np.array([0.10, -0.20]))
    assert out[0] == pytest.approx(0.10)
    assert out[1] == pytest.approx(-0.30)


def test_output_clamps_per_axis() -> None:
    joystick = WristRotationJoystick(0.05, 2.0, max_output=0.5)
    joystick.update(np.array([0.0, 0.0]))
    out = joystick.update(np.array([1.0, -1.0]))
    assert np.allclose(out, [0.5, -0.5])


def test_reset_neutral_reseeds() -> None:
    joystick = WristRotationJoystick(0.05, 2.0)
    joystick.update(np.array([0.0, 0.0]))
    joystick.reset_neutral()
    assert joystick.neutral is None
    assert np.allclose(joystick.update(np.array([0.4, -0.4])), 0.0)
    assert np.allclose(joystick.neutral, [0.4, -0.4])
