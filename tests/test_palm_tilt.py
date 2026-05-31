"""Tests for the knuckle-tilt joystick signal (default ``palm_tilt`` mode).

The original ``palm_normal`` signal was unreliable because tilting the hand left vs right
projected to nearly the same ``(x, y)`` and the normal's hemisphere clamp collapsed the two.
These tests pin the property that fixes that: the tilt signal is sign-stable and distinct for
left vs right (and up vs down), while staying invariant to whole-hand translation and to
finger curl / pinch.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from minecraft_cv.joystick.palm_tilt import palm_tilt_xy


def _expected(tx: float, ty: float, depth: float = 1.0) -> tuple[float, float]:
    norm = math.sqrt(tx * tx + ty * ty + depth * depth)
    return tx / norm, ty / norm


def test_neutral_hand_is_near_zero(make_tilt_landmarks: Callable[..., np.ndarray]) -> None:
    out = palm_tilt_xy(make_tilt_landmarks(tilt=(0.0, 0.0)))
    assert np.allclose(out, 0.0, atol=1e-6)


def test_exact_value_matches_geometry(make_tilt_landmarks: Callable[..., np.ndarray]) -> None:
    out = palm_tilt_xy(make_tilt_landmarks(tilt=(0.3, -0.2)))
    assert out[0] == pytest.approx(_expected(0.3, -0.2)[0], abs=1e-5)
    assert out[1] == pytest.approx(_expected(0.3, -0.2)[1], abs=1e-5)


def test_left_and_right_tilt_are_distinct(
    make_tilt_landmarks: Callable[..., np.ndarray],
) -> None:
    """The core regression: tilting right vs left must not collapse to the same signal."""
    right = palm_tilt_xy(make_tilt_landmarks(tilt=(0.4, 0.0)))
    left = palm_tilt_xy(make_tilt_landmarks(tilt=(-0.4, 0.0)))
    assert right[0] > 0.0
    assert left[0] < 0.0
    assert right[0] == pytest.approx(-left[0], abs=1e-6)
    assert not np.allclose(right, left)


def test_up_and_down_tilt_are_distinct(
    make_tilt_landmarks: Callable[..., np.ndarray],
) -> None:
    up = palm_tilt_xy(make_tilt_landmarks(tilt=(0.0, -0.4)))
    down = palm_tilt_xy(make_tilt_landmarks(tilt=(0.0, 0.4)))
    assert up[1] < 0.0
    assert down[1] > 0.0


def test_translation_invariant(make_tilt_landmarks: Callable[..., np.ndarray]) -> None:
    a = palm_tilt_xy(make_tilt_landmarks(tilt=(0.3, 0.1), offset=(0.1, 0.2, 0.3)))
    b = palm_tilt_xy(make_tilt_landmarks(tilt=(0.3, 0.1), offset=(0.7, -0.1, -0.4)))
    assert np.allclose(a, b, atol=1e-6)


def test_finger_curl_and_pinch_invariant(
    make_tilt_landmarks: Callable[..., np.ndarray],
) -> None:
    """Moving fingertips (curl / thumb pinch) must not move the steering signal."""
    open_hand = palm_tilt_xy(make_tilt_landmarks(tilt=(0.25, -0.15), fingertip=(0.05, 0.05, 0.0)))
    pinched = palm_tilt_xy(make_tilt_landmarks(tilt=(0.25, -0.15), fingertip=(-0.3, 0.4, 0.2)))
    assert np.allclose(open_hand, pinched, atol=1e-6)


def test_scale_invariant(make_tilt_landmarks: Callable[..., np.ndarray]) -> None:
    near = palm_tilt_xy(make_tilt_landmarks(tilt=(0.3, 0.2), scale=0.2))
    far = palm_tilt_xy(make_tilt_landmarks(tilt=(0.3, 0.2), scale=0.4))
    assert np.allclose(near, far, atol=1e-6)
