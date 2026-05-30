from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from minecraft_cv.gestures.finger_state import finger_extensions


def test_all_curled_low_ratios(make_extended_landmarks: Callable[..., np.ndarray]) -> None:
    # All fingers curled (ratio defaults to 0.8 in fixture), thumb_ext=0.3
    lm = make_extended_landmarks({}, thumb_ext=0.3)
    fs = finger_extensions(lm)

    assert fs.thumb_ext == pytest.approx(0.3, rel=1e-2)
    assert fs.index_ext == pytest.approx(0.8, rel=1e-2)
    assert fs.middle_ext == pytest.approx(0.8, rel=1e-2)
    assert fs.ring_ext == pytest.approx(0.8, rel=1e-2)
    assert fs.pinky_ext == pytest.approx(0.8, rel=1e-2)


def test_single_finger_extended(make_extended_landmarks: Callable[..., np.ndarray]) -> None:
    # Index extended to 1.3, others remain at default 0.8
    lm = make_extended_landmarks({"index": 1.3}, thumb_ext=0.3)
    fs = finger_extensions(lm)

    assert fs.index_ext > 1.2
    assert fs.middle_ext < 1.0
    assert fs.ring_ext < 1.0
    assert fs.pinky_ext < 1.0


def test_thumb_extended(make_extended_landmarks: Callable[..., np.ndarray]) -> None:
    # Thumb extended to 1.5, others default curled
    lm = make_extended_landmarks({}, thumb_ext=1.5)
    fs = finger_extensions(lm)

    assert fs.thumb_ext == pytest.approx(1.5, rel=1e-2)
    assert fs.index_ext < 1.0


def test_peace_sign(make_extended_landmarks: Callable[..., np.ndarray]) -> None:
    # Index + Middle extended (ratio 1.3)
    lm = make_extended_landmarks({"index": 1.3, "middle": 1.3}, thumb_ext=0.3)
    fs = finger_extensions(lm)

    assert fs.index_ext > 1.2
    assert fs.middle_ext > 1.2
    assert fs.ring_ext < 1.0
    assert fs.pinky_ext < 1.0
