"""Hotbar scroll tests — scroll moved off the hands onto head-roll (ear-to-shoulder tilt)."""
from __future__ import annotations

import math

import numpy as np

from conftest import make_screen_settings
from minecraft_cv.gestures.face_gestures import LEFT_EYE_OUTER, RIGHT_EYE_OUTER
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.face_tracker import FaceResult


def _face_with_roll(deg: float) -> FaceResult:
    """A FaceResult whose eye-corner line is rolled ``deg`` degrees in the image plane."""
    lm = np.full((478, 3), 0.5, dtype=np.float32)
    rad = math.radians(deg)
    left = np.array([0.45, 0.5, 0.0], dtype=np.float32)
    right = left + np.array([math.cos(rad) * 0.1, math.sin(rad) * 0.1, 0.0], dtype=np.float32)
    lm[LEFT_EYE_OUTER] = left
    lm[RIGHT_EYE_OUTER] = right
    return FaceResult(blendshapes={}, landmarks=lm)


def _scrolls(emitter: NullEmitter) -> list[int]:
    return [int(e[1]) for e in emitter.log if e[0] == "scroll"]


def test_head_roll_left_scrolls_up(null_emitter: NullEmitter) -> None:
    pipe = Pipeline.from_settings(make_screen_settings(), emitter=null_emitter)
    face = _face_with_roll(20.0)  # past the 12 deg engage threshold
    for _ in range(4):
        pipe.step([], face)
    scrolls = _scrolls(null_emitter)
    assert scrolls, "expected a scroll tick from head roll"
    assert scrolls[0] > 0  # positive = up (hotbar next)


def test_head_roll_right_scrolls_down(null_emitter: NullEmitter) -> None:
    pipe = Pipeline.from_settings(make_screen_settings(), emitter=null_emitter)
    face = _face_with_roll(-20.0)
    for _ in range(4):
        pipe.step([], face)
    scrolls = _scrolls(null_emitter)
    assert scrolls
    assert scrolls[0] < 0  # negative = down (hotbar prev)


def test_head_upright_does_not_scroll(null_emitter: NullEmitter) -> None:
    pipe = Pipeline.from_settings(make_screen_settings(), emitter=null_emitter)
    face = _face_with_roll(0.0)
    for _ in range(4):
        pipe.step([], face)
    assert not _scrolls(null_emitter)


def test_held_head_roll_repeats_scroll(null_emitter: NullEmitter) -> None:
    """A sustained tilt re-emits scroll ticks via the repeat-rate path."""
    pipe = Pipeline.from_settings(make_screen_settings(), emitter=null_emitter)
    now = {"t": 0.0}
    pipe._clock = lambda: now["t"]  # type: ignore[method-assign]
    face = _face_with_roll(20.0)
    for _ in range(10):
        now["t"] += 0.25  # 4 Hz stepping > 1/8 s repeat interval, so ticks accumulate
        pipe.step([], face)
    # Engage tick + several repeats while held.
    assert len(_scrolls(null_emitter)) >= 3
