"""Pipeline coverage for the ring/pinky Hotbar gestures (Task 1).

The project rules (``.claude/rules/gestures.md``) bind the hotbar to the **scroll wheel**
(ring -> scroll up = Hotbar Next, pinky -> scroll down = Hotbar Prev) with a momentary-pulse
+ repeat-rate model, rather than the number keys 1-9. These tests assert that the ring/pinky
Schmitt triggers fire scroll ticks, repeat while held, and are fully independent of the
index/middle (attack/use) triggers.

Convention: ``handedness="Left"`` is swapped to the physical **right** hand (the pinch hand).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from conftest import make_calibrated_settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _pipe(emitter: NullEmitter, clock: _Clock) -> Pipeline:
    p = Pipeline.from_settings(make_calibrated_settings(), emitter=emitter)
    p._clock = clock  # type: ignore[assignment]
    return p


def test_ring_pinch_scrolls_up(
    make_palm_normal_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    emitter = NullEmitter()
    pipe = _pipe(emitter, _Clock())
    pipe.step([make_hand_result(make_palm_normal_landmarks(distances={"ring": 0.20}), "Left")])
    pipe.step([make_hand_result(make_palm_normal_landmarks(distances={"ring": 0.20}), "Left")])
    assert ("scroll", "1") in emitter.log


def test_pinky_pinch_scrolls_down(
    make_palm_normal_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    emitter = NullEmitter()
    pipe = _pipe(emitter, _Clock())
    pipe.step([make_hand_result(make_palm_normal_landmarks(distances={"pinky": 0.20}), "Left")])
    pipe.step([make_hand_result(make_palm_normal_landmarks(distances={"pinky": 0.20}), "Left")])
    assert ("scroll", "-1") in emitter.log


def test_hotbar_independent_of_attack(
    make_palm_normal_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Index (attack) + ring (hotbar) pinch simultaneously: both fire, no conflict."""
    emitter = NullEmitter()
    pipe = _pipe(emitter, _Clock())
    lm = make_palm_normal_landmarks(distances={"index": 0.20, "ring": 0.20})
    pipe.step([make_hand_result(lm, "Left")])
    pipe.step([make_hand_result(lm, "Left")])
    assert ("key_down", "mouse_left") in emitter.log  # attack engaged
    assert ("scroll", "1") in emitter.log             # hotbar next fired
    # The hotbar pinch must NOT have produced a button key_down (it is scroll-only).
    assert ("key_down", "scroll_up") not in emitter.log


def test_held_hotbar_pinch_repeats_at_rate(
    make_palm_normal_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Holding the ring pinch re-emits scroll ticks at scroll_repeat_rate_hz (default 8)."""
    emitter = NullEmitter()
    clock = _Clock()
    pipe = _pipe(emitter, clock)
    held = make_hand_result(make_palm_normal_landmarks(distances={"ring": 0.20}), "Left")
    clock.t = 0.0
    pipe.step([held])  # frame 1
    pipe.step([held])  # engage -> 1 tick
    clock.t = 0.05
    pipe.step([held])  # < 1/8 s since last -> no repeat
    clock.t = 0.20
    pipe.step([held])  # > 1/8 s -> repeat tick
    scrolls = [e for e in emitter.log if e == ("scroll", "1")]
    assert len(scrolls) == 2


def test_hotbar_release_stops_repeat(
    make_palm_normal_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    emitter = NullEmitter()
    clock = _Clock()
    pipe = _pipe(emitter, clock)
    clock.t = 0.0
    pipe.step([make_hand_result(make_palm_normal_landmarks(distances={"ring": 0.20}), "Left")])
    pipe.step([make_hand_result(make_palm_normal_landmarks(distances={"ring": 0.20}), "Left")])
    clock.t = 0.1
    pipe.step([make_hand_result(make_palm_normal_landmarks(distances={"ring": 1.0}), "Left")])
    before = len([e for e in emitter.log if e == ("scroll", "1")])
    clock.t = 0.5
    pipe.step([make_hand_result(make_palm_normal_landmarks(distances={"ring": 1.0}), "Left")])
    after = len([e for e in emitter.log if e == ("scroll", "1")])
    assert before == after
