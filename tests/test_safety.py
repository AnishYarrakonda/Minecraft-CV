"""Tests for the tracking-loss fail-safe (no stuck keys on dropout)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from minecraft_cv.gestures.pinch import KEY_DOWN, KEY_UP, GestureSpec, PinchStateMachine
from minecraft_cv.gestures.safety import TrackingLossGuard

LEFT = {
    "jump": GestureSpec("index", 0.30, 0.45),
    "sneak": GestureSpec("middle", 0.30, 0.45),
}
RIGHT = {
    "attack": GestureSpec("index", 0.30, 0.45),
    "use": GestureSpec("middle", 0.30, 0.45),
}


def _guard() -> TrackingLossGuard:
    return TrackingLossGuard(PinchStateMachine("left", LEFT), PinchStateMachine("right", RIGHT))


def _names(events: list) -> set[tuple[str, str, str]]:
    return {(e.gesture, e.action, e.hand) for e in events}


def test_left_hand_dropout_releases_held_key(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    guard = _guard()
    # Left hand holds jump; right hand absent.
    guard.process(make_landmarks({"index": 0.20}), None)
    held = guard.process(make_landmarks({"index": 0.20}), None)
    assert _names(held) == {("jump", KEY_DOWN, "left")}
    # Left hand now disappears -> jump must be released.
    dropped = guard.process(None, None)
    assert _names(dropped) == {("jump", KEY_UP, "left")}
    # Subsequent absent frames are silent (idempotent).
    assert guard.process(None, None) == []


def test_both_hands_dropout_releases_everything(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    guard = _guard()
    guard.process(make_landmarks({"index": 0.20, "middle": 0.20}), make_landmarks({"index": 0.20}))
    guard.process(make_landmarks({"index": 0.20, "middle": 0.20}), make_landmarks({"index": 0.20}))
    dropped = guard.process(None, None)
    assert _names(dropped) == {
        ("jump", KEY_UP, "left"),
        ("sneak", KEY_UP, "left"),
        ("attack", KEY_UP, "right"),
    }


def test_hand_reentry_is_clean(make_landmarks: Callable[..., np.ndarray]) -> None:
    guard = _guard()
    guard.process(make_landmarks({"index": 0.20}), None)  # hold jump
    guard.process(make_landmarks({"index": 0.20}), None)  # hold jump
    guard.process(None, None)  # drop -> release
    # Re-enter with finger open (released): no spurious events, state is clean.
    guard.process(make_landmarks({"index": 0.60}), None)
    reentry = guard.process(make_landmarks({"index": 0.60}), None)
    assert reentry == []
    # And it can engage fresh again.
    guard.process(make_landmarks({"index": 0.20}), None)
    again = guard.process(make_landmarks({"index": 0.20}), None)
    assert _names(again) == {("jump", KEY_DOWN, "left")}


def test_release_all(make_landmarks: Callable[..., np.ndarray]) -> None:
    guard = _guard()
    guard.process(make_landmarks({"index": 0.20}), make_landmarks({"middle": 0.20}))
    guard.process(make_landmarks({"index": 0.20}), make_landmarks({"middle": 0.20}))
    released = guard.release_all()
    assert _names(released) == {("jump", KEY_UP, "left"), ("use", KEY_UP, "right")}


def test_present_hand_unaffected_by_other_dropout(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    guard = _guard()
    # Right hand holds attack the whole time; left hand flickers in and out.
    guard.process(make_landmarks({"index": 0.60}), make_landmarks({"index": 0.20}))
    guard.process(make_landmarks({"index": 0.60}), make_landmarks({"index": 0.20}))
    events = guard.process(None, make_landmarks({"index": 0.20}))
    # Right attack stays held (no new event); left had nothing held -> no release.
    assert events == []
