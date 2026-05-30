"""Tests for PinchStateMachine and the vectorized normalized-distance helper."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from minecraft_cv.gestures.pinch import (
    KEY_DOWN,
    KEY_UP,
    GestureSpec,
    PinchStateMachine,
    normalized_distances,
)

LEFT_GESTURES = {
    "jump": GestureSpec(finger="index", t_engage=0.30, t_release=0.45),
    "sneak": GestureSpec(finger="middle", t_engage=0.30, t_release=0.45),
}


def _names(events: list) -> set[tuple[str, str]]:
    return {(e.gesture, e.action) for e in events}


def test_normalized_distances_match_construction(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    lm = make_landmarks({"index": 0.2, "middle": 0.5, "ring": 0.8, "pinky": 0.9})
    dists = normalized_distances(lm)
    assert dists["index"] == pytest.approx(0.2, abs=1e-5)
    assert dists["middle"] == pytest.approx(0.5, abs=1e-5)
    assert dists["ring"] == pytest.approx(0.8, abs=1e-5)
    assert dists["pinky"] == pytest.approx(0.9, abs=1e-5)


def test_single_gesture_engage_release(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    sm = PinchStateMachine("left", LEFT_GESTURES)
    sm.update(make_landmarks({"index": 0.20}))
    engaged = sm.update(make_landmarks({"index": 0.20}))
    assert _names(engaged) == {("jump", KEY_DOWN)}
    assert sm.update(make_landmarks({"index": 0.20})) == []  # still holding, no new event
    sm.update(make_landmarks({"index": 0.60}))
    released = sm.update(make_landmarks({"index": 0.60}))
    assert _names(released) == {("jump", KEY_UP)}


def test_two_gestures_same_hand_concurrent(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    sm = PinchStateMachine("left", LEFT_GESTURES)
    sm.update(make_landmarks({"index": 0.20, "middle": 0.20}))
    events = sm.update(make_landmarks({"index": 0.20, "middle": 0.20}))
    assert _names(events) == {("jump", KEY_DOWN), ("sneak", KEY_DOWN)}
    assert all(e.hand == "left" for e in events)
    assert sm.held == {"jump", "sneak"}


@pytest.mark.parametrize("scale", [0.08, 0.2, 0.5])
def test_thresholds_are_scale_invariant(
    make_landmarks: Callable[..., np.ndarray], scale: float
) -> None:
    # Same normalized distance (0.2) at very different hand scales must engage identically.
    sm = PinchStateMachine("right", {"attack": GestureSpec("index", 0.30, 0.45)})
    sm.update(make_landmarks({"index": 0.20}, scale=scale))
    events = sm.update(make_landmarks({"index": 0.20}, scale=scale))
    assert _names(events) == {("attack", KEY_DOWN)}


def test_unsupported_finger_raises() -> None:
    with pytest.raises(ValueError):
        PinchStateMachine("left", {"inventory": GestureSpec("fist", 0.20, 0.35)})


def test_reset_releases_held_gestures(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    sm = PinchStateMachine("left", LEFT_GESTURES)
    sm.update(make_landmarks({"index": 0.20, "middle": 0.20}))
    sm.update(make_landmarks({"index": 0.20, "middle": 0.20}))
    released = sm.reset()
    assert _names(released) == {("jump", KEY_UP), ("sneak", KEY_UP)}
    assert sm.held == set()
    assert sm.reset() == []  # idempotent
