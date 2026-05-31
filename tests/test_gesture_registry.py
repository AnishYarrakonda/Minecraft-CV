"""Tests for the detector-backed gesture registry."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from minecraft_cv.config import GestureDetectorSettings, Settings
from minecraft_cv.gestures.registry import GestureStateMachine
from minecraft_cv.gestures.schmitt import KEY_DOWN, KEY_UP


def _names(events: list) -> set[tuple[str, str, str]]:
    return {(event.gesture, event.action, event.hand) for event in events}


def test_left_pinch_mapping_from_default_config(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    sm = GestureStateMachine("left", Settings().gestures.left_hand)
    lm = make_landmarks({"index": 0.20})
    sm.update(lm)
    events = sm.update(lm)
    assert ("jump", KEY_DOWN, "left") in _names(events)


def test_curl_only_requires_other_fingers_open(
    make_screen_landmarks: Callable[..., np.ndarray],
) -> None:
    gesture = GestureDetectorSettings(
        detector="curl_only",
        finger="pinky",
        t_engage=0.95,
        t_release=1.05,
        open_fingers=("thumb", "index", "middle", "ring"),
    )
    sm = GestureStateMachine("left", {"test": gesture})
    held = make_screen_landmarks(
        extensions={"index": 1.3, "middle": 1.3, "ring": 1.3, "pinky": 0.8},
        thumb_ext=1.3,
    )
    sm.update(held)
    events = sm.update(held)
    assert _names(events) == {("test", KEY_DOWN, "left")}

    not_open = make_screen_landmarks(
        extensions={"index": 0.8, "middle": 1.3, "ring": 1.3, "pinky": 0.8},
        thumb_ext=1.3,
    )
    released = sm.update(not_open)
    assert ("test", KEY_UP, "left") in _names(released)


def test_curl_combo_requires_all_listed_fingers_down_but_ignores_others(
    make_screen_landmarks: Callable[..., np.ndarray],
) -> None:
    sm = GestureStateMachine("right", {"sprint": Settings().gestures.right_hand["sprint"]})
    peace_with_thumb_relaxed = make_screen_landmarks(
        extensions={"index": 0.8, "middle": 1.3, "ring": 0.8, "pinky": 0.8},
        thumb_ext=0.4,
    )
    sm.update(peace_with_thumb_relaxed)
    events = sm.update(peace_with_thumb_relaxed)
    assert _names(events) == {("sprint", KEY_DOWN, "right")}

    ring_up = make_screen_landmarks(
        extensions={"index": 0.8, "middle": 1.3, "ring": 1.3, "pinky": 0.8},
        thumb_ext=0.4,
    )
    sm.update(ring_up)
    released = sm.update(ring_up)
    assert _names(released) == {("sprint", KEY_UP, "right")}


def test_left_sneak_holds_only_while_ring_pinky_curled(
    make_screen_landmarks: Callable[..., np.ndarray],
) -> None:
    sm = GestureStateMachine("left", Settings().gestures.left_hand)
    sneak = make_screen_landmarks(extensions={"ring": 0.8, "pinky": 0.8})
    sm.update(sneak)
    events = sm.update(sneak)
    assert _names(events) == {("sneak", KEY_DOWN, "left"), ("recenter", KEY_DOWN, "left")}
    assert sm.held == {"sneak", "recenter"}

    open_hand = make_screen_landmarks()
    sm.update(open_hand)
    events = sm.update(open_hand)
    assert _names(events) == {("sneak", KEY_UP, "left"), ("recenter", KEY_UP, "left")}
    assert sm.held == set()


def test_modifier_suppresses_same_finger_pinches(
    make_screen_landmarks: Callable[..., np.ndarray],
) -> None:
    sm = GestureStateMachine("left", Settings().gestures.left_hand)
    sneak_and_pinches = make_screen_landmarks(extensions={"ring": 0.8, "pinky": 0.8})
    scale = float(np.linalg.norm(sneak_and_pinches[9] - sneak_and_pinches[0]))
    sneak_and_pinches[4] = sneak_and_pinches[16] + np.array([0.20 * scale, 0.0, 0.0])
    sm.update(sneak_and_pinches)
    events = sm.update(sneak_and_pinches)
    names = _names(events)
    assert ("sneak", KEY_DOWN, "left") in names
    assert ("throw_item", KEY_DOWN, "left") not in names


def test_config_detector_swap_changes_behavior(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    gestures = {
        "jump": GestureDetectorSettings(
            detector="pinch", finger="middle", t_engage=0.30, t_release=0.45
        )
    }
    sm = GestureStateMachine("left", gestures)
    index = make_landmarks({"index": 0.20, "middle": 1.0})
    middle = make_landmarks({"middle": 0.20})
    sm.update(index)
    assert sm.update(index) == []
    sm.update(middle)
    assert _names(sm.update(middle)) == {("jump", KEY_DOWN, "left")}


def test_conflict_group_keeps_strongest_pinch(
    make_landmarks: Callable[..., np.ndarray],
) -> None:
    sm = GestureStateMachine("right", Settings().gestures.right_hand)
    both = make_landmarks({"index": 0.25, "middle": 0.20})
    sm.update(both)
    events = sm.update(both)
    names = _names(events)
    assert ("use", KEY_DOWN, "right") in names
    assert ("attack", KEY_DOWN, "right") not in names
    assert sm.held == {"use"}
