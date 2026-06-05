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
    assert ("right", KEY_DOWN, "left") in _names(events)


def test_curl_only_requires_other_fingers_open(
    make_wrist_rotation_landmarks: Callable[..., np.ndarray],
) -> None:
    gesture = GestureDetectorSettings(
        detector="curl_only",
        finger="pinky",
        t_engage=0.95,
        t_release=1.05,
        open_fingers=("thumb", "index", "middle", "ring"),
    )
    sm = GestureStateMachine("left", {"test": gesture})
    held = make_wrist_rotation_landmarks(
        extensions={"index": 1.3, "middle": 1.3, "ring": 1.3, "pinky": 0.8},
        thumb_ext=1.3,
    )
    sm.update(held)
    events = sm.update(held)
    assert _names(events) == {("test", KEY_DOWN, "left")}

    not_open = make_wrist_rotation_landmarks(
        extensions={"index": 0.8, "middle": 1.3, "ring": 1.3, "pinky": 0.8},
        thumb_ext=1.3,
    )
    released = sm.update(not_open)
    assert ("test", KEY_UP, "left") in _names(released)


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
    from minecraft_cv.gestures.registry import GestureStateMachine
    from minecraft_cv.config import GestureDetectorSettings

    spec = {
        "attack": GestureDetectorSettings(
            detector="pinch", finger="index", t_engage=0.3, t_release=0.4, conflict_group="click"
        ),
        "use": GestureDetectorSettings(
            detector="pinch", finger="middle", t_engage=0.3, t_release=0.4, conflict_group="click"
        ),
    }
    sm = GestureStateMachine("right", spec)
    # Give both pinches, make index stronger (smaller distance)
    lm = make_landmarks({"index": 0.15, "middle": 0.25})
    sm.update(lm)
    sm.update(lm)
    assert sm.held == {"attack"}
