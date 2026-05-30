from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from minecraft_cv.config import ExtensionThresholds
from minecraft_cv.gestures.extension import ExtensionStateMachine
from minecraft_cv.gestures.schmitt import KEY_DOWN, KEY_UP


@pytest.fixture
def gestures() -> dict[str, ExtensionThresholds]:
    return {
        "jump": ExtensionThresholds(type="thumb_out", t_engage=1.2, t_release=0.9),
        "sneak": ExtensionThresholds(type="index_only", t_engage=1.15, t_release=1.05),
        "sprint": ExtensionThresholds(type="middle_only", t_engage=1.15, t_release=1.05),
        "inventory": ExtensionThresholds(type="index_middle", t_engage=1.15, t_release=1.05, pulse=True),
        "throw_item": ExtensionThresholds(type="ring_only", t_engage=1.15, t_release=1.05, pulse=True),
        "switch_offhand": ExtensionThresholds(type="pinky_only", t_engage=1.15, t_release=1.05, pulse=True),
    }


def test_thumb_out_fires_jump(
    gestures: dict[str, ExtensionThresholds],
    make_extended_landmarks: Callable[..., np.ndarray]
) -> None:
    sm = ExtensionStateMachine("left", gestures)
    
    # Curled thumb doesn't engage
    lm = make_extended_landmarks({}, thumb_ext=0.5)
    events = sm.update(lm)
    assert not events
    
    # Extend thumb past t_engage (1.2)
    lm = make_extended_landmarks({}, thumb_ext=1.5)
    events = sm.update(lm)
    assert len(events) == 1
    assert events[0].gesture == "jump"
    assert events[0].action == KEY_DOWN
    
    # Stay extended (no new events)
    events = sm.update(lm)
    assert not events
    
    # Drop below t_release (0.9)
    lm = make_extended_landmarks({}, thumb_ext=0.5)
    events = sm.update(lm)
    assert len(events) == 1
    assert events[0].gesture == "jump"
    assert events[0].action == KEY_UP


def test_index_only_fires_sneak(
    gestures: dict[str, ExtensionThresholds],
    make_extended_landmarks: Callable[..., np.ndarray]
) -> None:
    sm = ExtensionStateMachine("left", gestures)
    
    # Extend index past 1.15
    lm = make_extended_landmarks({"index": 1.3})
    events = sm.update(lm)
    assert len(events) == 1
    assert events[0].gesture == "sneak"
    assert events[0].action == KEY_DOWN


def test_middle_only_fires_sprint(
    gestures: dict[str, ExtensionThresholds],
    make_extended_landmarks: Callable[..., np.ndarray]
) -> None:
    sm = ExtensionStateMachine("left", gestures)
    
    lm = make_extended_landmarks({"middle": 1.3})
    events = sm.update(lm)
    assert len(events) == 1
    assert events[0].gesture == "sprint"
    assert events[0].action == KEY_DOWN


def test_peace_sign_fires_inventory(
    gestures: dict[str, ExtensionThresholds],
    make_extended_landmarks: Callable[..., np.ndarray]
) -> None:
    sm = ExtensionStateMachine("left", gestures)
    
    lm = make_extended_landmarks({"index": 1.3, "middle": 1.3})
    events = sm.update(lm)
    assert len(events) == 1
    assert events[0].gesture == "inventory"
    assert events[0].action == KEY_DOWN


def test_exclusion_prevents_false_positive(
    gestures: dict[str, ExtensionThresholds],
    make_extended_landmarks: Callable[..., np.ndarray]
) -> None:
    sm = ExtensionStateMachine("left", gestures)
    
    # Extend all fingers
    lm = make_extended_landmarks({
        "index": 1.3,
        "middle": 1.3,
        "ring": 1.3,
        "pinky": 1.3
    })
    events = sm.update(lm)
    # The thumb is curled, so thumb_out shouldn't fire.
    # Because all fingers are extended, ALL 'only' gestures and 'index_middle' are excluded.
    # No gestures should fire.
    assert not events


def test_pulse_gesture_is_marked(gestures: dict[str, ExtensionThresholds]) -> None:
    sm = ExtensionStateMachine("left", gestures)
    assert sm.pulse_gestures == frozenset({"inventory", "throw_item", "switch_offhand"})


def test_reset_releases_held(
    gestures: dict[str, ExtensionThresholds],
    make_extended_landmarks: Callable[..., np.ndarray]
) -> None:
    sm = ExtensionStateMachine("left", gestures)
    
    # Hold thumb_out
    lm = make_extended_landmarks({}, thumb_ext=1.5)
    sm.update(lm)
    assert "jump" in sm.held
    
    # Reset
    events = sm.reset()
    assert len(events) == 1
    assert events[0].gesture == "jump"
    assert events[0].action == KEY_UP
    assert not sm.held
