"""Tests for face gestures."""

import math

import numpy as np

from minecraft_cv.config import FaceGestureDetectorSettings, HeadRollGestureSettings, Settings
from minecraft_cv.gestures.face_gestures import (
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    FaceGestureStateMachine,
    HeadRollDetector,
)
from minecraft_cv.gestures.registry import KEY_DOWN, KEY_UP
from minecraft_cv.tracking.face_tracker import FaceResult


def _face_roll(deg: float) -> FaceResult:
    """FaceResult whose eye-corner line is rolled ``deg`` degrees in the image plane."""
    lm = np.full((478, 3), 0.5, dtype=np.float32)
    rad = math.radians(deg)
    left = np.array([0.45, 0.5, 0.0], dtype=np.float32)
    right = left + np.array([math.cos(rad) * 0.1, math.sin(rad) * 0.1, 0.0], dtype=np.float32)
    lm[LEFT_EYE_OUTER] = left
    lm[RIGHT_EYE_OUTER] = right
    return FaceResult(blendshapes={}, landmarks=lm)

def test_face_gesture_schmitt_trigger() -> None:
    settings = {
        "inventory": FaceGestureDetectorSettings(
            blendshape="browInnerUp",
            t_engage=0.5,
            t_release=0.3,
            engage_frames=3,
            release_frames=2,
        )
    }
    sm = FaceGestureStateMachine(settings)

    # Below threshold -> no events
    events = sm.update(FaceResult({"browInnerUp": 0.2}))
    assert not events

    # Above engage threshold, but need 3 frames
    events = sm.update(FaceResult({"browInnerUp": 0.6}))
    assert not events
    events = sm.update(FaceResult({"browInnerUp": 0.7}))
    assert not events

    # 3rd frame -> KEY_DOWN
    events = sm.update(FaceResult({"browInnerUp": 0.8}))
    assert len(events) == 1
    assert events[0].gesture == "inventory"
    assert events[0].action == KEY_DOWN

    # Active -> stay active even if it drops a bit
    events = sm.update(FaceResult({"browInnerUp": 0.4}))
    assert not events

    # Below release threshold -> need 2 frames
    events = sm.update(FaceResult({"browInnerUp": 0.2}))
    assert not events

    # 2nd frame -> KEY_UP
    events = sm.update(FaceResult({"browInnerUp": 0.1}))
    assert len(events) == 1
    assert events[0].gesture == "inventory"
    assert events[0].action == KEY_UP

def test_face_gesture_reset() -> None:
    settings = {
        "inventory": FaceGestureDetectorSettings(
            blendshape="browInnerUp",
            t_engage=0.5,
            t_release=0.3,
            engage_frames=1,
            release_frames=1,
        )
    }
    sm = FaceGestureStateMachine(settings)
    sm.update(FaceResult({"browInnerUp": 0.8}))

    events = sm.reset()
    assert len(events) == 1
    assert events[0].gesture == "inventory"
    assert events[0].action == KEY_UP


def test_swap_offhand_default_uses_cheek_puff() -> None:
    """F (swap offhand) is now triggered by cheekPuff, not eyeBlinkLeft."""
    face = Settings().gestures.face
    assert face["swap_offhand"].blendshape == "cheekPuff"


def _names(events: list) -> set:
    return {(e.gesture, e.action) for e in events}


def test_head_roll_engages_and_releases_with_hysteresis() -> None:
    det = HeadRollDetector(
        left_gesture="hotbar_next",
        right_gesture="hotbar_prev",
        engage_deg=12.0,
        release_deg=7.0,
        engage_frames=2,
        release_frames=2,
    )
    # Below engage -> nothing.
    assert det.update(_face_roll(5.0)) == []
    # Past engage, needs 2 frames.
    assert det.update(_face_roll(20.0)) == []
    assert _names(det.update(_face_roll(20.0))) == {("hotbar_next", KEY_DOWN)}
    # Inside the release band but not past it: still held (hysteresis).
    assert det.update(_face_roll(10.0)) == []
    # Back inside release band for 2 frames -> release.
    assert det.update(_face_roll(3.0)) == []
    assert _names(det.update(_face_roll(3.0))) == {("hotbar_next", KEY_UP)}


def _snappy_head_roll() -> HeadRollDetector:
    return HeadRollDetector(
        "hotbar_next", "hotbar_prev", 12.0, 7.0, engage_frames=1, release_frames=1
    )


def test_head_roll_opposite_direction_fires_right_gesture() -> None:
    det = _snappy_head_roll()
    assert _names(det.update(_face_roll(-20.0))) == {("hotbar_prev", KEY_DOWN)}


def test_head_roll_directions_are_mutually_exclusive() -> None:
    det = _snappy_head_roll()
    assert _names(det.update(_face_roll(20.0))) == {("hotbar_next", KEY_DOWN)}
    # Swinging to the other side releases the first before engaging the second.
    events = _names(det.update(_face_roll(-20.0)))
    assert ("hotbar_next", KEY_UP) in events
    assert det._active is None  # released; re-engage happens on subsequent frames


def test_head_roll_releases_on_missing_landmarks() -> None:
    det = _snappy_head_roll()
    det.update(_face_roll(20.0))
    assert _names(det.update(FaceResult())) == {("hotbar_next", KEY_UP)}


def test_state_machine_runs_head_roll_from_default_config() -> None:
    head = Settings().gestures.head_tilt
    assert isinstance(head, HeadRollGestureSettings)
    sm = FaceGestureStateMachine(Settings().gestures.face, head_roll=head)
    sm.update(_face_roll(20.0))
    events = _names(sm.update(_face_roll(20.0)))
    assert ("hotbar_next", KEY_DOWN) in events
