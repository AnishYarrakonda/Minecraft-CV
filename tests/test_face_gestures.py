"""Tests for face gestures."""

from minecraft_cv.config import FaceGestureDetectorSettings
from minecraft_cv.gestures.face_gestures import FaceGestureStateMachine
from minecraft_cv.tracking.face_tracker import FaceResult
from minecraft_cv.gestures.registry import KEY_DOWN, KEY_UP

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
