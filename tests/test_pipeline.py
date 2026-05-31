"""Integration tests for the Pipeline with ScreenJoystick."""
from __future__ import annotations
import pytest
import numpy as np
from typing import Callable

from conftest import make_screen_settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult

def _pipeline(emitter: NullEmitter) -> Pipeline:
    settings = make_screen_settings()
    settings.joystick.fixed_left_neutral = None
    settings.joystick.fixed_right_neutral = None
    return Pipeline.from_settings(settings, emitter=emitter)

def test_screen_joystick_anchors_first_frame(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    
    # Stabilize
    for _ in range(5):
        lm = make_screen_landmarks(offset=(0.3, 0.3))
        res1 = pipe.step([make_hand_result(lm, "Right")])
    assert res1.wasd_held == frozenset()
    
    # Moving from that neutral triggers WASD.
    lm_moved = make_screen_landmarks(offset=(0.3, 0.1))  # Moved up/forward (-y)
    res2 = pipe.step([make_hand_result(lm_moved, "Right")])
    assert "w" in res2.wasd_held

def test_left_pinch_jump(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    # Stabilize
    for _ in range(5):
        pipe.step([make_hand_result(make_screen_landmarks(), "Right")])
    # Pinch frames (needs 2 consecutive to engage)
    lm = make_screen_landmarks(distances={"index": 0.01})
    pipe.step([make_hand_result(lm, "Right")])
    pipe.step([make_hand_result(lm, "Right")])
    assert ("key_down", "space") in null_emitter.log

def test_tracking_loss_releases_held_keys(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    # Stabilize
    for _ in range(5):
        pipe.step([make_hand_result(make_screen_landmarks(), "Right")])
    # Move forward
    pipe.step([make_hand_result(make_screen_landmarks(offset=(0.0, -0.5)), "Right")])
    assert "w" in null_emitter.held_keys
    
    # Tracking drop for 10 frames > grace period (3)
    for _ in range(10):
        pipe.step([])
        
    assert "w" not in null_emitter.held_keys
