"""Inventory mode tests."""
from __future__ import annotations
import pytest
import numpy as np
from typing import Callable

from conftest import make_screen_settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult

def _pipeline(emitter: NullEmitter) -> Pipeline:
    settings = make_screen_settings(inventory={"enabled": True, "open_threshold": 0.20})
    return Pipeline.from_settings(settings, emitter=emitter)

def test_inventory_mode_pauses_wasd(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    # Stabilize AND toggle inventory mode (hold_frames=8)
    for _ in range(10):
        result = pipe.step([
            make_hand_result(make_screen_landmarks(), "Right"),
            make_hand_result(make_screen_landmarks(), "Left")
        ])
        
    assert result.inventory_active is True
    
    # Try to move forward -> WASD should be empty because we are in inventory mode
    lm_moved = make_screen_landmarks(offset=(0.0, -0.5))
    result = pipe.step([
        make_hand_result(lm_moved, "Right"),
        make_hand_result(make_screen_landmarks(), "Left")
    ])
    
    assert result.wasd_held == frozenset()
