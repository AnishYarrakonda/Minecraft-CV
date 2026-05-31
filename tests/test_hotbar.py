"""Hotbar scroll tests."""
from __future__ import annotations
import pytest
import numpy as np
from typing import Callable

from conftest import make_screen_settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult

def test_ring_pinch_scrolls_up(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = Pipeline.from_settings(make_screen_settings(), emitter=null_emitter)
    for _ in range(5):
        pipe.step([make_hand_result(make_screen_landmarks(), "Left")])
    lm = make_screen_landmarks(distances={"pinky": 0.01})
    pipe.step([make_hand_result(lm, "Left")])
    pipe.step([make_hand_result(lm, "Left")])
    assert any(e[0] == "scroll" for e in null_emitter.log)
