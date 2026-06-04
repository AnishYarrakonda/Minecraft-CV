"""Inventory is not a modal controller state."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from conftest import make_screen_settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult


def _pipeline(emitter: NullEmitter) -> Pipeline:
    settings = make_screen_settings()
    settings.joystick.fixed_left_neutral = None
    settings.joystick.fixed_right_neutral = None
    return Pipeline.from_settings(settings, emitter=emitter)


def test_open_palms_do_not_pause_wasd_or_right_look(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    left_neutral = make_screen_landmarks(offset=(0.20, 0.20))
    right_neutral = make_screen_landmarks(offset=(0.65, 0.20))

    for _ in range(10):
        pipe.step(
            [
                make_hand_result(left_neutral, "Right"),
                make_hand_result(right_neutral, "Left"),
            ]
        )

    left_forward = make_screen_landmarks()
    left_forward[0, :2] = [0.20, 0.20]  # wrist
    left_forward[9, :2] = [0.20, -0.20] # MCP (tilted forward -> W)
    right_look = make_screen_landmarks(offset=(0.85, 0.20))
    result = pipe.step(
        [
            make_hand_result(left_forward, "Right"),
            make_hand_result(right_look, "Left"),
        ]
    )

    assert "w" in result.wasd_held
    assert any(event[0] == "mouse_move" for event in null_emitter.log)
    assert not any(event[0] == "mouse_move_abs" for event in null_emitter.log)
