"""Integration tests for the Pipeline with ScreenJoystick."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from conftest import make_screen_settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult


def _pipeline(emitter: NullEmitter) -> Pipeline:
    settings = make_screen_settings()
    settings.joystick.fixed_left_neutral = None
    settings.joystick.fixed_right_neutral = None
    return Pipeline.from_settings(settings, emitter=emitter)


def test_right_joystick_still_carries_right_specific_sensitivity(
    null_emitter: NullEmitter,
) -> None:
    settings = make_screen_settings(
        joystick={
            "smoothing": 0.1,
            "right_smoothing": 0.7,
            "fixed_left_neutral": None,
            "fixed_right_neutral": None,
        }
    )
    pipe = Pipeline.from_settings(settings, emitter=null_emitter)
    assert pipe.left_joystick.smoothing == pytest.approx(0.1)
    assert pipe.right_joystick.smoothing == pytest.approx(0.7)

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


def test_left_pinches_are_held_not_tapped(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    for _ in range(5):
        pipe.step([make_hand_result(make_screen_landmarks(), "Right")])

    inventory = make_screen_landmarks(distances={"middle": 0.01})
    pipe.step([make_hand_result(inventory, "Right")])
    pipe.step([make_hand_result(inventory, "Right")])
    assert ("key_down", "e") in null_emitter.log
    assert ("key_tap", "e") not in null_emitter.log
    assert "e" in null_emitter.held_keys

    released = make_screen_landmarks(distances={"middle": 1.0})
    pipe.step([make_hand_result(released, "Right")])
    pipe.step([make_hand_result(released, "Right")])
    assert ("key_up", "e") in null_emitter.log
    assert "e" not in null_emitter.held_keys

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


def test_left_peace_relocalizes_left_joystick_only(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    left_base = make_screen_landmarks(offset=(0.20, 0.20))
    right_base = make_screen_landmarks(offset=(0.65, 0.20))
    for _ in range(4):
        pipe.step([
            make_hand_result(left_base, "Right"),
            make_hand_result(right_base, "Left"),
        ])

    old_right = pipe.right_joystick.neutral.copy()
    peace = make_screen_landmarks(
        offset=(0.42, 0.34),
        extensions={"index": 1.3, "middle": 1.3, "ring": 0.8, "pinky": 0.8},
        thumb_ext=0.2,
    )
    pipe.step([make_hand_result(peace, "Right"), make_hand_result(right_base, "Left")])
    res = pipe.step([make_hand_result(peace, "Right"), make_hand_result(right_base, "Left")])

    assert res.relocalized_hands == frozenset({"left"})
    assert np.allclose(pipe.left_joystick.neutral, pipe.left_joystick_signal(peace))
    assert np.allclose(pipe.right_joystick.neutral, old_right)
    assert ("key_down", "shift") not in null_emitter.log


def test_right_peace_relocalizes_right_joystick_without_clicks(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    left_base = make_screen_landmarks(offset=(0.20, 0.20))
    right_base = make_screen_landmarks(offset=(0.65, 0.20))
    for _ in range(4):
        pipe.step([
            make_hand_result(left_base, "Right"),
            make_hand_result(right_base, "Left"),
        ])

    old_left = pipe.left_joystick.neutral.copy()
    peace = make_screen_landmarks(
        offset=(0.74, 0.36),
        extensions={"index": 1.3, "middle": 1.3, "ring": 0.8, "pinky": 0.8},
        thumb_ext=1.6,
    )
    pipe.step([make_hand_result(left_base, "Right"), make_hand_result(peace, "Left")])
    res = pipe.step([make_hand_result(left_base, "Right"), make_hand_result(peace, "Left")])

    assert res.relocalized_hands == frozenset({"right"})
    assert np.allclose(pipe.right_joystick.neutral, pipe.right_joystick_signal(peace))
    assert np.allclose(pipe.left_joystick.neutral, old_left)
    assert not any(event[1:] == ("mouse_left",) for event in null_emitter.log)
    assert not any(event[1:] == ("mouse_right",) for event in null_emitter.log)


def test_right_look_follows_thumb_tip_not_palm_centroid(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    right_base = make_screen_landmarks(offset=(0.65, 0.25), thumb_ext=1.1)
    for _ in range(4):
        pipe.step([make_hand_result(right_base, "Left")])

    moved_thumb = right_base.copy()
    moved_thumb[4, :2] += np.array([0.14, 0.0], dtype=np.float32)
    res = pipe.step([make_hand_result(moved_thumb, "Left")])

    assert np.allclose(res.right_signal, moved_thumb[4, :2])
    assert res.right_output[0] > 0.0
    assert any(event[0] == "mouse_move" for event in null_emitter.log)


def test_right_thumb_has_no_deadzone(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    right_base = make_screen_landmarks(offset=(0.65, 0.25), thumb_ext=1.1)
    for _ in range(4):
        pipe.step([make_hand_result(right_base, "Left")])

    moved_thumb = right_base.copy()
    moved_thumb[4, :2] += np.array([0.005, 0.0], dtype=np.float32)
    res = pipe.step([make_hand_result(moved_thumb, "Left")])

    assert res.right_output[0] > 0.0
    assert any(event[0] == "mouse_move" for event in null_emitter.log)


def test_right_peace_clutches_mouse_and_recenters_thumb_while_held(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    left_base = make_screen_landmarks(offset=(0.20, 0.20))
    right_base = make_screen_landmarks(offset=(0.65, 0.20), thumb_ext=1.1)
    for _ in range(4):
        pipe.step([
            make_hand_result(left_base, "Right"),
            make_hand_result(right_base, "Left"),
        ])

    peace_a = make_screen_landmarks(
        offset=(0.70, 0.25),
        extensions={"index": 1.3, "middle": 1.3, "ring": 0.8, "pinky": 0.8},
        thumb_ext=0.8,
    )
    peace_b = make_screen_landmarks(
        offset=(0.70, 0.25),
        extensions={"index": 1.3, "middle": 1.3, "ring": 0.8, "pinky": 0.8},
        thumb_ext=1.8,
    )
    pipe.step([make_hand_result(left_base, "Right"), make_hand_result(peace_a, "Left")])
    engaged = pipe.step([
        make_hand_result(left_base, "Right"),
        make_hand_result(peace_a, "Left"),
    ])
    assert engaged.relocalized_hands == frozenset({"right"})

    log_start = len(null_emitter.log)
    held = pipe.step([make_hand_result(left_base, "Right"), make_hand_result(peace_b, "Left")])

    assert np.allclose(held.right_output, np.zeros(2))
    assert np.allclose(pipe.right_joystick.neutral, pipe.right_joystick_signal(peace_b))
    assert not any(event[0] == "mouse_move" for event in null_emitter.log[log_start:])
