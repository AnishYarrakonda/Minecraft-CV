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

def test_wrist_tilt_joystick_anchors_first_frame(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)

    # Stabilize with a straight hand (wrist and MCP at x=0.5, y=0.5, 0.3)
    base_lm = make_screen_landmarks()
    base_lm[0, :2] = [0.5, 0.5]  # wrist
    base_lm[9, :2] = [0.5, 0.3]  # middle MCP (pointing up)
    for _ in range(5):
        res1 = pipe.step([make_hand_result(base_lm, "Right")])
    assert res1.wasd_held == frozenset()

    # Tilt wrist forward (W) by moving the wrist down relative to the MCP
    tilted_lm = base_lm.copy()
    tilted_lm[0, :2] = [0.5, 0.6]  # wrist moved down -> hand tilted more up
    tilted_lm[9, :2] = [0.5, 0.2]  # MCP moved up
    res2 = pipe.step([make_hand_result(tilted_lm, "Right")])
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

    sneak = make_screen_landmarks(distances={"pinky": 0.01})
    pipe.step([make_hand_result(sneak, "Right")])
    pipe.step([make_hand_result(sneak, "Right")])
    assert ("key_down", "shift") in null_emitter.log
    assert ("key_tap", "shift") not in null_emitter.log
    assert "shift" in null_emitter.held_keys

    released = make_screen_landmarks(distances={"pinky": 1.0})
    pipe.step([make_hand_result(released, "Right")])
    pipe.step([make_hand_result(released, "Right")])
    assert ("key_up", "shift") in null_emitter.log
    assert "shift" not in null_emitter.held_keys

def test_tracking_loss_releases_held_keys(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    # Stabilize
    base_lm = make_screen_landmarks()
    base_lm[0, :2] = [0.5, 0.5]
    base_lm[9, :2] = [0.5, 0.3]
    for _ in range(5):
        pipe.step([make_hand_result(base_lm, "Right")])
    
    # Tilt forward
    tilted_lm = base_lm.copy()
    tilted_lm[0, :2] = [0.5, 0.6]
    tilted_lm[9, :2] = [0.5, 0.2]
    pipe.step([make_hand_result(tilted_lm, "Right")])
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


def test_right_look_follows_index_mcp_not_palm_centroid(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    right_base = make_screen_landmarks(offset=(0.65, 0.25), thumb_ext=1.1)
    for _ in range(4):
        pipe.step([make_hand_result(right_base, "Left")])

    moved_cursor = right_base.copy()
    moved_cursor[5, :2] += np.array([0.14, 0.0], dtype=np.float32)
    res = pipe.step([make_hand_result(moved_cursor, "Left")])

    assert np.allclose(res.right_signal, moved_cursor[5, :2])
    assert res.right_output[0] > 0.0
    assert any(event[0] == "mouse_move" for event in null_emitter.log)


def test_right_index_mcp_has_no_deadzone(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    right_base = make_screen_landmarks(offset=(0.65, 0.25), thumb_ext=1.1)
    for _ in range(4):
        pipe.step([make_hand_result(right_base, "Left")])

    moved_cursor = right_base.copy()
    moved_cursor[5, :2] += np.array([0.005, 0.0], dtype=np.float32)
    res = pipe.step([make_hand_result(moved_cursor, "Left")])

    assert res.right_output[0] > 0.0
    assert any(event[0] == "mouse_move" for event in null_emitter.log)

def test_right_cursor_stable_during_pinch(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    right_base = make_screen_landmarks(offset=(0.65, 0.25), thumb_ext=1.1)
    for _ in range(4):
        pipe.step([make_hand_result(right_base, "Left")])

    # Simulate a pinch by moving the thumb tip (landmark 4) towards the index finger.
    # The index MCP (landmark 5) remains stationary.
    pinched = right_base.copy()
    pinched[4, :2] += np.array([0.15, -0.05], dtype=np.float32)
    res = pipe.step([make_hand_result(pinched, "Left")])

    # Because index MCP didn't move, the output should be zero.
    assert res.right_output[0] == 0.0
    assert res.right_output[1] == 0.0
    assert not any(event[0] == "mouse_move" for event in null_emitter.log)


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


def _one_euro_look_pipeline(emitter: NullEmitter) -> tuple[Pipeline, float]:
    """Pipeline with the One-Euro look filter active and a deterministic 30 FPS clock.

    Returns the pipeline and the per-frame ``dt`` so tests can reason about the filter.
    """
    settings = make_screen_settings(
        joystick={
            "deadzone": 0.08,
            "left_sensitivity": 5.0,
            "right_sensitivity": 5.0,
            "look_accel_exponent": 1.0,
            "smoothing": 0.0,
            "right_smoothing": 0.0,
            "look_filter": "one_euro",
        }
    )
    settings.joystick.fixed_left_neutral = None
    settings.joystick.fixed_right_neutral = None
    pipe = Pipeline.from_settings(settings, emitter=emitter)
    dt = 1.0 / 30.0
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += dt
        return state["t"]

    pipe._clock = clock
    return pipe, dt


def test_right_look_one_euro_smooths_spike_and_glides_past_frame(
    null_emitter: NullEmitter,
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """A single thumb jump must not be emitted as one raw delta then stop dead.

    With the One-Euro look filter active, the smoothed thumb point drives the camera:
    the spike frame is attenuated and the motion continues (glides) over subsequent
    frames while the hand holds still, instead of the sputtery one-frame-on/one-frame-off
    behaviour. Because successive filtered positions telescope, the total emitted motion
    still equals the thumb's true displacement (unity DC gain).
    """
    pipe, _dt = _one_euro_look_pipeline(null_emitter)
    sens = 5.0

    right_base = make_screen_landmarks(offset=(0.65, 0.25), thumb_ext=1.1)
    for _ in range(5):
        pipe.step([make_hand_result(right_base, "Left")])

    jump = right_base.copy()
    jump[5, :2] += np.array([0.20, 0.0], dtype=np.float32)
    raw_dx = 0.20 * sens  # what an unfiltered single-frame delta would emit

    spike = pipe.step([make_hand_result(jump, "Left")])
    spike_dx = float(spike.right_output[0])

    # The spike frame is smoothed, not a full raw jump.
    assert 0.0 < spike_dx < raw_dx * 0.5

    # Holding still, the look keeps gliding for several frames (motion extended past
    # the single jump frame) rather than stopping dead.
    total = spike_dx
    glide_frames = 0
    for _ in range(60):
        res = pipe.step([make_hand_result(jump, "Left")])
        dx = float(res.right_output[0])
        if dx > 1e-9:
            glide_frames += 1
        total += dx

    assert glide_frames >= 3
    # Total emitted look motion converges to the true thumb displacement.
    assert total == pytest.approx(raw_dx, rel=0.05)
