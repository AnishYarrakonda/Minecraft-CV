"""Tests for the shared FrameProcessor and the new Pipeline recenter/emitter-swap helpers.

The FrameProcessor packet-flow test uses an injected fake source + fake tracker and a
NullEmitter, so it runs with no camera and emits no OS input (hard invariant #2). The
recenter/set_emitter tests are pure pipeline logic and need neither OpenCV nor a camera.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from conftest import make_screen_settings
from minecraft_cv.capture.source import FrameSource
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline, StepResult
from minecraft_cv.runtime import FrameProcessor
from minecraft_cv.tracking.tracker import HandResult, HandTracker


class _FakeSource(FrameSource):
    """A frame source that returns a fresh small BGR frame on each read."""

    def __init__(self) -> None:
        self._n = 0

    def read(self) -> np.ndarray | None:
        time.sleep(0.001)  # keep the buffer thread from spinning at 100% CPU
        self._n += 1
        base = np.full((48, 64, 3), self._n % 251 + 3, dtype=np.uint8)
        return base

    def release(self) -> None:
        return None

    @property
    def fps(self) -> float:
        return 30.0


class _FakeTracker(HandTracker):
    """A tracker that detects no hands (enough to exercise the loop plumbing)."""

    def detect(self, rgb_frame: np.ndarray) -> list[HandResult]:
        return []


def _screen_pipeline(emitter: NullEmitter) -> Pipeline:
    settings = make_screen_settings()
    settings.joystick.fixed_left_neutral = None
    settings.joystick.fixed_right_neutral = None
    return Pipeline.from_settings(settings, emitter=emitter)


def test_frame_processor_emits_packets_in_dry_run() -> None:
    pytest.importorskip("cv2")
    settings = make_screen_settings()
    emitter = NullEmitter()
    pipeline = Pipeline.from_settings(settings, emitter=emitter)
    proc = FrameProcessor(pipeline, _FakeSource(), _FakeTracker(), settings).start()
    try:
        packet = None
        for _ in range(300):
            packet = proc.process_once()
            if packet is not None:
                break
            time.sleep(0.005)
        assert packet is not None
        assert isinstance(packet.step, StepResult)
        assert packet.frame.shape == (48, 64, 3)  # display frame kept at source resolution
        assert packet.live is False
        # No real key/mouse-move input in dry-run (mouse_stop safety calls are allowed).
        assert all(entry[0] not in ("key_down", "key_up", "mouse_move") for entry in emitter.log)
    finally:
        proc.shutdown()


def test_recenter_clears_joystick_neutrals(
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _screen_pipeline(NullEmitter())
    for _ in range(5):
        pipe.step([make_hand_result(make_screen_landmarks(offset=(0.3, 0.3)), "Right")])
    assert pipe.left_joystick.neutral is not None

    pipe.recenter()
    assert pipe.left_joystick.neutral is None
    assert pipe.right_joystick.neutral is None


def test_set_emitter_releases_held_keys_on_old_emitter(
    make_screen_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    old = NullEmitter()
    pipe = _screen_pipeline(old)
    for _ in range(5):
        pipe.step([make_hand_result(make_screen_landmarks(), "Right")])
    pinch = make_screen_landmarks(distances={"index": 0.01})
    pipe.step([make_hand_result(pinch, "Right")])
    pipe.step([make_hand_result(pinch, "Right")])
    assert "d" in old.held_keys  # left index pinch -> move_right -> "d"

    new = NullEmitter()
    pipe.set_emitter(new)

    assert old.held_keys == frozenset()  # handoff released everything
    assert pipe.emitter is new
