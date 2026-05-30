"""Integration tests: synthetic tracker results -> Pipeline -> NullEmitter.

No camera, no MediaPipe, no OS input. Drives the pure ``Pipeline.step`` with synthetic
HandResults and asserts on the recorded NullEmitter log. Also exercises the threaded
FrameBuffer with a fake source (no OpenCV required).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from minecraft_cv.capture.buffer import FrameBuffer
from minecraft_cv.capture.source import FrameSource
from minecraft_cv.config import Settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult


def _pipeline(emitter: NullEmitter) -> Pipeline:
    return Pipeline.from_settings(Settings(), emitter=emitter)


def test_default_pipeline_uses_null_emitter() -> None:
    pipe = Pipeline.from_settings(Settings())
    assert isinstance(pipe.emitter, NullEmitter)


def test_left_index_pinch_emits_jump(
    null_emitter: NullEmitter,
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    left = make_hand_result(make_landmarks({"index": 0.20}), "Left")
    result = pipe.step([left])
    assert ("jump", "KEY_DOWN", "left") in [(e.gesture, e.action, e.hand) for e in result.events]
    assert ("key_down", "space") in null_emitter.log


def test_right_index_pinch_emits_attack_mouse(
    null_emitter: NullEmitter,
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    right = make_hand_result(make_landmarks({"index": 0.20}), "Right")
    pipe.step([right])
    assert ("key_down", "mouse_left") in null_emitter.log


def test_left_translation_presses_wasd(
    null_emitter: NullEmitter,
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    # Frame 1 establishes neutral at the wrist offset; no movement keys yet.
    f1 = make_hand_result(make_landmarks({}, offset=(0.5, 0.5, 0.0)), "Left")
    assert pipe.step([f1]).wasd_held == frozenset()
    # Frame 2 moves the wrist right of neutral -> 'd' (right) held.
    f2 = make_hand_result(make_landmarks({}, offset=(0.7, 0.5, 0.0)), "Left")
    result = pipe.step([f2])
    assert result.wasd_held == frozenset({"d"})
    assert ("key_down", "d") in null_emitter.log


def test_right_translation_emits_mouse_move(
    null_emitter: NullEmitter,
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    pipe.step([make_hand_result(make_landmarks({}, offset=(0.5, 0.5, 0.0)), "Right")])
    pipe.step([make_hand_result(make_landmarks({}, offset=(0.7, 0.5, 0.0)), "Right")])
    assert any(entry[0] == "mouse_move" for entry in null_emitter.log)


def test_both_hands_concurrent_independent(
    null_emitter: NullEmitter,
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    left = make_hand_result(make_landmarks({"index": 0.20}), "Left")  # jump
    right = make_hand_result(make_landmarks({"index": 0.20}), "Right")  # attack
    pipe.step([left, right])
    assert ("key_down", "space") in null_emitter.log
    assert ("key_down", "mouse_left") in null_emitter.log


def test_tracking_loss_releases_held_keys(
    null_emitter: NullEmitter,
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    pipe.step([make_hand_result(make_landmarks({"index": 0.20}), "Left")])  # hold jump
    pipe.step([])  # both hands gone -> jump released
    assert ("key_up", "space") in null_emitter.log
    assert null_emitter.held_keys == frozenset()


def test_shutdown_releases_everything(
    null_emitter: NullEmitter,
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _pipeline(null_emitter)
    pipe.step([make_hand_result(make_landmarks({"index": 0.20}), "Left")])
    assert "space" in null_emitter.held_keys
    pipe.shutdown()
    assert null_emitter.held_keys == frozenset()
    assert ("key_up", "space") in null_emitter.log


def test_disabled_input_never_creates_mac_emitter() -> None:
    # The default config has input.enabled False -> NullEmitter, never MacInputEmitter.
    pipe = Pipeline.from_settings(Settings())
    assert type(pipe.emitter).__name__ == "NullEmitter"


# --- threaded frame buffer ----------------------------------------------------
class _FakeSource(FrameSource):
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._i = 0
        self.released = False

    def read(self) -> np.ndarray | None:
        if self._i >= len(self._frames):
            return None
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def release(self) -> None:
        self.released = True

    @property
    def fps(self) -> float:
        return 30.0


def test_frame_buffer_keeps_latest_and_exhausts() -> None:
    frames = [np.full((4, 4, 3), v, dtype=np.uint8) for v in (10, 20, 30)]
    source = _FakeSource(frames)
    buf = FrameBuffer(source).start()
    deadline = time.time() + 2.0
    while not buf.exhausted and time.time() < deadline:
        time.sleep(0.01)
    assert buf.exhausted
    seq, latest = buf.latest()
    assert seq == 3
    assert latest is not None and int(latest[0, 0, 0]) == 30  # newest frame retained
    buf.stop()
    assert source.released is True
