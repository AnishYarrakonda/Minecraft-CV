"""Integration tests: synthetic tracker results -> Pipeline -> NullEmitter.

No camera, no MediaPipe, no OS input. Drives the pure ``Pipeline.step`` with synthetic
HandResults and asserts on the recorded NullEmitter log. Also exercises the threaded
FrameBuffer with a fake source (no OpenCV required).

Key conventions (post-redesign):
  - ``swap_handedness=True`` (default): a HandResult with ``handedness="Right"`` is the
    user's **physical left** hand (drives WASD + extension gestures). ``"Left"`` is the
    user's **physical right** hand (drives mouse look + pinch gestures).
  - Left-hand gestures are extension-based (closed fist → extend fingers).
  - WASD uses angular cardinal zones (``cardinal_half_width``).
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


# ---------------------------------------------------------------------------
# Basic pipeline wiring
# ---------------------------------------------------------------------------


def test_default_pipeline_uses_null_emitter() -> None:
    pipe = Pipeline.from_settings(Settings())
    assert isinstance(pipe.emitter, NullEmitter)


# ---------------------------------------------------------------------------
# Left hand — extension gestures (pass handedness="Right" → swapped to left)
# ---------------------------------------------------------------------------


def test_left_thumb_extension_emits_jump(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Extending the thumb on the left hand fires jump (space)."""
    pipe = _pipeline(null_emitter)
    # handedness="Right" is swapped to left by swap_handedness=True
    left = make_hand_result(make_extended_landmarks(thumb_ext=1.5), "Right")
    pipe.step([left])
    result = pipe.step([left])
    assert ("jump", "KEY_DOWN", "left") in [(e.gesture, e.action, e.hand) for e in result.events]
    assert ("key_down", "space") in null_emitter.log


# ---------------------------------------------------------------------------
# Right hand — pinch gestures (pass handedness="Left" → swapped to right)
# ---------------------------------------------------------------------------


def test_right_index_pinch_emits_attack_mouse(
    null_emitter: NullEmitter,
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Index pinch on right hand fires attack (mouse_left)."""
    pipe = _pipeline(null_emitter)
    # handedness="Left" is swapped to right by swap_handedness=True
    right = make_hand_result(make_landmarks({"index": 0.20}), "Left")
    pipe.step([right])
    pipe.step([right])
    assert ("key_down", "mouse_left") in null_emitter.log


# ---------------------------------------------------------------------------
# Spatial joystick — WASD (left hand) and mouse look (right hand)
# ---------------------------------------------------------------------------


def test_left_translation_presses_wasd(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Moving the left hand right of neutral → 'd' pressed."""
    pipe = _pipeline(null_emitter)
    # Frame 1 establishes neutral at the wrist offset; no movement keys yet.
    f1 = make_hand_result(
        make_extended_landmarks(offset=(0.5, 0.5, 0.0)), "Right"
    )
    assert pipe.step([f1]).wasd_held == frozenset()
    # Frame 2 moves the wrist right of neutral -> 'd' (right) held.
    f2 = make_hand_result(
        make_extended_landmarks(offset=(0.7, 0.5, 0.0)), "Right"
    )
    result = pipe.step([f2])
    assert result.wasd_held == frozenset({"d"})
    assert ("key_down", "d") in null_emitter.log


def test_right_translation_emits_mouse_move(
    null_emitter: NullEmitter,
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Moving the right hand emits mouse_move events."""
    pipe = _pipeline(null_emitter)
    # handedness="Left" → swapped to right
    pipe.step([make_hand_result(make_landmarks({}, offset=(0.5, 0.5, 0.0)), "Left")])
    pipe.step([make_hand_result(make_landmarks({}, offset=(0.7, 0.5, 0.0)), "Left")])
    assert any(entry[0] == "mouse_move" for entry in null_emitter.log)


# ---------------------------------------------------------------------------
# Cardinal zone tests
# ---------------------------------------------------------------------------


def test_cardinal_zone_pure_forward(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Moving hand straight up (y < 0, x ≈ 0) → only 'w' (forward)."""
    pipe = _pipeline(null_emitter)
    neutral = make_hand_result(
        make_extended_landmarks(offset=(0.5, 0.5, 0.0)), "Right"
    )
    pipe.step([neutral])
    # Move straight up: y decreases, x stays the same
    up = make_hand_result(
        make_extended_landmarks(offset=(0.5, 0.3, 0.0)), "Right"
    )
    result = pipe.step([up])
    assert "w" in result.wasd_held
    assert "d" not in result.wasd_held
    assert "a" not in result.wasd_held
    assert "s" not in result.wasd_held


def test_cardinal_zone_pure_right(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Moving hand right (x > 0, y ≈ 0) → only 'd'."""
    pipe = _pipeline(null_emitter)
    neutral = make_hand_result(
        make_extended_landmarks(offset=(0.5, 0.5, 0.0)), "Right"
    )
    pipe.step([neutral])
    right = make_hand_result(
        make_extended_landmarks(offset=(0.7, 0.5, 0.0)), "Right"
    )
    result = pipe.step([right])
    assert "d" in result.wasd_held
    assert "w" not in result.wasd_held
    assert "s" not in result.wasd_held
    assert "a" not in result.wasd_held


def test_cardinal_zone_diagonal(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Moving hand at ~45° up-right → both 'w' + 'd' (diagonal)."""
    pipe = _pipeline(null_emitter)
    neutral = make_hand_result(
        make_extended_landmarks(offset=(0.5, 0.5, 0.0)), "Right"
    )
    pipe.step([neutral])
    # Diagonal: move both right (+x) and up (-y) by equal amounts
    diag = make_hand_result(
        make_extended_landmarks(offset=(0.7, 0.3, 0.0)), "Right"
    )
    result = pipe.step([diag])
    assert "w" in result.wasd_held
    assert "d" in result.wasd_held


# ---------------------------------------------------------------------------
# Pulse gestures
# ---------------------------------------------------------------------------


def test_pulse_gesture_emits_key_tap(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Peace sign (index + middle extended) fires inventory as a key_tap."""
    pipe = _pipeline(null_emitter)
    lm = make_extended_landmarks({"index": 1.3, "middle": 1.3})
    left = make_hand_result(lm, "Right")  # swapped to left
    pipe.step([left])
    pipe.step([left])
    assert ("key_tap", "e") in null_emitter.log


# ---------------------------------------------------------------------------
# Both hands concurrent
# ---------------------------------------------------------------------------


def test_both_hands_concurrent_independent(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Left thumb extension (jump) + right index pinch (attack) fire independently."""
    pipe = _pipeline(null_emitter)
    left = make_hand_result(make_extended_landmarks(thumb_ext=1.5), "Right")  # → left
    right = make_hand_result(make_landmarks({"index": 0.20}), "Left")  # → right
    pipe.step([left, right])
    pipe.step([left, right])
    assert ("key_down", "space") in null_emitter.log
    assert ("key_down", "mouse_left") in null_emitter.log


# ---------------------------------------------------------------------------
# Tracking loss and shutdown safety
# ---------------------------------------------------------------------------


def test_tracking_loss_releases_held_keys(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """When hand tracking is lost, held keys are released."""
    pipe = _pipeline(null_emitter)
    # Hold jump via thumb extension
    pipe.step([make_hand_result(make_extended_landmarks(thumb_ext=1.5), "Right")])
    pipe.step([make_hand_result(make_extended_landmarks(thumb_ext=1.5), "Right")])
    pipe.step([])  # both hands gone -> jump released
    assert ("key_up", "space") in null_emitter.log
    assert null_emitter.held_keys == frozenset()


def test_shutdown_releases_everything(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """Pipeline shutdown releases all held keys."""
    pipe = _pipeline(null_emitter)
    pipe.step([make_hand_result(make_extended_landmarks(thumb_ext=1.5), "Right")])
    pipe.step([make_hand_result(make_extended_landmarks(thumb_ext=1.5), "Right")])
    assert "space" in null_emitter.held_keys
    pipe.shutdown()
    assert null_emitter.held_keys == frozenset()
    assert ("key_up", "space") in null_emitter.log


# ---------------------------------------------------------------------------
# Tracking blip resilience
# ---------------------------------------------------------------------------


def test_brief_tracking_blip_preserves_neutral(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    """A single dropped frame preserves the neutral so movement resumes correctly."""
    pipe = _pipeline(null_emitter)
    # Use handedness="Right" (swapped to left)
    pipe.step([make_hand_result(
        make_extended_landmarks(offset=(0.5, 0.5, 0.0)), "Right"
    )])  # neutral
    assert pipe.step(
        [make_hand_result(make_extended_landmarks(offset=(0.7, 0.5, 0.0)), "Right")]
    ).wasd_held == frozenset({"d"})
    pipe.step([])  # single dropped frame -> keys released, but neutral preserved
    assert null_emitter.held_keys == frozenset()
    # Hand returns to the same displaced spot: still right of the *original* neutral -> 'd'.
    resumed = pipe.step([make_hand_result(
        make_extended_landmarks(offset=(0.7, 0.5, 0.0)), "Right"
    )])
    assert resumed.wasd_held == frozenset({"d"})


# ---------------------------------------------------------------------------
# Input / emitter configuration
# ---------------------------------------------------------------------------


def test_disabled_input_never_creates_mac_emitter() -> None:
    """The default config has input.enabled False -> NullEmitter, never MacInputEmitter."""
    pipe = Pipeline.from_settings(Settings())
    assert type(pipe.emitter).__name__ == "NullEmitter"


# ---------------------------------------------------------------------------
# Threaded frame buffer
# ---------------------------------------------------------------------------


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


class _ErrorFakeSource(FrameSource):
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._i = 0
        self.released = False

    def read(self) -> np.ndarray | None:
        if self._i == 2:
            raise OSError("Transient read error")
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


def test_pipeline_shutdown_on_camera_error() -> None:
    from minecraft_cv.pipeline import run_pipeline
    from unittest.mock import patch
    import pytest

    frames = [np.full((4, 4, 3), v, dtype=np.uint8) for v in (10, 20)]
    source = _ErrorFakeSource(frames)

    settings = Settings()
    settings.input.enabled = False

    with patch("minecraft_cv.pipeline.Pipeline.shutdown") as mock_shutdown:
        with pytest.raises(OSError, match="Transient read error"):
            run_pipeline(settings, source=source)
        
        mock_shutdown.assert_called_once()
        assert source.released is True
