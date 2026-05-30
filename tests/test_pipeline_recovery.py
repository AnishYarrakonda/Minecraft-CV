"""Pipeline-level integration of Sprint-by-velocity (Task 2) and tracking-loss recovery
(Task 5), driven by an injected deterministic clock (no real time, no camera).

Convention reminder: a ``HandResult`` with ``handedness="Right"`` is the user's physical
**left** hand (swap_handedness default), which drives WASD + the velocity sprint.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from minecraft_cv.config import Settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult


class _Clock:
    """A settable monotonic clock for deterministic pipeline timing."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _pipeline(settings: Settings, emitter: NullEmitter, clock: _Clock) -> Pipeline:
    pipe = Pipeline.from_settings(settings, emitter=emitter)
    pipe._clock = clock  # type: ignore[assignment]
    return pipe


# ---------------------------------------------------------------------------
# Task 2 — Sprint via forward-push velocity
# ---------------------------------------------------------------------------


def test_velocity_sprint_engages_ctrl_and_forward(
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    settings = Settings(
        sprint={"enabled": True, "v_sprint": 1.0, "trigger_frames": 3, "release_margin": 0.02}
    )
    emitter = NullEmitter()
    clock = _Clock()
    pipe = _pipeline(settings, emitter, clock)

    def left_at(z: float) -> list[HandResult]:
        lm = make_extended_landmarks(offset=(0.5, 0.5, z))
        return [make_hand_result(lm, "Right")]  # swapped -> physical left

    clock.t = 0.0
    pipe.step(left_at(0.0))  # seed neutral (joystick + sprint depth)
    clock.t = 0.1
    pipe.step(left_at(-0.2))  # forward 2.0 u/s -> count 1
    clock.t = 0.2
    pipe.step(left_at(-0.4))  # count 2
    clock.t = 0.3
    result = pipe.step(left_at(-0.6))  # count 3 -> ENGAGE

    assert ("key_down", "ctrl") in emitter.log
    assert "w" in result.wasd_held  # Sprint forces forward
    assert pipe._sprint_active

    # Retreat back to neutral -> sprint releases Ctrl and drops forward.
    clock.t = 0.4
    result = pipe.step(left_at(0.0))
    assert ("key_up", "ctrl") in emitter.log
    assert "w" not in result.wasd_held
    assert not pipe._sprint_active


def test_velocity_sprint_disabled_by_default(
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    emitter = NullEmitter()
    clock = _Clock()
    pipe = _pipeline(Settings(), emitter, clock)
    for i, z in enumerate((0.0, -0.3, -0.6, -0.9)):
        clock.t = i * 0.1
        pipe.step([make_hand_result(make_extended_landmarks(offset=(0.5, 0.5, z)), "Right")])
    assert ("key_down", "ctrl") not in emitter.log


def test_velocity_sprint_releases_on_tracking_loss(
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    settings = Settings(sprint={"enabled": True, "v_sprint": 1.0, "trigger_frames": 3})
    emitter = NullEmitter()
    clock = _Clock()
    pipe = _pipeline(settings, emitter, clock)

    def left_at(z: float) -> list[HandResult]:
        return [make_hand_result(make_extended_landmarks(offset=(0.5, 0.5, z)), "Right")]

    for i, z in enumerate((0.0, -0.2, -0.4, -0.6)):
        clock.t = i * 0.1
        pipe.step(left_at(z))
    assert pipe._sprint_active
    # Hand vanishes -> Ctrl must be released (no sprint-lock).
    clock.t = 0.4
    pipe.step([])
    assert ("key_up", "ctrl") in emitter.log
    assert "ctrl" not in emitter.held_keys


# ---------------------------------------------------------------------------
# Task 5 — long dropout hard-flush + re-entry stabilization (no camera snap)
# ---------------------------------------------------------------------------


def test_long_dropout_then_stabilization_prevents_snap(
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    emitter = NullEmitter()
    clock = _Clock()
    pipe = _pipeline(Settings(), emitter, clock)  # defaults: flush 100 ms, stabilize 500 ms

    def left_at(x: float) -> list[HandResult]:
        return [make_hand_result(make_extended_landmarks(offset=(x, 0.5, 0.0)), "Right")]

    clock.t = 0.0
    pipe.step(left_at(0.5))  # neutral
    clock.t = 0.01
    assert pipe.step(left_at(0.7)).wasd_held == frozenset({"d"})  # moving right

    # Long dropout (>100 ms): keys release, neutral flushed.
    clock.t = 0.2
    assert pipe.step([]).wasd_held == frozenset()
    assert "d" not in emitter.held_keys

    # Re-entry far from the old neutral: during stabilization NO movement is emitted
    # (this is what prevents the violent snap), and the neutral re-seeds here.
    clock.t = 0.25
    assert pipe.step(left_at(0.7)).wasd_held == frozenset()
    clock.t = 0.5
    assert pipe.step(left_at(0.7)).wasd_held == frozenset()  # still stabilizing

    # After the 500 ms window, emission resumes — and because the neutral re-seeded at
    # x=0.7, holding there yields no movement (no snap). Real motion still works.
    clock.t = 0.8
    assert pipe.step(left_at(0.7)).wasd_held == frozenset()
    clock.t = 0.85
    assert pipe.step(left_at(0.9)).wasd_held == frozenset({"d"})


def test_low_confidence_hand_treated_as_absent(
    make_extended_landmarks: Callable[..., np.ndarray],
) -> None:
    settings = Settings(tracking={"min_emit_confidence": 0.6})
    emitter = NullEmitter()
    clock = _Clock()
    pipe = _pipeline(settings, emitter, clock)
    lm = make_extended_landmarks(thumb_ext=1.5, offset=(0.5, 0.5, 0.0))
    # score below the floor -> dropped before reaching the gesture machine -> no jump.
    clock.t = 0.0
    pipe.step([HandResult(landmarks=lm, handedness="Right", score=0.3)])
    assert ("key_down", "space") not in emitter.log
    # score above the floor -> jump fires.
    clock.t = 0.05
    pipe.step([HandResult(landmarks=lm, handedness="Right", score=0.9)])
    assert ("key_down", "space") in emitter.log
