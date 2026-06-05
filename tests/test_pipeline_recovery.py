"""Pipeline-level integration of Sprint-by-velocity (Task 2) and tracking-loss recovery
(Task 5), driven by an injected deterministic clock (no real time, no camera).

Convention reminder: a ``HandResult`` with ``handedness="Right"`` is the user's physical
**left** hand (swap_handedness default), which drives WASD + the velocity sprint.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from conftest import make_calibrated_settings
from minecraft_cv.config import Settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult

_OPEN_EXTENSIONS = {"index": 1.3, "middle": 1.3, "ring": 1.3, "pinky": 1.3}


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


def test_long_dropout_then_stabilization_prevents_snap(
    make_palm_normal_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    emitter = NullEmitter()
    clock = _Clock()
    pipe = _pipeline(make_calibrated_settings(), emitter, clock)

    def left_at(x: float) -> list[HandResult]:
        return [make_hand_result(make_palm_normal_landmarks(normal_xy=(x, 0.0)), "Right")]

    clock.t = 0.0
    pipe.step(left_at(0.0))  # neutral
    clock.t = 0.01
    pipe.step(left_at(0.3))  # rotating right

    # Long dropout (>100 ms): keys release, neutral flushed.
    clock.t = 0.2
    pipe.step([])
    assert "d" not in emitter.held_keys

    # Re-entry tilted away from calibrated neutral: during stabilization NO movement is
    # emitted, preventing a camera/movement snap.
    clock.t = 0.25
    pipe.step(left_at(0.3))
    clock.t = 0.5
    pipe.step(left_at(0.3))  # still stabilizing

    # After the 500 ms window, emission resumes relative to calibrated neutral.
    clock.t = 0.8
    pipe.step(left_at(0.3))


def test_low_confidence_hand_treated_as_absent(
    make_palm_normal_landmarks: Callable[..., np.ndarray],
) -> None:
    settings = make_calibrated_settings(tracking={"min_emit_confidence": 0.6})
    emitter = NullEmitter()
    clock = _Clock()
    pipe = _pipeline(settings, emitter, clock)
    lm = make_palm_normal_landmarks(distances={"index": 0.20})
    # score below the floor -> dropped before reaching the gesture machine -> no jump.
    clock.t = 0.0
    pipe.step([HandResult(landmarks=lm, handedness="Right", score=0.3)])
    assert ("key_down", "d") not in emitter.log
    # score above the floor -> jump fires.
    clock.t = 0.05
    pipe.step([HandResult(landmarks=lm, handedness="Right", score=0.9)])
    pipe.step([HandResult(landmarks=lm, handedness="Right", score=0.9)])
    assert ("key_down", "d") in emitter.log
