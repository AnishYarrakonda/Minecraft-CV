"""Pipeline: wire capture -> tracking -> gestures + joysticks -> input.

The per-frame decision logic lives in :meth:`Pipeline.step`, which is **pure** with respect
to the OS: it takes tracker results and drives an injected :class:`InputEmitter`. With a
:class:`NullEmitter` and synthetic :class:`HandResult` objects this is fully unit-testable
with no camera, no MediaPipe, and no OS input (hard invariant #2).

The live loop :func:`run_pipeline` owns the camera, the threaded frame buffer, the optional
debug overlay (HighGUI on the main thread only), and fail-safe shutdown. OpenCV is imported
lazily inside that function so importing this module stays light.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from minecraft_cv.gestures.pinch import KEY_DOWN, GestureEvent, PinchStateMachine
from minecraft_cv.gestures.safety import TrackingLossGuard
from minecraft_cv.input.emitter import InputEmitter, NullEmitter, create_emitter
from minecraft_cv.joystick.deadzone import DeadzoneJoystick, anchor_xy
from minecraft_cv.tracking.tracker import HandResult, HandTracker

if TYPE_CHECKING:
    from minecraft_cv.capture.source import FrameSource
    from minecraft_cv.config import Settings


@dataclass
class StepResult:
    """Outcome of one :meth:`Pipeline.step` (for tests / overlay / introspection)."""

    events: list[GestureEvent] = field(default_factory=list)
    left_output: np.ndarray = field(default_factory=lambda: np.zeros(2))
    right_output: np.ndarray = field(default_factory=lambda: np.zeros(2))
    wasd_held: frozenset[str] = field(default_factory=frozenset)


class Pipeline:
    """Stateful per-frame controller: gestures + spatial joysticks -> input emitter."""

    def __init__(
        self,
        emitter: InputEmitter,
        guard: TrackingLossGuard,
        left_joystick: DeadzoneJoystick,
        right_joystick: DeadzoneJoystick,
        bindings: dict[str, str],
        anchor: str = "wrist",
    ) -> None:
        """Assemble a pipeline from already-constructed components.

        Args:
            emitter: OS-input emitter (``NullEmitter`` for tests/dry-runs).
            guard: Tracking-loss guard wrapping both hands' pinch state machines.
            left_joystick: Left-hand translation joystick (-> WASD).
            right_joystick: Right-hand look joystick (-> relative mouse move).
            bindings: Map of gesture/direction name -> OS key name.
            anchor: Landmark anchor for the joysticks (``"wrist"`` or ``"middle_mcp"``).
        """
        self.emitter = emitter
        self.guard = guard
        self.left_joystick = left_joystick
        self.right_joystick = right_joystick
        self.bindings = bindings
        self.anchor = anchor
        self._wasd_held: set[str] = set()

    @classmethod
    def from_settings(
        cls, settings: Settings, emitter: InputEmitter | None = None
    ) -> Pipeline:
        """Build a pipeline from a :class:`Settings` model.

        Args:
            settings: Loaded configuration.
            emitter: Override emitter; if None, one is created from ``settings`` (a
                ``NullEmitter`` unless ``input.enabled`` is True).

        Returns:
            A ready-to-run :class:`Pipeline`.
        """
        left_sm = PinchStateMachine("left", settings.gestures.left_hand)
        right_sm = PinchStateMachine("right", settings.gestures.right_hand)
        guard = TrackingLossGuard(left_sm, right_sm)
        j = settings.joystick
        left_joy = DeadzoneJoystick(j.deadzone_radius, j.sensitivity, j.accel_exponent, j.max_output)
        right_joy = DeadzoneJoystick(j.deadzone_radius, j.sensitivity, j.accel_exponent, j.max_output)
        return cls(
            emitter=emitter if emitter is not None else create_emitter(settings),
            guard=guard,
            left_joystick=left_joy,
            right_joystick=right_joy,
            bindings=dict(settings.bindings),
            anchor=j.anchor,
        )

    # --- per-frame logic ----------------------------------------------------
    def step(self, results: list[HandResult]) -> StepResult:
        """Process one frame of tracker results and drive the emitter.

        Args:
            results: Detected hands for this frame (0-2). Split by handedness label.

        Returns:
            A :class:`StepResult` describing what happened (events + joystick outputs).
        """
        left_lm, right_lm = self._split(results)

        events = self.guard.process(left_lm, right_lm)
        for event in events:
            key = self.bindings[event.gesture]
            if event.action == KEY_DOWN:
                self.emitter.key_down(key)
            else:
                self.emitter.key_up(key)

        left_out = self._update_translation(left_lm)
        right_out = self._update_look(right_lm)
        return StepResult(
            events=events,
            left_output=left_out,
            right_output=right_out,
            wasd_held=frozenset(self._wasd_held),
        )

    def _update_translation(self, landmarks: np.ndarray | None) -> np.ndarray:
        if landmarks is None:
            self.left_joystick.reset_neutral()  # re-entry recalibrates (recenter macro)
            self._apply_wasd(set())
            return self.left_joystick.zero()
        out = self.left_joystick.update(anchor_xy(landmarks, self.anchor))
        self._apply_wasd(self._wasd_targets(out))
        return out

    def _update_look(self, landmarks: np.ndarray | None) -> np.ndarray:
        if landmarks is None:
            self.right_joystick.reset_neutral()
            return self.right_joystick.zero()
        out = self.right_joystick.update(anchor_xy(landmarks, self.anchor))
        if out[0] != 0.0 or out[1] != 0.0:
            self.emitter.mouse_move(float(out[0]), float(out[1]))
        return out

    def _wasd_targets(self, output: np.ndarray) -> set[str]:
        """Translate a joystick output vector into the set of WASD keys to hold.

        Frame of reference: normalized image coords, y increases downward. Moving the hand
        up (``y < 0``) is forward (W); right (``x > 0``) is D.
        """
        keys: set[str] = set()
        x, y = float(output[0]), float(output[1])
        if x > 0.0:
            keys.add(self.bindings["right"])
        elif x < 0.0:
            keys.add(self.bindings["left"])
        if y < 0.0:
            keys.add(self.bindings["forward"])
        elif y > 0.0:
            keys.add(self.bindings["back"])
        return keys

    def _apply_wasd(self, target: set[str]) -> None:
        for key in self._wasd_held - target:
            self.emitter.key_up(key)
        for key in target - self._wasd_held:
            self.emitter.key_down(key)
        self._wasd_held = target

    @staticmethod
    def _split(results: list[HandResult]) -> tuple[np.ndarray | None, np.ndarray | None]:
        left: np.ndarray | None = None
        right: np.ndarray | None = None
        for r in results:
            if r.handedness == "Left" and left is None:
                left = r.landmarks
            elif r.handedness == "Right" and right is None:
                right = r.landmarks
        return left, right

    def shutdown(self) -> None:
        """Release every held key/button (gestures + WASD) — fail-safe on any exit."""
        for event in self.guard.release_all():
            self.emitter.key_up(self.bindings[event.gesture])
        self._apply_wasd(set())
        self.emitter.release_all()


def run_pipeline(settings: Settings, source: FrameSource | None = None) -> None:
    """Run the live capture loop until the source is exhausted or interrupted.

    Args:
        settings: Loaded configuration (camera, tracking, gestures, input, debug).
        source: Optional injected frame source. If None, a camera or clip source is built
            from ``settings`` (a clip would be wired by the CLI; default is the camera).

    Notes:
        OpenCV (color convert / resize / overlay) is imported lazily here. HighGUI calls run
        on this (main) thread only. On any exit the emitter releases all held keys.
    """
    import cv2  # lazy: keeps this module importable without OpenCV (tests)

    from minecraft_cv.capture.buffer import FrameBuffer
    from minecraft_cv.capture.source import AVFoundationSource

    if source is None:
        source = AVFoundationSource(
            index=settings.camera.index,
            width=settings.camera.width,
            height=settings.camera.height,
            fps=settings.camera.fps,
        )

    tracker = HandTracker.create(settings.tracking.backend, settings.tracking.device)
    pipeline = Pipeline.from_settings(settings)
    buffer = FrameBuffer(source).start()
    res_w, res_h = settings.tracking.input_resolution
    overlay = settings.debug.overlay
    window = "minecraft_cv"
    last_seq = -1

    try:
        while True:
            seq, frame = buffer.latest()
            if frame is None:
                if buffer.exhausted:
                    break
                time.sleep(0.001)
                continue
            if seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (res_w, res_h))
            results = tracker.detect(small)
            result = pipeline.step(results)

            if overlay:
                _draw_overlay(frame, results, result)
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if buffer.exhausted:
                break
    finally:
        pipeline.shutdown()
        buffer.stop()
        tracker.close()
        if overlay:
            cv2.destroyAllWindows()


def _draw_overlay(frame: np.ndarray, results: list[HandResult], step: StepResult) -> None:
    """Draw landmark dots + gesture state onto ``frame`` (debug only)."""
    import cv2

    h, w = frame.shape[:2]
    for hand in results:
        for x, y, _ in hand.landmarks:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
    text = " ".join(f"{e.gesture}:{e.action}" for e in step.events) or "-"
    cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    wasd = "".join(sorted(step.wasd_held)) or "."
    cv2.putText(frame, f"WASD:{wasd}", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


__all__ = ["Pipeline", "StepResult", "run_pipeline"]
