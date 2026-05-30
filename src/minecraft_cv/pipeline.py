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

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from minecraft_cv.gestures.extension import ExtensionStateMachine, GestureEvent
from minecraft_cv.gestures.pinch import KEY_DOWN, GestureEvent as PinchGestureEvent, PinchStateMachine
from minecraft_cv.gestures.safety import TrackingLossGuard
from minecraft_cv.input.emitter import InputEmitter, create_emitter
from minecraft_cv.joystick.deadzone import DeadzoneJoystick, anchor_xy
from minecraft_cv.tracking.tracker import HandResult, HandTracker

if TYPE_CHECKING:
    from minecraft_cv.capture.source import FrameSource
    from minecraft_cv.config import Settings


@dataclass
class StepResult:
    """Outcome of one :meth:`Pipeline.step` (for tests / overlay / introspection)."""

    events: list = field(default_factory=list)
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
        recenter_grace_frames: int = 0,
        cardinal_half_width: float = 35.0,
        swap_handedness: bool = True,
        pulse_gestures: frozenset[str] = frozenset(),
        scroll_repeat_rate_hz: float = 8.0,
    ) -> None:
        """Assemble a pipeline from already-constructed components.

        Args:
            emitter: OS-input emitter (``NullEmitter`` for tests/dry-runs).
            guard: Tracking-loss guard wrapping both hands' gesture state machines.
            left_joystick: Left-hand translation joystick (-> WASD).
            right_joystick: Right-hand look joystick (-> relative mouse move).
            bindings: Map of gesture/direction name -> OS key name.
            anchor: Landmark anchor for the joysticks (``"wrist"`` or ``"middle_mcp"``).
            recenter_grace_frames: Consecutive missing-hand frames tolerated before a hand's
                joystick neutral is recentered.
            cardinal_half_width: Half-width in degrees of each pure cardinal direction zone.
            swap_handedness: If True, invert MediaPipe L/R labels in ``_split``.
            pulse_gestures: Set of gesture names that fire as one-shot key taps.
            scroll_repeat_rate_hz: Repeat rate for hotbar scroll while pinch is held.
        """
        self.emitter = emitter
        self.guard = guard
        self.left_joystick = left_joystick
        self.right_joystick = right_joystick
        self.bindings = bindings
        self.anchor = anchor
        self.recenter_grace_frames = recenter_grace_frames
        self.cardinal_half_width = cardinal_half_width
        self.swap_handedness = swap_handedness
        self.pulse_gestures = pulse_gestures
        self.scroll_repeat_rate_hz = scroll_repeat_rate_hz
        self._wasd_held: set[str] = set()
        self._left_miss = 0
        self._right_miss = 0
        # Scroll repeat state: track last scroll time per direction for rate limiting.
        self._last_scroll_time: dict[str, float] = {}

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
        left_sm = ExtensionStateMachine("left", settings.gestures.left_hand)
        right_sm = PinchStateMachine("right", settings.gestures.right_hand)
        guard = TrackingLossGuard(left_sm, right_sm)
        j = settings.joystick
        left_joy = DeadzoneJoystick(
            j.deadzone_radius, j.sensitivity, j.accel_exponent, j.max_output, j.smoothing
        )
        right_joy = DeadzoneJoystick(
            j.deadzone_radius, j.sensitivity, j.accel_exponent, j.max_output, j.smoothing
        )
        return cls(
            emitter=emitter if emitter is not None else create_emitter(settings),
            guard=guard,
            left_joystick=left_joy,
            right_joystick=right_joy,
            bindings=dict(settings.bindings),
            anchor=j.anchor,
            recenter_grace_frames=j.recenter_grace_frames,
            cardinal_half_width=j.cardinal_half_width,
            swap_handedness=settings.tracking.swap_handedness,
            pulse_gestures=left_sm.pulse_gestures,
            scroll_repeat_rate_hz=settings.input.scroll_repeat_rate_hz,
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
            binding = self.bindings.get(event.gesture)
            if binding is None:
                continue

            # Pulse gestures: fire a key tap on engage, ignore the release.
            if event.gesture in self.pulse_gestures:
                if event.action == KEY_DOWN:
                    self.emitter.key_tap(binding)
                # KEY_UP for pulse gestures is handled internally — no OS key_up needed.
                continue

            # Scroll gestures: emit scroll ticks instead of key presses.
            if binding in ("scroll_up", "scroll_down"):
                if event.action == KEY_DOWN:
                    direction = 1 if binding == "scroll_up" else -1
                    self.emitter.scroll(direction)
                    self._last_scroll_time[event.gesture] = time.perf_counter()
                elif event.action != KEY_DOWN:
                    self._last_scroll_time.pop(event.gesture, None)
                continue

            if event.action == KEY_DOWN:
                self.emitter.key_down(binding)
            else:
                self.emitter.key_up(binding)

        # Handle scroll repeat for held hotbar gestures.
        self._repeat_scroll()

        left_out = self._update_translation(left_lm)
        right_out = self._update_look(right_lm)
        return StepResult(
            events=events,
            left_output=left_out,
            right_output=right_out,
            wasd_held=frozenset(self._wasd_held),
        )

    def _repeat_scroll(self) -> None:
        """Re-emit scroll ticks for held hotbar gestures at the configured repeat rate."""
        if not self._last_scroll_time:
            return
        now = time.perf_counter()
        interval = 1.0 / self.scroll_repeat_rate_hz if self.scroll_repeat_rate_hz > 0 else 1.0
        for gesture, last_time in list(self._last_scroll_time.items()):
            if (now - last_time) >= interval:
                binding = self.bindings.get(gesture)
                if binding is not None:
                    direction = 1 if binding == "scroll_up" else -1
                    self.emitter.scroll(direction)
                    self._last_scroll_time[gesture] = now

    def _update_translation(self, landmarks: np.ndarray | None) -> np.ndarray:
        if landmarks is None:
            self._left_miss += 1
            # Release movement keys immediately (fail-safe), but only recenter the neutral
            # after a *sustained* dropout so a one-frame blip doesn't snap it (recenter macro).
            if self._left_miss >= self.recenter_grace_frames:
                self.left_joystick.reset_neutral()
            self._apply_wasd(set())
            return self.left_joystick.zero()
        self._left_miss = 0
        out = self.left_joystick.update(anchor_xy(landmarks, self.anchor))
        self._apply_wasd(self._wasd_targets(out))
        return out

    def _update_look(self, landmarks: np.ndarray | None) -> np.ndarray:
        if landmarks is None:
            self._right_miss += 1
            if self._right_miss >= self.recenter_grace_frames:
                self.right_joystick.reset_neutral()
            return self.right_joystick.zero()
        self._right_miss = 0
        out = self.right_joystick.update(anchor_xy(landmarks, self.anchor))
        if out[0] != 0.0 or out[1] != 0.0:
            self.emitter.mouse_move(float(out[0]), float(out[1]))
        return out

    def _wasd_targets(self, output: np.ndarray) -> set[str]:
        """Translate a joystick output vector into the set of WASD keys to hold.

        Uses angular cardinal zones: each axis direction has a pure zone of
        ``±cardinal_half_width`` degrees where only that direction's key is pressed.
        Between zones, two adjacent keys are pressed (diagonal).

        Frame of reference: normalized image coords, y increases downward. Moving the hand
        up (``y < 0``) is forward (W); right (``x > 0``) is D.
        """
        x, y = float(output[0]), float(output[1])
        if x == 0.0 and y == 0.0:
            return set()

        # Compute angle in degrees. atan2 uses standard math convention (counter-clockwise
        # from +x axis). We convert to our compass where 0° = right (+x), 90° = down (+y).
        angle_deg = math.degrees(math.atan2(y, x))  # range [-180, 180]
        if angle_deg < 0:
            angle_deg += 360.0  # range [0, 360)

        keys: set[str] = set()
        hw = self.cardinal_half_width

        # Cardinal zones centered at: Right=0°, Down=90°, Left=180°, Up=270°
        # In our coordinate system (y-down is positive):
        #   Right (+x):  0° center
        #   Down  (+y): 90° center  -> Back (S)
        #   Left  (-x): 180° center -> Left (A)
        #   Up    (-y): 270° center -> Forward (W)

        # Check each cardinal zone. If the angle is within ±hw of the center,
        # that key is pressed. Zones can overlap at diagonals.
        # Right (D): centered at 0° (also 360°)
        if angle_deg <= hw or angle_deg >= (360.0 - hw):
            keys.add(self.bindings["right"])
        # Down/Back (S): centered at 90°
        if (90.0 - hw) <= angle_deg <= (90.0 + hw):
            keys.add(self.bindings["back"])
        # Left (A): centered at 180°
        if (180.0 - hw) <= angle_deg <= (180.0 + hw):
            keys.add(self.bindings["left"])
        # Up/Forward (W): centered at 270°
        if (270.0 - hw) <= angle_deg <= (270.0 + hw):
            keys.add(self.bindings["forward"])

        # Diagonal zones: the gaps between cardinal zones. If we're in a gap, press
        # both adjacent keys.
        if not keys:
            # We're in a diagonal gap. Figure out which two cardinals are adjacent.
            if hw < angle_deg < (90.0 - hw):
                # Between Right and Down -> D + S
                keys.add(self.bindings["right"])
                keys.add(self.bindings["back"])
            elif (90.0 + hw) < angle_deg < (180.0 - hw):
                # Between Down and Left -> S + A
                keys.add(self.bindings["back"])
                keys.add(self.bindings["left"])
            elif (180.0 + hw) < angle_deg < (270.0 - hw):
                # Between Left and Up -> A + W
                keys.add(self.bindings["left"])
                keys.add(self.bindings["forward"])
            elif (270.0 + hw) < angle_deg < (360.0 - hw):
                # Between Up and Right -> W + D
                keys.add(self.bindings["forward"])
                keys.add(self.bindings["right"])

        return keys

    def _apply_wasd(self, target: set[str]) -> None:
        for key in self._wasd_held - target:
            self.emitter.key_up(key)
        for key in target - self._wasd_held:
            self.emitter.key_down(key)
        self._wasd_held = target

    def _split(self, results: list[HandResult]) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Split tracker results into left/right landmark arrays.

        When ``swap_handedness`` is True, MediaPipe's ``"Left"``/``"Right"`` labels are
        inverted. This fixes the common issue where a mirrored camera feed causes MediaPipe
        to label the user's physical left hand as the image-convention right hand.
        """
        left: np.ndarray | None = None
        right: np.ndarray | None = None
        for r in results:
            label = r.handedness
            if self.swap_handedness:
                label = "Right" if label == "Left" else "Left"
            if label == "Left" and left is None:
                left = r.landmarks
            elif label == "Right" and right is None:
                right = r.landmarks
        return left, right

    def shutdown(self) -> None:
        """Release every held key/button (gestures + WASD) — fail-safe on any exit."""
        for event in self.guard.release_all():
            binding = self.bindings.get(event.gesture)
            if binding is not None and binding not in ("scroll_up", "scroll_down"):
                self.emitter.key_up(binding)
        self._apply_wasd(set())
        self._last_scroll_time.clear()
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
    mirror = settings.camera.mirror
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

            # Mirror first, in place, so tracking, the joystick vectors, the WASD directions,
            # and the debug overlay all share one consistent (mirrored) frame of reference.
            if mirror:
                frame = cv2.flip(frame, 1)
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
