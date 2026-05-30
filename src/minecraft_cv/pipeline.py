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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from minecraft_cv.gestures.extension import ExtensionStateMachine
from minecraft_cv.gestures.inventory import InventoryModeToggle
from minecraft_cv.gestures.pinch import KEY_DOWN, PinchStateMachine
from minecraft_cv.gestures.safety import TrackingLossGuard
from minecraft_cv.input.emitter import InputEmitter, create_emitter
from minecraft_cv.joystick.deadzone import ANCHOR_INDEX, DeadzoneJoystick, anchor_xy
from minecraft_cv.joystick.one_euro import OneEuroFilter
from minecraft_cv.joystick.sprint_velocity import ENGAGE, RELEASE, SprintVelocityTrigger
from minecraft_cv.recovery import HandRecovery, RecoveryDecision
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
    inventory_active: bool = False


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
        look_filter: OneEuroFilter | None = None,
        inventory_toggle: InventoryModeToggle | None = None,
        cursor_gain: float = 1.0,
        sprint_trigger: SprintVelocityTrigger | None = None,
        left_recovery: HandRecovery | None = None,
        right_recovery: HandRecovery | None = None,
        min_emit_confidence: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
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
            look_filter: Optional One-Euro filter smoothing the right-hand mouse-look output
                before emission. ``None`` falls back to the joystick's EMA smoothing only.
            inventory_toggle: Optional two-hand-pose detector for inventory mode. ``None``
                disables inventory mode entirely.
            cursor_gain: Gain mapping the right-hand normalized anchor displacement to
                normalized screen displacement when driving the absolute cursor (inventory).
            sprint_trigger: Optional depth-velocity Sprint trigger (Task 2). ``None`` disables
                velocity sprint (the static extension ``sprint`` gesture is unaffected).
            left_recovery: Per-hand tracking-loss recovery controller for the left hand
                (Task 5). ``None`` builds a default one.
            right_recovery: As above for the right hand.
            min_emit_confidence: Drop detected hands whose handedness score is below this in
                :meth:`_split` (they are then treated as absent by the recovery path).
            clock: Monotonic seconds source; injectable for deterministic tests. Used for
                scroll-repeat timing, the look filter, sprint velocity, and recovery windows.
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
        self.look_filter = look_filter
        self.inventory_toggle = inventory_toggle
        self.cursor_gain = float(cursor_gain)
        self.sprint_trigger = sprint_trigger
        self.left_recovery = left_recovery if left_recovery is not None else HandRecovery()
        self.right_recovery = right_recovery if right_recovery is not None else HandRecovery()
        self.min_emit_confidence = float(min_emit_confidence)
        self._clock = clock
        self._wasd_held: set[str] = set()
        self._left_miss = 0
        self._right_miss = 0
        self._sprint_active = False
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
        dz = dict(
            dynamic=j.dynamic_deadzone,
            calibration_frames=j.calibration_frames,
            dynamic_margin=j.dynamic_deadzone_margin,
        )
        left_joy = DeadzoneJoystick(
            j.deadzone_radius, j.sensitivity, j.accel_exponent, j.max_output, j.smoothing, **dz
        )
        right_joy = DeadzoneJoystick(
            j.deadzone_radius, j.sensitivity, j.accel_exponent, j.max_output, j.smoothing, **dz
        )
        look_filter = (
            OneEuroFilter(
                min_cutoff=j.one_euro_min_cutoff,
                beta=j.one_euro_beta,
                d_cutoff=j.one_euro_d_cutoff,
            )
            if j.look_filter == "one_euro"
            else None
        )
        inv = settings.inventory
        inventory_toggle = InventoryModeToggle(
            enabled=inv.enabled,
            open_threshold=inv.open_threshold,
            thumb_open_threshold=inv.thumb_open_threshold,
            hold_frames=inv.hold_frames,
            cooldown_frames=inv.cooldown_frames,
        )
        sp = settings.sprint
        sprint_trigger = (
            SprintVelocityTrigger(
                v_sprint=sp.v_sprint,
                trigger_frames=sp.trigger_frames,
                release_margin=sp.release_margin,
                enabled=True,
            )
            if sp.enabled
            else None
        )
        tr = settings.tracking
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
            look_filter=look_filter,
            inventory_toggle=inventory_toggle,
            cursor_gain=inv.cursor_gain,
            sprint_trigger=sprint_trigger,
            left_recovery=HandRecovery(tr.dropout_flush_ms, tr.stabilization_ms),
            right_recovery=HandRecovery(tr.dropout_flush_ms, tr.stabilization_ms),
            min_emit_confidence=tr.min_emit_confidence,
        )

    # --- per-frame logic ----------------------------------------------------
    def step(self, results: list[HandResult]) -> StepResult:
        """Process one frame of tracker results and drive the emitter.

        Args:
            results: Detected hands for this frame (0-2). Split by handedness label.

        Returns:
            A :class:`StepResult` describing what happened (events + joystick outputs).
        """
        now = self._clock()
        left_lm, right_lm = self._split(results)

        # Tracking-loss recovery (Task 5): decide per hand whether to emit, track, or flush.
        left_dec = self.left_recovery.update(left_lm is not None, now)
        right_dec = self.right_recovery.update(right_lm is not None, now)
        if left_dec.flush:
            self._flush_left()
        if right_dec.flush:
            self._flush_right()

        # Inventory-mode toggle is evaluated first; a flip cleans up movement state so the
        # mode boundary never leaves a key stuck or a stale neutral behind.
        inventory_active = False
        if self.inventory_toggle is not None:
            toggle = self.inventory_toggle.update(left_lm, right_lm)
            inventory_active = toggle.active
            if toggle.toggled:
                self._on_inventory_toggle()

        # Feed each hand's landmarks to its gesture machine only when that hand may emit
        # (NORMAL phase). Absent or stabilizing -> pass None so the machine resets: no stuck
        # keys, and no phantom presses fired during the re-entry settle window.
        events = self.guard.process(
            left_lm if left_dec.emit else None,
            right_lm if right_dec.emit else None,
        )
        for event in events:
            # In inventory mode, suppress new LEFT-hand gameplay actions (jump/sneak/sprint/
            # pulses) so menu navigation can't fire them. KEY_UP still passes through so any
            # key held when the mode was entered is released, never stuck.
            if inventory_active and event.hand == "left" and event.action == KEY_DOWN:
                continue

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
                    self._last_scroll_time[event.gesture] = now
                elif event.action != KEY_DOWN:
                    self._last_scroll_time.pop(event.gesture, None)
                continue

            if event.action == KEY_DOWN:
                self.emitter.key_down(binding)
            else:
                self.emitter.key_up(binding)

        # Handle scroll repeat for held hotbar gestures.
        self._repeat_scroll(now)

        if inventory_active:
            # WASD paused; the right hand drives the OS cursor in absolute screen coords.
            self._apply_wasd(set())
            self._release_sprint()
            right_out = self._update_cursor(right_lm if right_dec.emit else None)
            return StepResult(
                events=events,
                left_output=self.left_joystick.zero(),
                right_output=right_out,
                wasd_held=frozenset(),
                inventory_active=True,
            )

        left_out = self._update_translation(left_lm, left_dec, now)
        right_out = self._update_look(right_lm, right_dec, now)
        return StepResult(
            events=events,
            left_output=left_out,
            right_output=right_out,
            wasd_held=frozenset(self._wasd_held),
            inventory_active=False,
        )

    def _on_inventory_toggle(self) -> None:
        """Clean up movement state when inventory mode flips (either direction).

        Releases held WASD keys and any held left-hand gesture keys, recenters both joysticks,
        and resets the look filter so neither mode inherits stale state from the other.
        """
        self._apply_wasd(set())
        self._release_sprint()
        for event in self.guard.reset_left():
            binding = self.bindings.get(event.gesture)
            if binding is not None and binding not in ("scroll_up", "scroll_down"):
                self.emitter.key_up(binding)
        self.left_joystick.reset_neutral()
        self.right_joystick.reset_neutral()
        if self.look_filter is not None:
            self.look_filter.reset()

    # --- tracking-loss flush helpers (Task 5) -------------------------------
    def _flush_left(self) -> None:
        """Hard-flush the left hand after a sustained dropout: recenter + drop sprint."""
        self._release_sprint()
        if self.sprint_trigger is not None:
            self.sprint_trigger.reset_neutral()
        self.left_joystick.reset_neutral()

    def _flush_right(self) -> None:
        """Hard-flush the right hand: recenter the look joystick + drop look velocity."""
        self.right_joystick.reset_neutral()
        if self.look_filter is not None:
            self.look_filter.reset()

    def _release_sprint(self) -> None:
        """Release a held velocity-sprint Ctrl and disarm the trigger (fail-safe)."""
        if self._sprint_active:
            key = self.bindings.get("sprint")
            if key is not None:
                self.emitter.key_up(key)
            self._sprint_active = False
        if self.sprint_trigger is not None:
            self.sprint_trigger.reset()

    def _update_sprint(self, landmarks: np.ndarray, now: float) -> None:
        """Advance the depth-velocity sprint trigger and emit/clear the Sprint key (Ctrl).

        Args:
            landmarks: ``(21, 3)`` left-hand landmarks for this frame.
            now: Monotonic timestamp (seconds) for the velocity estimate.
        """
        if self.sprint_trigger is None:
            return
        z = float(landmarks[ANCHOR_INDEX[self.anchor]][2])
        token = self.sprint_trigger.update(z, now)
        key = self.bindings.get("sprint")
        if key is None:
            return
        if token == ENGAGE:
            self.emitter.key_down(key)
            self._sprint_active = True
        elif token == RELEASE:
            self.emitter.key_up(key)
            self._sprint_active = False

    def _update_cursor(self, landmarks: np.ndarray | None) -> np.ndarray:
        """Map the right-hand anchor to an absolute cursor position (inventory mode).

        Args:
            landmarks: ``(21, 3)`` right-hand landmarks, or ``None`` if absent this frame.

        Returns:
            The ``(2,)`` normalized screen position commanded this frame, or zeros if the hand
            is absent. Frame of reference: normalized image coords (already mirrored upstream),
            mapped about screen-center and scaled by ``cursor_gain``, clamped to ``[0, 1]``.
        """
        if landmarks is None:
            return np.zeros(2, dtype=np.float64)
        pos = anchor_xy(landmarks, self.anchor)
        screen = 0.5 + (pos - 0.5) * self.cursor_gain
        screen = np.clip(screen, 0.0, 1.0)
        self.emitter.mouse_move_abs(float(screen[0]), float(screen[1]))
        return screen

    def _repeat_scroll(self, now: float) -> None:
        """Re-emit scroll ticks for held hotbar gestures at the configured repeat rate."""
        if not self._last_scroll_time:
            return
        interval = 1.0 / self.scroll_repeat_rate_hz if self.scroll_repeat_rate_hz > 0 else 1.0
        for gesture, last_time in list(self._last_scroll_time.items()):
            if (now - last_time) >= interval:
                binding = self.bindings.get(gesture)
                if binding is not None:
                    direction = 1 if binding == "scroll_up" else -1
                    self.emitter.scroll(direction)
                    self._last_scroll_time[gesture] = now

    def _update_translation(
        self, landmarks: np.ndarray | None, dec: RecoveryDecision, now: float
    ) -> np.ndarray:
        if not dec.present or landmarks is None:
            self._left_miss += 1
            # Release movement keys immediately (fail-safe), but only recenter the neutral
            # after a *sustained* dropout so a one-frame blip doesn't snap it (recenter macro).
            if self._left_miss >= self.recenter_grace_frames:
                self.left_joystick.reset_neutral()
            self._apply_wasd(set())
            self._release_sprint()
            return self.left_joystick.zero()
        self._left_miss = 0
        if not dec.emit:
            # Stabilizing on re-entry: feed coords so the neutral re-seeds, but emit nothing.
            self.left_joystick.update(anchor_xy(landmarks, self.anchor))
            self._apply_wasd(set())
            return self.left_joystick.zero()
        out = self.left_joystick.update(anchor_xy(landmarks, self.anchor))
        self._update_sprint(landmarks, now)
        target = self._wasd_targets(out)
        if self._sprint_active:
            # Sprint forces forward (W) held alongside Ctrl, even at joystick neutral.
            target.add(self.bindings["forward"])
        self._apply_wasd(target)
        return out

    def _update_look(
        self, landmarks: np.ndarray | None, dec: RecoveryDecision, now: float
    ) -> np.ndarray:
        if not dec.present or landmarks is None:
            self._right_miss += 1
            if self._right_miss >= self.recenter_grace_frames:
                self.right_joystick.reset_neutral()
            # Drop the look filter's velocity history so re-entry doesn't jerk the camera.
            if self.look_filter is not None:
                self.look_filter.reset()
            return self.right_joystick.zero()
        self._right_miss = 0
        if not dec.emit:
            # Stabilizing: re-seed the neutral, keep the filter clear, emit no mouse-look.
            self.right_joystick.update(anchor_xy(landmarks, self.anchor))
            if self.look_filter is not None:
                self.look_filter.reset()
            return self.right_joystick.zero()
        out = self.right_joystick.update(anchor_xy(landmarks, self.anchor))
        if self.look_filter is not None:
            # Velocity-adaptive smoothing: steady at rest, snappy in motion. (Task 4.)
            out = self.look_filter.filter(out, now)
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
            # Below-confidence detections are treated as absent (fed to the recovery path).
            if r.score < self.min_emit_confidence:
                continue
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
        self._release_sprint()
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
    overlay_every = max(1, settings.debug.overlay_every)
    window = "minecraft_cv"
    
    # Pre-allocate reuse buffers for the hot loop
    small_bgr = np.empty((res_h, res_w, 3), dtype=np.uint8)
    small_rgb = np.empty((res_h, res_w, 3), dtype=np.uint8)
    
    last_seq = -1
    processed = 0
    dropped = 0
    t_start = time.monotonic()
    last_frame_time = t_start

    try:
        while True:
            if buffer.error:
                raise buffer.error
            if time.monotonic() - last_frame_time > 2.0:
                raise RuntimeError("Camera stalled")

            seq, frame = buffer.latest()
            if frame is None:
                if buffer.exhausted:
                    break
                time.sleep(0.001)
                continue
            if seq == last_seq:
                time.sleep(0.001)
                continue
            if last_seq != -1 and seq > last_seq + 1:
                dropped += (seq - last_seq - 1)
            last_seq = seq
            last_frame_time = time.monotonic()

            # Mirror first, in place, so tracking, the joystick vectors, the WASD directions,
            # and the debug overlay all share one consistent (mirrored) frame of reference.
            if mirror:
                frame = cv2.flip(frame, 1)
            
            # Resize BEFORE color convert, and use pre-allocated buffers
            cv2.resize(frame, (res_w, res_h), dst=small_bgr)
            cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB, dst=small_rgb)
            
            results = tracker.detect(small_rgb)
            result = pipeline.step(results)
            processed += 1

            # Overlay is a debug-only luxury and not free; decimate the (HighGUI) draw to
            # protect the real-time loop. Tracking/gestures/input still run every frame.
            if overlay and processed % overlay_every == 0:
                _draw_overlay(frame, results, result)
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if buffer.exhausted:
                break
    finally:
        t_elapsed = time.monotonic() - t_start
        fps = processed / t_elapsed if t_elapsed > 0 else 0.0
        print(f"Pipeline shutdown. Processed {processed} frames in {t_elapsed:.2f}s ({fps:.1f} FPS), dropped {dropped} frames.")
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
