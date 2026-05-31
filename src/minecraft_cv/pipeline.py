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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from minecraft_cv.gestures.inventory import InventoryModeToggle
from minecraft_cv.gestures.pinch import KEY_DOWN
from minecraft_cv.gestures.registry import GestureStateMachine
from minecraft_cv.gestures.safety import AnyGestureEvent, TrackingLossGuard
from minecraft_cv.input.emitter import InputEmitter, create_emitter
from minecraft_cv.joystick.deadzone import ANCHOR_INDEX, anchor_xy
from minecraft_cv.joystick.one_euro import OneEuroFilter
from minecraft_cv.joystick.palm_normal import PalmNormalJoystick, palm_normal_xy
from minecraft_cv.joystick.palm_tilt import palm_tilt_xy
from minecraft_cv.joystick.sprint_velocity import ENGAGE, RELEASE, SprintVelocityTrigger
from minecraft_cv.joystick.steering import accel_curve, cardinal_keys
from minecraft_cv.joystick.wrist_rotation import WristRotationJoystick, palm_xz
from minecraft_cv.recovery import HandRecovery, RecoveryDecision
from minecraft_cv.tracking.tracker import HandResult, HandTracker

if TYPE_CHECKING:
    from minecraft_cv.capture.source import FrameSource
    from minecraft_cv.config import Settings


# Per-hand tracking status for the HUD (readable name, not a bool).
HandStatus = Literal["normal", "stabilizing", "absent"]


class JoystickLike(Protocol):
    """Shared surface for palm-normal and legacy wrist-rotation joysticks."""

    @property
    def neutral(self) -> np.ndarray | None: ...
    @property
    def sensitivity(self) -> np.ndarray: ...
    def reset_neutral(self) -> None: ...
    def update(self, signal: np.ndarray) -> np.ndarray: ...
    def zero(self) -> np.ndarray: ...


@dataclass
class StepResult:
    """Outcome of one :meth:`Pipeline.step` (for tests / overlay / introspection)."""

    events: Sequence[AnyGestureEvent] = field(default_factory=list)
    left_output: np.ndarray = field(default_factory=lambda: np.zeros(2))
    right_output: np.ndarray = field(default_factory=lambda: np.zeros(2))
    wasd_held: frozenset[str] = field(default_factory=frozenset)
    inventory_active: bool = False

    # Debug-only fields (populated when the overlay is drawn; zero/None otherwise).
    # These never affect emission (hard invariant #2).
    left_signal: np.ndarray | None = None
    """Raw tilt/normal signal ``(x, y)`` from the left hand this frame."""
    right_signal: np.ndarray | None = None
    """Raw tilt/normal signal ``(x, y)`` from the right hand this frame."""
    left_neutral: np.ndarray | None = None
    """Left joystick neutral ``(x, y)`` at the time of this step."""
    right_neutral: np.ndarray | None = None
    """Right joystick neutral ``(x, y)`` at the time of this step."""
    deadzone: float = 0.0
    """Deadzone radius used by the left joystick (for overlay ring)."""
    cardinal_half_width: float = 35.0
    """Cardinal-zone half-width (degrees) used for WASD this frame."""
    left_status: HandStatus = "absent"
    """Tracking state of the left hand: ``normal``, ``stabilizing``, or ``absent``."""
    right_status: HandStatus = "absent"
    """Tracking state of the right hand: ``normal``, ``stabilizing``, or ``absent``."""


class Pipeline:
    """Stateful per-frame controller: gestures + spatial joysticks -> input emitter."""

    def __init__(
        self,
        emitter: InputEmitter,
        guard: TrackingLossGuard,
        left_joystick: JoystickLike,
        right_joystick: JoystickLike,
        bindings: dict[str, str],
        joystick_signal: Callable[[np.ndarray], np.ndarray] = palm_xz,
        joystick_mode: str = "wrist_rotation",
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
        look_accel_exponent: float = 1.6,
    ) -> None:
        """Assemble a pipeline from already-constructed components.

        Args:
            emitter: OS-input emitter (``NullEmitter`` for tests/dry-runs).
            guard: Tracking-loss guard wrapping both hands' gesture state machines.
            left_joystick: Left-hand joystick (-> WASD).
            right_joystick: Right-hand joystick (-> relative mouse move).
            bindings: Map of gesture/direction name -> OS key name.
            joystick_signal: Landmark signal extractor passed into both joysticks.
            joystick_mode: Active joystick mode (``palm_tilt``/``palm_normal``/
                ``wrist_rotation``). In ``palm_tilt`` the inventory cursor is driven by the
                calibrated tilt signal (absolute pointer); other modes use the legacy
                anchor-position cursor.
            anchor: Legacy anchor selector used by velocity sprint and optional cursor mode.
            recenter_grace_frames: Consecutive missing-hand frames tolerated before a hand's
                joystick neutral is recentered.
            cardinal_half_width: Half-angle of each pure cardinal direction zone (degrees).
                Replaces the old independent per-axis sign check; fires ``cardinal_keys`` so
                only the geometrically-nearest cardinal(s) are pressed.
            swap_handedness: If True, invert MediaPipe L/R labels in ``_split``.
            pulse_gestures: Legacy one-shot gestures; the default detector map uses holds.
            scroll_repeat_rate_hz: Repeat rate for hotbar scroll while pinch is held.
            look_filter: Optional One-Euro filter smoothing the right-hand mouse-look output
                before emission. ``None`` falls back to the joystick's EMA smoothing only.
            inventory_toggle: Optional two-hand-pose detector for inventory mode. ``None``
                disables inventory mode entirely.
            cursor_gain: Gain mapping the right-hand normalized anchor displacement to
                normalized screen displacement when driving the absolute cursor (inventory).
            sprint_trigger: Optional depth-velocity Sprint trigger (Task 2). ``None`` disables
                velocity sprint (the configured ``sprint`` gesture is unaffected).
            left_recovery: Per-hand tracking-loss recovery controller for the left hand
                (Task 5). ``None`` builds a default one.
            right_recovery: As above for the right hand.
            min_emit_confidence: Drop detected hands whose handedness score is below this in
                :meth:`_split` (they are then treated as absent by the recovery path).
            clock: Monotonic seconds source; injectable for deterministic tests. Used for
                scroll-repeat timing, the look filter, sprint velocity, and recovery windows.
            look_accel_exponent: Ease-in exponent for the exponential acceleration curve
                applied to the mouse-look output. ``> 1`` keeps small tilts precise and
                large tilts fast; ``1.0`` is a linear pass-through.
        """
        self.emitter = emitter
        self.guard = guard
        self.left_joystick = left_joystick
        self.right_joystick = right_joystick
        self.bindings = bindings
        self.joystick_signal = joystick_signal
        self.joystick_mode = joystick_mode
        self.anchor = anchor
        self.recenter_grace_frames = recenter_grace_frames
        self.cardinal_half_width = cardinal_half_width
        self.swap_handedness = swap_handedness
        self.pulse_gestures = pulse_gestures
        self.scroll_repeat_rate_hz = scroll_repeat_rate_hz
        self.look_filter = look_filter
        self.inventory_toggle = inventory_toggle
        self.cursor_gain = cursor_gain
        self.sprint_trigger = sprint_trigger
        self.left_recovery = left_recovery if left_recovery is not None else HandRecovery()
        self.right_recovery = right_recovery if right_recovery is not None else HandRecovery()
        self.min_emit_confidence = min_emit_confidence
        self._clock = clock
        self._wasd_held: set[str] = set()
        self._left_miss = 0
        self._right_miss = 0
        self._sprint_active = False
        self._look_accel_exponent = look_accel_exponent
        # Scroll repeat state: track last scroll time per direction for rate limiting.
        self._last_scroll_time: dict[str, float] = {}

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        emitter: InputEmitter | None = None,
        allow_uncalibrated_palm_normal: bool = False,
    ) -> Pipeline:
        """Build a pipeline from a :class:`Settings` model.

        Args:
            settings: Loaded configuration.
            emitter: Override emitter; if None, one is created from ``settings`` (a
                ``NullEmitter`` unless ``input.enabled`` is True).
            allow_uncalibrated_palm_normal: Allow missing palm-normal neutrals for safe
                dry-run preview. The first visible sample temporarily seeds neutral.

        Returns:
            A ready-to-run :class:`Pipeline`.
        """
        left_sm = GestureStateMachine("left", settings.gestures.left_hand)
        right_sm = GestureStateMachine("right", settings.gestures.right_hand)
        guard = TrackingLossGuard(left_sm, right_sm)
        j = settings.joystick
        joystick_signal: Callable[[np.ndarray], np.ndarray]
        left_joy: JoystickLike
        right_joy: JoystickLike
        if j.mode == "palm_tilt":
            t = j.tilt
            missing_tilt_neutral = t.left_neutral is None or t.right_neutral is None
            if missing_tilt_neutral and not allow_uncalibrated_palm_normal:
                raise ValueError(
                    "Palm-tilt joystick requires calibration before live input. Run "
                    "`mcv calibrate --apply` to write left/right neutral tilt vectors, or use "
                    "`mcv run --no-input --debug-overlay` for an uncalibrated preview."
                )
            left_joy = PalmNormalJoystick(
                t.left_neutral,
                t.deadzone,
                t.left_sensitivity,
                j.max_output,
                j.smoothing,
                sensitivity_neg=t.left_sensitivity_neg,
            )
            right_joy = PalmNormalJoystick(
                t.right_neutral,
                t.deadzone,
                t.right_sensitivity,
                j.max_output,
                j.smoothing,
                sensitivity_neg=t.right_sensitivity_neg,
            )
            joystick_signal = palm_tilt_xy
        elif j.mode == "palm_normal":
            pn = j.palm_normal
            missing_palm_neutral = pn.left_neutral is None or pn.right_neutral is None
            if missing_palm_neutral and not allow_uncalibrated_palm_normal:
                raise ValueError(
                    "Palm-normal joystick requires calibration before live input. Run "
                    "`mcv calibrate --apply` to write left/right neutral normals, or use "
                    "`mcv run --no-input --debug-overlay` for an uncalibrated preview."
                )
            left_joy = PalmNormalJoystick(
                pn.left_neutral,
                pn.deadzone,
                pn.left_sensitivity,
                j.max_output,
                j.smoothing,
                sensitivity_neg=pn.left_sensitivity_neg,
            )
            right_joy = PalmNormalJoystick(
                pn.right_neutral,
                pn.deadzone,
                pn.right_sensitivity,
                j.max_output,
                j.smoothing,
                sensitivity_neg=pn.right_sensitivity_neg,
            )
            joystick_signal = palm_normal_xy
        else:
            left_joy = WristRotationJoystick(
                j.deadzone_radius, j.sensitivity, j.max_output, j.smoothing
            )
            right_joy = WristRotationJoystick(
                j.deadzone_radius, j.sensitivity, j.max_output, j.smoothing
            )
            joystick_signal = palm_xz
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
            joystick_signal=joystick_signal,
            joystick_mode=j.mode,
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
            look_accel_exponent=j.look_accel_exponent,
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

        # Derive per-hand HUD status from recovery decisions.
        left_status: HandStatus = _hand_status(left_lm, left_dec)
        right_status: HandStatus = _hand_status(right_lm, right_dec)

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
            # In inventory mode, suppress new LEFT-hand gameplay actions (jump/sneak/
            # inventory/Q/F) so menu navigation can't fire them. KEY_UP still passes through so any
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
                left_status=left_status,
                right_status=right_status,
            )

        left_out = self._update_translation(left_lm, left_dec, now)
        right_out = self._update_look(right_lm, right_dec, now)

        # Collect debug signals for the HUD (cheap attribute reads; no allocation when overlay is off).
        left_sig = (
            self.joystick_signal(left_lm) if left_lm is not None else None
        )
        right_sig = (
            self.joystick_signal(right_lm) if right_lm is not None else None
        )

        return StepResult(
            events=events,
            left_output=left_out,
            right_output=right_out,
            wasd_held=frozenset(self._wasd_held),
            inventory_active=False,
            left_signal=left_sig,
            right_signal=right_sig,
            left_neutral=self.left_joystick.neutral.copy(),
            right_neutral=self.right_joystick.neutral.copy(),
            deadzone=getattr(self.left_joystick, "deadzone", 0.0),
            cardinal_half_width=self.cardinal_half_width,
            left_status=left_status,
            right_status=right_status,
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
        """Map the right hand to an absolute cursor position (inventory mode).

        In ``palm_tilt`` mode the cursor is a *tilt-to-point* pointer: the calibrated tilt
        deviation (signal minus the right hand's neutral, scaled by its per-axis sensitivity)
        is mapped about screen-center, so a full comfortable tilt spans the screen and a
        resting hand sits at center. This is precise from a still, resting hand — unlike the
        legacy mapping below, which keys off raw hand *position* and is twitchy when the hand
        barely moves. Other modes keep the legacy anchor-position mapping.

        Args:
            landmarks: ``(21, 3)`` right-hand landmarks, or ``None`` if absent this frame.

        Returns:
            The ``(2,)`` normalized screen position commanded this frame, or zeros if the hand
            is absent. Frame of reference: normalized screen coords (``y`` down), clamped to
            ``[0, 1]``.
        """
        if landmarks is None:
            return np.zeros(2, dtype=np.float64)
        if self.joystick_mode == "palm_tilt":
            signal = self.joystick_signal(landmarks)
            neutral = self.right_joystick.neutral
            sensitivity = self.right_joystick.sensitivity
            norm_delta = np.clip((signal[:2] - neutral) * sensitivity, -1.0, 1.0)
            screen = 0.5 + norm_delta * 0.5 * self.cursor_gain
        else:
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
            self.left_joystick.update(self.joystick_signal(landmarks))
            self._apply_wasd(set())
            return self.left_joystick.zero()
        out = self.left_joystick.update(self.joystick_signal(landmarks))
        self._update_sprint(landmarks, now)
        # Use cardinal-zone selection instead of independent per-axis sign checks.
        target = cardinal_keys(out, self.cardinal_half_width, self.bindings)
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
            self.right_joystick.update(self.joystick_signal(landmarks))
            if self.look_filter is not None:
                self.look_filter.reset()
            return self.right_joystick.zero()
        out = self.right_joystick.update(self.joystick_signal(landmarks))
        # Apply acceleration curve before the One-Euro filter so the filter smooths the
        # already-shaped signal (not the pre-shaped raw delta).
        max_out = getattr(self.right_joystick, "max_output", 1.0)
        out = accel_curve(out, self._look_accel_exponent, max_out)
        if self.look_filter is not None:
            # Velocity-adaptive smoothing: steady at rest, snappy in motion. (Task 4.)
            out = self.look_filter.filter(out, now)
        if out[0] != 0.0 or out[1] != 0.0:
            self.emitter.mouse_move(float(out[0]), float(out[1]))
        return out

    def _wasd_targets(self, output: np.ndarray) -> set[str]:
        """Translate a joystick output vector into the set of WASD keys to hold.

        Delegates to :func:`cardinal_keys` for angular zone selection; kept for backward
        compatibility with code that calls this method directly.
        """
        return cardinal_keys(output, self.cardinal_half_width, self.bindings)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hand_status(landmarks: np.ndarray | None, dec: RecoveryDecision) -> HandStatus:
    """Map recovery decision to a human-readable HUD status string."""
    if not dec.present or landmarks is None:
        return "absent"
    if not dec.emit:
        return "stabilizing"
    return "normal"


# ---------------------------------------------------------------------------
# Live capture loop
# ---------------------------------------------------------------------------


def run_pipeline(
    settings: Settings,
    source: FrameSource | None = None,
    allow_uncalibrated_palm_normal: bool = False,
) -> None:
    """Run the live capture loop until the source is exhausted or interrupted.

    Args:
        settings: Loaded configuration (camera, tracking, gestures, input, debug).
        source: Optional injected frame source. If None, a camera or clip source is built
            from ``settings`` (a clip would be wired by the CLI; default is the camera).
        allow_uncalibrated_palm_normal: Safe preview escape hatch for dry-runs with missing
            palm-normal calibration. Never enable this for real input emission.

    Notes:
        OpenCV (color convert / resize / overlay) is imported lazily here. HighGUI calls run
        on this (main) thread only. On any exit the emitter releases all held keys.
    """
    import cv2  # lazy: keeps this module importable without OpenCV (tests)

    from minecraft_cv.capture.buffer import FrameBuffer
    from minecraft_cv.capture.source import AVFoundationSource

    pipeline = Pipeline.from_settings(
        settings,
        allow_uncalibrated_palm_normal=allow_uncalibrated_palm_normal,
    )
    if source is None:
        source = AVFoundationSource(
            index=settings.camera.index,
            width=settings.camera.width,
            height=settings.camera.height,
            fps=settings.camera.fps,
        )

    tracker = HandTracker.create(settings.tracking.backend, settings.tracking.device)
    buffer = FrameBuffer(source).start()
    res_w, res_h = settings.tracking.input_resolution
    mirror = settings.camera.mirror
    overlay = True
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
        print(
            "Pipeline shutdown. "
            f"Processed {processed} frames in {t_elapsed:.2f}s "
            f"({fps:.1f} FPS), dropped {dropped} frames."
        )
        pipeline.shutdown()
        buffer.stop()
        tracker.close()
        if overlay:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Full debug HUD
# ---------------------------------------------------------------------------

# Joystick gizmo rendering scale (pixels per unit of joystick output).
_GIZMO_SCALE = 80
# Gizmo anchor offsets from the frame corners (pixels).
_GIZMO_MARGIN = 100

# Colour palette (BGR).
_COL_ACTIVE = (0, 255, 80)       # bright green — active key / live vector
_COL_IDLE = (80, 80, 80)         # dark grey — deadzone ring / idle elements
_COL_ZONE = (0, 180, 255)        # amber-yellow — cardinal zone wedge outline
_COL_ZONE_ACTIVE = (0, 255, 255) # bright yellow — active cardinal zone
_COL_WASD = (200, 200, 200)      # light grey — WASD key labels
_COL_WASD_ON = (0, 255, 80)      # green — pressed WASD key label
_COL_STATUS_OK = (0, 200, 80)    # green — TRACKING badge
_COL_STATUS_STAB = (0, 140, 255) # orange — STABILIZING badge
_COL_STATUS_ABSENT = (60, 60, 60) # dark grey — NO HAND badge
_COL_LOOK = (255, 120, 0)        # blue — look vector

_STATUS_COLOUR: dict[HandStatus, tuple[int, int, int]] = {
    "normal": _COL_STATUS_OK,
    "stabilizing": _COL_STATUS_STAB,
    "absent": _COL_STATUS_ABSENT,
}
_STATUS_LABEL: dict[HandStatus, str] = {
    "normal": "TRACKING",
    "stabilizing": "STABILIZING",
    "absent": "NO HAND",
}


def _draw_overlay(frame: np.ndarray, results: list[HandResult], step: StepResult) -> None:
    """Draw a full HUD onto ``frame`` (debug only, gated behind ``--debug-overlay``)."""
    import cv2
    import math as _math

    h, w = frame.shape[:2]

    # --- Landmark dots ----------------------------------------------------------
    for hand in results:
        for x, y, _ in hand.landmarks:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)

    # --- Gesture / WASD text (top-left) ----------------------------------------
    text = " ".join(f"{e.gesture}:{e.action}" for e in step.events) or "-"
    cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    wasd = "".join(sorted(step.wasd_held)) or "."
    cv2.putText(frame, f"WASD:{wasd}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    # --- Left-hand joystick gizmo (bottom-left) --------------------------------
    lx = _GIZMO_MARGIN
    ly = h - _GIZMO_MARGIN
    _draw_joystick_gizmo(
        frame,
        cx=lx,
        cy=ly,
        scale=_GIZMO_SCALE,
        signal=step.left_signal,
        neutral=step.left_neutral,
        output=np.array(step.left_output) if step.left_output is not None else None,
        deadzone=step.deadzone,
        half_width=step.cardinal_half_width,
        wasd_held=step.wasd_held,
        bindings_fwd_back_left_right=(
            step.wasd_held,  # passed as held set; gizmo knows the key names
        ),
        status=step.left_status,
        is_look=False,
    )

    # --- Right-hand look gizmo (bottom-right) ----------------------------------
    rx = w - _GIZMO_MARGIN
    ry = h - _GIZMO_MARGIN
    _draw_joystick_gizmo(
        frame,
        cx=rx,
        cy=ry,
        scale=_GIZMO_SCALE,
        signal=step.right_signal,
        neutral=step.right_neutral,
        output=np.array(step.right_output) if step.right_output is not None else None,
        deadzone=step.deadzone,
        half_width=step.cardinal_half_width,
        wasd_held=step.wasd_held,
        bindings_fwd_back_left_right=(step.wasd_held,),
        status=step.right_status,
        is_look=True,
    )


def _draw_joystick_gizmo(
    frame: np.ndarray,
    *,
    cx: int,
    cy: int,
    scale: int,
    signal: np.ndarray | None,
    neutral: np.ndarray | None,
    output: np.ndarray | None,
    deadzone: float,
    half_width: float,
    wasd_held: frozenset[str],
    bindings_fwd_back_left_right: tuple,
    status: HandStatus,
    is_look: bool,
) -> None:
    """Draw one joystick gizmo (deadzone ring, zone wedges, live vector, labels, status)."""
    import cv2
    import math as _math

    # --- Status badge -----------------------------------------------------------
    label = _STATUS_LABEL[status]
    col = _STATUS_COLOUR[status]
    cv2.putText(frame, label, (cx - 45, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # --- Neutral dot ------------------------------------------------------------
    cv2.circle(frame, (cx, cy), 4, (200, 200, 200), -1)

    # --- Deadzone ring ----------------------------------------------------------
    dz_px = int(deadzone * scale)
    if dz_px > 1:
        cv2.circle(frame, (cx, cy), dz_px, _COL_IDLE, 1)

    if is_look:
        # --- Look gizmo: simple arrow for the look vector -----------------------
        if output is not None and (output[0] != 0.0 or output[1] != 0.0):
            ox = int(output[0] * scale)
            oy = int(-output[1] * scale)  # screen y is inverted
            tip = (cx + ox, cy + oy)
            cv2.arrowedLine(frame, (cx, cy), tip, _COL_LOOK, 2, tipLength=0.3)
        return

    # --- WASD zone wedges (left-hand gizmo only) --------------------------------
    # Cardinals: right=0°, forward=90°, left=180°, back=270° (using math convention).
    # In screen coords y is down, so forward (+y joystick) points UP on screen.
    firing_radius = 90.0 - half_width
    cardinals = [
        ("forward", 90.0, "w"),
        ("left", 180.0, "a"),
        ("back", -90.0, "s"),
        ("right", 0.0, "d"),
    ]
    for _dir, angle_deg, key in cardinals:
        active = key in wasd_held
        color = _COL_ZONE_ACTIVE if active else _COL_ZONE
        # Draw two boundary rays at ±firing_radius around the cardinal centre.
        for offset in (-firing_radius, firing_radius):
            ray_deg = angle_deg + offset
            # Screen convention: x right, y down; forward (+y) maps to screen-up (-y).
            rx = _math.cos(_math.radians(ray_deg)) * scale
            ry = -_math.sin(_math.radians(ray_deg)) * scale  # flip y for screen
            end = (int(cx + rx), int(cy + ry))
            cv2.line(frame, (cx, cy), end, color, 1)

    # --- Live deviation vector arrow -------------------------------------------
    if signal is not None and neutral is not None:
        dev = signal[:2] - neutral[:2]
        dx = int(dev[0] * scale)
        dy = int(-dev[1] * scale)  # flip y
        # Clamp to gizmo area
        mag = _math.sqrt(dx * dx + dy * dy)
        max_px = scale
        if mag > max_px:
            dx = int(dx * max_px / mag)
            dy = int(dy * max_px / mag)
        tip = (cx + dx, cy + dy)
        is_active = output is not None and (output[0] != 0.0 or output[1] != 0.0)
        col = _COL_ACTIVE if is_active else (160, 160, 160)
        cv2.arrowedLine(frame, (cx, cy), tip, col, 2, tipLength=0.25)

    # --- WASD key labels around gizmo -----------------------------------------
    label_dist = scale + 16
    label_positions = {
        "w": (cx, cy - label_dist),
        "a": (cx - label_dist, cy),
        "s": (cx, cy + label_dist),
        "d": (cx + label_dist, cy),
    }
    for key, (kx, ky) in label_positions.items():
        col = _COL_WASD_ON if key in wasd_held else _COL_WASD
        cv2.putText(frame, key.upper(), (kx - 5, ky + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)


__all__ = ["Pipeline", "StepResult", "run_pipeline"]
