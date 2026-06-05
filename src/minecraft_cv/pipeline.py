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
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from minecraft_cv.gestures.pinch import KEY_DOWN
from minecraft_cv.gestures.registry import GestureStateMachine
from minecraft_cv.gestures.safety import AnyGestureEvent, TrackingLossGuard
from minecraft_cv.input.emitter import InputEmitter, create_emitter
from minecraft_cv.joystick.one_euro import OneEuroFilter
from minecraft_cv.joystick.screen import ScreenJoystick, screen_index_mcp
from minecraft_cv.joystick.steering import octant_keys
from minecraft_cv.joystick.wrist_tilt import WristTiltJoystick, wrist_tilt_vector
from minecraft_cv.recovery import HandRecovery, RecoveryDecision
from minecraft_cv.tracking.tracker import HandResult

if TYPE_CHECKING:
    from minecraft_cv.capture.source import FrameSource
    from minecraft_cv.config import Settings
    from minecraft_cv.gestures.face_gestures import FaceGestureStateMachine


# Per-hand tracking status for the HUD (readable name, not a bool).
HandStatus = Literal["normal", "stabilizing", "absent"]


class JoystickLike(Protocol):
    """Shared surface for palm-normal and legacy wrist-rotation joysticks."""

    @property
    def neutral(self) -> np.ndarray | None: ...
    @property
    def sensitivity(self) -> np.ndarray: ...
    @property
    def deadzone(self) -> float: ...
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
    left_gestures: frozenset[str] = field(default_factory=frozenset)
    """Left-hand logical gestures currently held by the detector layer."""
    right_gestures: frozenset[str] = field(default_factory=frozenset)
    """Right-hand logical gestures currently held by the detector layer."""
    face_gestures: frozenset[str] = field(default_factory=frozenset)
    """Face logical gestures currently held by the detector layer."""
    relocalized_hands: frozenset[str] = field(default_factory=frozenset)
    """Hands whose joystick neutral was relocalized on this frame."""

    # Debug-only fields (populated when the overlay is drawn; zero/None otherwise).
    # These never affect emission (hard invariant #2).
    left_signal: np.ndarray | None = None
    """Raw tilt/normal signal ``(x, y)`` from the left hand this frame."""
    right_signal: np.ndarray | None = None
    """Raw tilt/normal signal ``(x, y)`` from the right hand this frame."""
    left_neutral: np.ndarray | None = None
    """Left joystick neutral ``(x, y)`` at the time of this step."""
    right_neutral: np.ndarray | None = None
    """Current right thumb cursor point ``(x, y)`` at the time of this step."""
    deadzone: float = 0.0
    """Deadzone radius used by the left joystick (for overlay ring)."""
    left_status: HandStatus = "absent"
    """Tracking state of the left hand: ``normal``, ``stabilizing``, or ``absent``."""
    right_status: HandStatus = "absent"
    """Tracking state of the right hand: ``normal``, ``stabilizing``, or ``absent``."""
    face_status: Literal["tracking", "absent"] = "absent"
    """Tracking state of the face."""


class Pipeline:
    """Stateful per-frame controller: gestures + spatial joysticks -> input emitter."""

    def __init__(
        self,
        emitter: InputEmitter,
        bindings: dict[str, str],
        left_joystick: JoystickLike,
        right_joystick: JoystickLike,
        left_sm: GestureStateMachine,
        right_sm: GestureStateMachine,
        face_sm: FaceGestureStateMachine | None = None,
        guard: TrackingLossGuard | None = None,
        left_joystick_signal: Callable[[np.ndarray], np.ndarray] = wrist_tilt_vector,
        right_joystick_signal: Callable[[np.ndarray], np.ndarray] = screen_index_mcp,
        swap_handedness: bool = True,
        scroll_repeat_rate_hz: float = 8.0,
        look_filter: OneEuroFilter | None = None,
        left_recovery: HandRecovery | None = None,
        right_recovery: HandRecovery | None = None,
        min_emit_confidence: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
        look_accel_exponent: float = 1.6,
    ) -> None:
        """Initialize the pipeline with all constituent processors."""
        self.emitter = emitter
        self.bindings = bindings
        self.left_joystick = left_joystick
        self.right_joystick = right_joystick
        self.left_sm = left_sm
        self.right_sm = right_sm
        self.face_sm = face_sm
        self.guard = guard or TrackingLossGuard(left_sm, right_sm)
        self.left_joystick_signal = left_joystick_signal
        self.right_joystick_signal = right_joystick_signal
        self.recenter_grace_frames = 3
        self.swap_handedness = swap_handedness
        self.scroll_repeat_rate_hz = scroll_repeat_rate_hz
        self.look_filter = look_filter
        self.left_recovery = left_recovery if left_recovery is not None else HandRecovery()
        self.right_recovery = right_recovery if right_recovery is not None else HandRecovery()
        self.min_emit_confidence = min_emit_confidence
        self._clock = clock
        self._wasd_held: set[str] = set()
        self._left_miss = 0
        self._right_miss = 0
        self._right_cursor_prev: np.ndarray | None = None
        self._look_accel_exponent = look_accel_exponent
        self._last_scroll_time: dict[str, float] = {}

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        emitter: InputEmitter | None = None,
        allow_uncalibrated_palm_normal: bool = False,
    ) -> Pipeline:
        """Build a pipeline from a :class:`Settings` model."""
        left_sm = GestureStateMachine("left", settings.gestures.left_hand)
        right_sm = GestureStateMachine("right", settings.gestures.right_hand)

        face_sm = None
        if hasattr(settings, "face_tracking") and settings.face_tracking.enabled:
            from minecraft_cv.gestures.face_gestures import FaceGestureStateMachine
            face_sm = FaceGestureStateMachine(
                settings.gestures.face,
                head_roll=settings.gestures.head_tilt,
                head_pitch=settings.gestures.head_pitch,
            )

        j = settings.joystick
        left_joy = WristTiltJoystick(
            deadzone_deg=j.deadzone,
            sensitivity=j.left_sensitivity,
            smoothing=j.smoothing,
        )
        right_joy = ScreenJoystick(
            j.deadzone,
            j.right_sensitivity,
            j.right_smoothing if j.right_smoothing is not None else j.smoothing,
            fixed_neutral=j.fixed_right_neutral,
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
        tr = settings.tracking
        return cls(
            emitter=emitter if emitter is not None else create_emitter(settings),
            bindings=dict(settings.bindings),
            left_joystick=left_joy,
            right_joystick=right_joy,
            left_sm=left_sm,
            right_sm=right_sm,
            face_sm=face_sm,
            left_joystick_signal=wrist_tilt_vector,
            right_joystick_signal=screen_index_mcp,
            swap_handedness=settings.tracking.swap_handedness,
            scroll_repeat_rate_hz=settings.input.scroll_repeat_rate_hz,
            look_filter=look_filter,
            left_recovery=HandRecovery(tr.dropout_flush_ms, tr.stabilization_ms),
            right_recovery=HandRecovery(tr.dropout_flush_ms, tr.stabilization_ms),
            min_emit_confidence=tr.min_emit_confidence,
            look_accel_exponent=j.look_accel_exponent,
        )

    # --- per-frame logic ----------------------------------------------------
    def step(self, hand_results: list[HandResult], face_result: Any | None = None) -> StepResult:
        """Process one frame of tracking data.

        Args:
            hand_results: List of detected hands.
            face_result: Optional FaceResult.

        Returns:
            A snapshot of the system state for telemetry/HUD.
        """
        now = self._clock()
        left_lm, right_lm = self._split(hand_results)

        # Tracking-loss recovery (Task 5): decide per hand whether to emit, track, or flush.
        left_dec = self.left_recovery.update(left_lm is not None, now)
        right_dec = self.right_recovery.update(right_lm is not None, now)
        if left_dec.flush:
            self._flush_left()
        if right_dec.flush:
            self._flush_right()

        # Feed each hand's landmarks to its gesture machine only when that hand may emit
        # (NORMAL phase).
        raw_events = self.guard.process(
            left_lm if left_dec.emit else None,
            right_lm if right_dec.emit else None,
        )

        events = list(raw_events)
        relocalized_hands: set[str] = set()

        # Update face gestures if provided
        if self.face_sm and face_result is not None:
            face_events = self.face_sm.update(face_result)
            events.extend(face_events)

        for event in events:
            if event.gesture == "recenter" and event.action == KEY_DOWN:
                if event.hand == "left" and left_lm is not None:
                    self._relocalize_left(left_lm)
                    relocalized_hands.add("left")
                elif event.hand == "right" and right_lm is not None:
                    self._relocalize_right(right_lm)
                    relocalized_hands.add("right")
                continue

            binding = self.bindings.get(event.gesture)
            if binding is None:
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

        # Collect debug signals for the HUD.
        left_held = self.guard.left_held
        right_held = self.guard.right_held
        # WASD now comes from the left-hand pinch gestures, not the (removed) tilt-joystick.
        # Derive the HUD's movement-key set from the held left gestures' bindings.
        wasd_held = frozenset(
            key
            for g in left_held
            if (key := self.bindings.get(g)) in ("w", "a", "s", "d")
        )
        left_sig = self.left_joystick_signal(left_lm) if left_lm is not None else None
        right_sig = self.right_joystick_signal(right_lm) if right_lm is not None else None

        left_out = self._update_translation(left_lm, left_dec, now)
        right_out = self._update_look(
            right_lm,
            right_dec,
            now,
            suppress_emit="recenter" in right_held,
        )

        right_neutral = (
            self.right_joystick.neutral.copy()
            if self.right_joystick.neutral is not None
            else None
        )

        return StepResult(
            events=events,
            left_output=left_out,
            right_output=right_out,
            wasd_held=wasd_held,
            relocalized_hands=frozenset(relocalized_hands),
            left_gestures=left_held,
            right_gestures=right_held,
            face_gestures=self.face_sm.active_gestures() if self.face_sm else frozenset(),
            left_signal=left_sig,
            right_signal=right_sig,
            left_neutral=self.left_joystick.neutral,
            right_neutral=right_neutral,
            deadzone=self.left_joystick.deadzone,
            left_status=_hand_status(left_lm, left_dec),
            right_status=_hand_status(right_lm, right_dec),
            face_status=self.face_sm.status() if self.face_sm else "absent",
        )

    def _relocalize_left(self, landmarks: np.ndarray) -> None:
        """Recenter movement at the current left hand and clear movement state."""
        self._apply_wasd(set())
        self.left_joystick.recenter_at(self.left_joystick_signal(landmarks))

    def _relocalize_right(self, landmarks: np.ndarray) -> None:
        """Reset the right cursor point and emit no mouse movement."""
        self.emitter.mouse_stop()
        self._seed_right_cursor(self.right_joystick_signal(landmarks))
        if self.look_filter is not None:
            self.look_filter.reset()

    # --- tracking-loss flush helpers (Task 5) -------------------------------
    def _flush_left(self) -> None:
        """Hard-flush the left hand after a sustained dropout."""
        self.left_joystick.reset_neutral()

    def _flush_right(self) -> None:
        """Hard-flush the right hand: clear cursor history and stop mouse output."""
        self.emitter.mouse_stop()
        self._right_cursor_prev = None
        self.right_joystick.reset_neutral()
        if self.look_filter is not None:
            self.look_filter.reset()

    def _repeat_scroll(self, now: float) -> None:
        """Re-emit scroll ticks for held hotbar gestures at the configured repeat rate."""
        if not self._last_scroll_time or self.scroll_repeat_rate_hz <= 0:
            return
        interval = 1.0 / self.scroll_repeat_rate_hz
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
        # WASD is no longer driven by the left tilt-joystick — it comes entirely from the
        # left-hand pinch gestures (move_forward/back/left/right) through the gesture-event
        # path, which the TrackingLossGuard releases on dropout. The joystick is kept only to
        # produce a HUD signal; it never presses keys. ``_apply_wasd`` stays available so
        # ``shutdown()`` keeps its fail-safe, but nothing populates ``_wasd_held`` anymore.
        if not dec.present or landmarks is None:
            self._left_miss += 1
            if self._left_miss >= self.recenter_grace_frames:
                self.left_joystick.reset_neutral()
            return self.left_joystick.zero()
        self._left_miss = 0
        if not dec.emit:
            # Stabilizing on re-entry: feed coords so the HUD neutral re-seeds.
            self.left_joystick.update(self.left_joystick_signal(landmarks))
            return self.left_joystick.zero()
        return self.left_joystick.update(self.left_joystick_signal(landmarks))

    def _update_look(
        self,
        landmarks: np.ndarray | None,
        dec: RecoveryDecision,
        now: float,
        *,
        suppress_emit: bool = False,
    ) -> np.ndarray:
        if not dec.present or landmarks is None:
            self._right_miss += 1
            self._right_cursor_prev = None
            if self._right_miss >= self.recenter_grace_frames:
                self.right_joystick.reset_neutral()
            # Drop cursor history so re-entry does not jump from a stale thumb point.
            if self.look_filter is not None:
                self.look_filter.reset()
            self.emitter.mouse_stop()
            return self.right_joystick.zero()
        self._right_miss = 0
        if suppress_emit:
            # Peace sign is the "mouse lifted" clutch: keep moving the neutral to the
            # cursor point while held, but never send look/click movement from the right hand.
            self._seed_right_cursor(self.right_joystick_signal(landmarks))
            if self.look_filter is not None:
                self.look_filter.reset()
            self.emitter.mouse_stop()
            return self.right_joystick.zero()
        if not dec.emit:
            # Stabilizing: re-seed the neutral, keep the filter clear, emit no mouse-look.
            self._seed_right_cursor(self.right_joystick_signal(landmarks))
            if self.look_filter is not None:
                self.look_filter.reset()
            self.emitter.mouse_stop()
            return self.right_joystick.zero()
        signal = self.right_joystick_signal(landmarks)
        # Smooth the cursor *position* with the velocity-adaptive One-Euro filter before
        # differencing, then drive the camera from the motion of the smoothed point. Raw
        # frame-to-frame cursor deltas carry MediaPipe landmark jitter and frame-rate hitches
        # straight to the mouse, which reads as a sputtery "start-stop" look. One-Euro cuts
        # that jitter hard at rest yet barely lags fast looks. Because successive filtered
        # positions telescope, the total emitted motion still equals the cursor's true
        # displacement (unity DC gain): the same look, merely spread smoothly across a few
        # frames so it glides to a stop instead of stuttering frame by frame.
        if self.look_filter is not None:
            signal = self.look_filter.filter(signal, now)
        if self._right_cursor_prev is None:
            self._seed_right_cursor(signal)
            self.emitter.mouse_stop()
            return self.right_joystick.zero()

        out = (signal - self._right_cursor_prev) * self.right_joystick.sensitivity
        self._seed_right_cursor(signal)

        # Apply exponential acceleration for smoother micro-movements and faster flicks
        mag = np.linalg.norm(out)
        if mag > 0 and self._look_accel_exponent != 1.0:
            # Scale magnitude exponentially. A scale factor ensures 1.0 px movement stays 1.0.
            # Using absolute coordinates for the exponent so small moves (<1) get smaller,
            # and large moves get larger.
            out = out * ((mag ** self._look_accel_exponent) / mag)

        if out[0] != 0.0 or out[1] != 0.0:
            self.emitter.mouse_move(float(out[0]), float(out[1]))
        else:
            self.emitter.mouse_stop()
        return out

    def _seed_right_cursor(self, signal: np.ndarray) -> None:
        """Set the right hand's current cursor point without emitting movement."""
        cursor = np.asarray(signal, dtype=np.float64)[:2]
        self._right_cursor_prev = cursor.copy()
        # Keep the existing HUD/relocalization surface pointed at the current cursor point.
        self.right_joystick.recenter_at(cursor)

    def _wasd_targets(self, output: np.ndarray) -> set[str]:
        return octant_keys(output, self.bindings)

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
        self._last_scroll_time.clear()
        self.emitter.release_all()

    def recenter(self) -> None:
        """Recenter both spatial joysticks at the current hand position on the next frame.

        The screen-space joysticks re-seed their neutral from the first sample after a reset,
        so clearing the neutrals here makes the *next* processed frame adopt the hand's current
        rest pose as center — the manual equivalent of the peace-sign recenter macro. Held
        movement keys and mouse motion are released first so nothing is stranded. Used by the
        desktop app's "Calibrate" button.
        """
        self._apply_wasd(set())
        self.left_joystick.reset_neutral()
        self._right_cursor_prev = None
        self.right_joystick.reset_neutral()
        self.emitter.mouse_stop()
        if self.look_filter is not None:
            self.look_filter.reset()

    def set_emitter(self, emitter: InputEmitter) -> None:
        """Swap the OS-input emitter at runtime (e.g. the Dry-Run <-> Live toggle).

        Releases everything currently held on the old emitter first (via :meth:`shutdown`), so
        toggling never leaves a key stuck down. Detector/joystick state is reset; the user
        re-engages gestures after the swap.

        Args:
            emitter: The new emitter to drive (``NullEmitter`` for Dry-Run, the macOS emitter
                for Live).
        """
        self.shutdown()
        self.emitter = emitter


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

    from minecraft_cv.runtime import FrameProcessor

    processor = FrameProcessor.from_settings(
        settings,
        source=source,
        allow_uncalibrated_palm_normal=allow_uncalibrated_palm_normal,
    ).start()

    overlay = settings.debug.overlay
    overlay_every = max(1, settings.debug.overlay_every)
    window = "minecraft_cv"
    decimation = 0
    relocalized_flash_until: dict[str, float] = {}

    try:
        while True:
            packet = processor.process_once()
            if packet is None:
                if processor.exhausted:
                    break
                time.sleep(0.001)
                continue

            decimation += 1
            step = packet.step
            now = time.monotonic()
            if step.relocalized_hands:
                for hand in step.relocalized_hands:
                    relocalized_flash_until[hand] = now + 0.75
            flash_hands = frozenset(
                hand for hand, until in relocalized_flash_until.items() if until >= now
            )

            # Overlay is a debug-only luxury and not free; decimate the (HighGUI) draw to
            # protect the real-time loop. Tracking/gestures/input still run every frame.
            if overlay and (decimation % overlay_every == 0 or step.relocalized_hands):
                _draw_overlay(
                    packet.frame,
                    list(packet.hands),
                    step,
                    live_input=settings.input.enabled,
                    flash_hands=flash_hands,
                )
                cv2.imshow(window, packet.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if processor.exhausted:
                break
    finally:
        t_elapsed = processor.elapsed
        fps = processor.processed / t_elapsed if t_elapsed > 0 else 0.0
        print(
            "Pipeline shutdown. "
            f"Processed {processor.processed} frames in {t_elapsed:.2f}s "
            f"({fps:.1f} FPS), dropped {processor.dropped} frames."
        )
        processor.shutdown()
        if overlay:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Full debug HUD
# ---------------------------------------------------------------------------

# Colour palette (BGR).
_COL_PANEL = (24, 28, 32)
_COL_PANEL_BORDER = (92, 102, 112)
_COL_TEXT = (238, 242, 245)
_COL_MUTED = (148, 158, 168)
_COL_ACTIVE = (80, 245, 132)
_COL_IDLE = (80, 88, 96)
_COL_WARN = (0, 128, 255)
_COL_LIVE = (52, 82, 255)
_COL_DRY = (80, 210, 120)
_COL_FLASH = (0, 255, 255)
_COL_LOOK = (255, 150, 42)
_COL_MOVE = (86, 218, 255)

_STATUS_COLOUR: dict[HandStatus, tuple[int, int, int]] = {
    "normal": _COL_ACTIVE,
    "stabilizing": _COL_WARN,
    "absent": _COL_IDLE,
}
_STATUS_LABEL: dict[HandStatus, str] = {
    "normal": "OK",
    "stabilizing": "SETTLING",
    "absent": "NO HAND",
}

# The left hand now drives WASD only — shown in the WASD badge row, not a command column.
# The first column shows face/head-roll gestures; the second shows the right hand.
_FACE_COMMANDS = (
    ("Throw", "throw_item"),
    ("Inventory", "inventory"),
    ("Swap", "swap_offhand"),
    ("Hotbar +", "hotbar_next"),
    ("Hotbar -", "hotbar_prev"),
)
_RIGHT_COMMANDS = (
    ("Attack", "attack"),
    ("Use", "use"),
    ("Jump", "jump"),
    ("Sneak", "sneak"),
    ("Relocalize", "recenter"),
)


def _draw_overlay(
    frame: np.ndarray,
    results: list[HandResult],
    step: StepResult,
    *,
    live_input: bool,
    flash_hands: frozenset[str],
) -> None:
    """Draw a full HUD onto ``frame`` (debug only, gated behind ``--debug-overlay``)."""
    import cv2

    h, w = frame.shape[:2]

    for hand in results:
        for x, y, _ in hand.landmarks:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, _COL_ACTIVE, -1)

    _draw_command_panel(frame, step, live_input=live_input, flash_hands=flash_hands)

    if step.left_neutral is not None:
        _draw_joystick_gizmo(
            frame,
            w=w,
            h=h,
            signal=step.left_signal,
            neutral=step.left_neutral,
            output=np.array(step.left_output) if step.left_output is not None else None,
            deadzone=step.deadzone,
            wasd_held=step.wasd_held,
            status=step.left_status,
            label="MOVE",
            relocalized="left" in flash_hands,
            is_look=False,
        )

    if step.right_neutral is not None:
        _draw_joystick_gizmo(
            frame,
            w=w,
            h=h,
            signal=step.right_signal,
            neutral=step.right_neutral,
            output=np.array(step.right_output) if step.right_output is not None else None,
            deadzone=0.0,
            wasd_held=step.wasd_held,
            status=step.right_status,
            label="CURSOR",
            relocalized="right" in flash_hands,
            is_look=True,
        )


def _draw_command_panel(
    frame: np.ndarray,
    step: StepResult,
    *,
    live_input: bool,
    flash_hands: frozenset[str],
) -> None:
    """Draw the player-facing command state panel."""
    import cv2

    x, y = 10, 10
    width, height = 330, 260
    frame_h, frame_w = frame.shape[:2]
    width = min(width, frame_w - 20)
    height = min(height, frame_h - 20)
    _alpha_rect(frame, x, y, x + width, y + height, _COL_PANEL, 0.78)
    cv2.rectangle(frame, (x, y), (x + width, y + height), _COL_PANEL_BORDER, 1)

    mode_text = "LIVE INPUT" if live_input else "DRY RUN"
    mode_col = _COL_LIVE if live_input else _COL_DRY
    _draw_text(frame, mode_text, x + 12, y + 24, 0.56, mode_col, 2)
    status = f"L {_STATUS_LABEL[step.left_status]}   R {_STATUS_LABEL[step.right_status]}"
    _draw_text(frame, status, x + 132, y + 24, 0.43, _COL_TEXT, 1)

    if flash_hands:
        label = "RELOCALIZED " + "/".join(hand.upper()[0] for hand in sorted(flash_hands))
        _draw_text(frame, label, x + 12, y + 48, 0.48, _COL_FLASH, 2)
    else:
        latest = " ".join(f"{e.gesture}:{e.action.replace('KEY_', '')}" for e in step.events)
        _draw_text(frame, latest[:34] or "ready", x + 12, y + 48, 0.43, _COL_MUTED, 1)

    _draw_text(frame, "WASD", x + 12, y + 75, 0.45, _COL_MUTED, 1)
    for i, key in enumerate(("w", "a", "s", "d")):
        _draw_badge(frame, x + 58 + i * 34, y + 59, key.upper(), key in step.wasd_held)

    _draw_command_column(
        frame,
        "FACE",
        _FACE_COMMANDS,
        step.face_gestures,
        x + 12,
        y + 104,
        flash=False,
    )
    _draw_command_column(
        frame,
        "RIGHT",
        _RIGHT_COMMANDS,
        step.right_gestures,
        x + 172,
        y + 104,
        flash="right" in flash_hands,
    )


def _draw_command_column(
    frame: np.ndarray,
    title: str,
    commands: tuple[tuple[str, str], ...],
    held: frozenset[str],
    x: int,
    y: int,
    *,
    flash: bool,
) -> None:
    """Draw one compact command list with active indicators."""
    import cv2

    _draw_text(frame, title, x, y, 0.43, _COL_MUTED, 1)
    row_y = y + 20
    for label, gesture in commands:
        active = gesture in held or (gesture == "recenter" and flash)
        col = _COL_FLASH if gesture == "recenter" and flash else _COL_ACTIVE
        cv2.circle(frame, (x + 6, row_y - 4), 5, col if active else _COL_IDLE, -1)
        _draw_text(frame, label, x + 18, row_y, 0.41, _COL_TEXT if active else _COL_MUTED, 1)
        row_y += 21


def _draw_badge(frame: np.ndarray, x: int, y: int, text: str, active: bool) -> None:
    """Draw a small fixed-size key badge."""
    import cv2

    col = _COL_ACTIVE if active else _COL_IDLE
    cv2.rectangle(frame, (x, y), (x + 26, y + 22), col, 1)
    if active:
        _alpha_rect(frame, x + 1, y + 1, x + 25, y + 21, col, 0.30)
    _draw_text(frame, text, x + 7, y + 16, 0.42, _COL_TEXT if active else _COL_MUTED, 1)


def _draw_joystick_gizmo(
    frame: np.ndarray,
    *,
    w: int,
    h: int,
    signal: np.ndarray | None,
    neutral: np.ndarray | None,
    output: np.ndarray | None,
    deadzone: float,
    wasd_held: frozenset[str],
    status: HandStatus,
    label: str,
    relocalized: bool,
    is_look: bool,
) -> None:
    """Draw one joystick gizmo exactly at the physical anchor point on the screen."""
    import math as _math

    import cv2

    if neutral is None:
        return

    cx = int(np.clip(neutral[0], 0.0, 1.0) * w)
    cy = int(np.clip(neutral[1], 0.0, 1.0) * h)

    dz_px_x = int(deadzone * w)
    dz_px_y = int(deadzone * h)
    ring_col = _COL_FLASH if relocalized else (_COL_LOOK if is_look else _COL_MOVE)
    radius = (max(dz_px_x, dz_px_y) if deadzone > 0.0 else 16) + (
        18 if relocalized else 8
    )

    if relocalized:
        cv2.circle(frame, (cx, cy), radius, _COL_FLASH, 3)

    if deadzone > 0.0:
        cv2.ellipse(frame, (cx, cy), (dz_px_x, dz_px_y), 0, 0, 360, _COL_IDLE, 1)
    cv2.circle(frame, (cx, cy), 6, _COL_TEXT, -1)
    _draw_text(frame, label, cx - 20, cy - radius - 10, 0.48, ring_col, 2)
    _draw_text(
        frame,
        _STATUS_LABEL[status],
        cx - 30,
        cy + radius + 20,
        0.42,
        _STATUS_COLOUR[status],
        1,
    )

    if not is_look:
        ray_len_x = int(0.12 * w)
        ray_len_y = int(0.12 * h)
        for i in range(8):
            ray_deg = 22.5 + i * 45.0
            rx = _math.cos(_math.radians(ray_deg)) * ray_len_x
            ry = _math.sin(_math.radians(ray_deg)) * ray_len_y
            cv2.line(frame, (cx, cy), (int(cx + rx), int(cy + ry)), (75, 105, 110), 1)

    if signal is not None:
        hx = int(signal[0] * w)
        hy = int(signal[1] * h)
        active = output is not None and float(np.linalg.norm(output[:2])) > 1e-6
        line_col = ring_col if active else _COL_MUTED
        cv2.line(frame, (cx, cy), (hx, hy), line_col, 2)
        cv2.circle(frame, (hx, hy), 5, line_col, -1)

    if output is not None and float(np.linalg.norm(output[:2])) > 1e-6:
        ox = int(output[0] * 74)
        oy = int(output[1] * 74)
        cv2.arrowedLine(frame, (cx, cy), (cx + ox, cy + oy), ring_col, 3, tipLength=0.28)

    if not is_look:
        label_dist_x = int(0.14 * w)
        label_dist_y = int(0.14 * h)
        positions = {
            "w": (cx - 13, cy - label_dist_y - 11),
            "a": (cx - label_dist_x - 13, cy - 11),
            "s": (cx - 13, cy + label_dist_y - 11),
            "d": (cx + label_dist_x - 13, cy - 11),
        }
        for key, (px, py) in positions.items():
            _draw_badge(frame, px, py, key.upper(), key in wasd_held)


def _alpha_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Blend a solid rectangle over ``frame`` in-place."""
    x1 = max(0, min(frame.shape[1], x1))
    x2 = max(0, min(frame.shape[1], x2))
    y1 = max(0, min(frame.shape[0], y1))
    y2 = max(0, min(frame.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    tint = np.full_like(roi, color)
    import cv2

    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _draw_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw readable stroked text."""
    import cv2

    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


__all__ = ["Pipeline", "StepResult", "run_pipeline"]
