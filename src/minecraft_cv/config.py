"""Typed configuration for minecraft_cv.

Loads ``config.yaml`` into a validated pydantic ``Settings`` model. Every tunable value
lives here so gesture/joystick/input code never holds magic numbers. The most important
guarantee enforced at construction time is hard-invariant #1: ``t_release > t_engage`` for
the lower-is-engaged detector gestures (pinches and curled-finger detectors).

Pinch distances are normalized by hand scale, and calibrated palm-normal joystick signals
are translation-invariant, so thresholds stay stable as the hand moves in the camera frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

FingerName = Literal["index", "middle", "ring", "pinky", "fist"]
DetectorFingerName = Literal["thumb", "index", "middle", "ring", "pinky"]
CurlFingerName = Literal["index", "middle", "ring", "pinky"]
GestureDetectorName = Literal["pinch", "curl_only", "curl_combo", "extension_combo"]
GestureMode = Literal["hold", "toggle"]
Anchor = Literal["wrist", "middle_mcp"]
JoystickMode = Literal["palm_tilt", "palm_normal", "wrist_rotation"]
ExtensionGestureType = Literal[
    "thumb_out", "index_only", "middle_only", "index_middle", "ring_only", "pinky_only"
]


class CameraSettings(BaseModel):
    """Hardware capture parameters for ``cv2.VideoCapture``."""

    model_config = {"extra": "forbid"}

    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "avfoundation"
    mirror: bool = True
    """Horizontally flip each frame so the view behaves like a mirror.

    A webcam pointed at the user is *not* mirrored: moving a hand right makes it travel left
    on screen, which is disorienting and inverts left-hand WASD. Flipping before tracking also
    fixes MediaPipe handedness, which is labelled assuming a mirrored (selfie) image.
    """


class TrackingSettings(BaseModel):
    """Hand-tracking backend selection and inference parameters."""

    model_config = {"extra": "forbid"}

    backend: str = "mediapipe"
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    input_resolution: tuple[int, int] = (256, 256)
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_num_hands: int = Field(default=2, ge=1, le=2)
    swap_handedness: bool = True
    """Invert MediaPipe Left/Right handedness labels.

    With ``mirror: true``, the user's physical left hand appears on the left side of the
    image. MediaPipe may label this as ``"Left"`` (matching image convention) but the pipeline
    expects the user's physical left hand to drive the left-hand gestures/WASD. If the labels
    come out swapped, enabling this flag fixes it.
    """

    # --- Tracking-loss recovery (Task 5) ----------------------------------------------
    min_emit_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    """Drop any detected hand whose handedness score is below this before it reaches the
    pipeline. ``0.0`` (default) keeps every hand. Raising it treats low-confidence detections
    as *absent*, feeding them into the same fail-safe/recovery path as a true dropout."""
    dropout_flush_ms: float = Field(default=100.0, ge=0.0)
    """If a hand is continuously absent for longer than this, perform a *hard flush*: release
    every key that hand could hold and reset its joystick neutral + look filter. (Per-frame
    key release still happens immediately; this governs the heavier recenter/flush.)"""
    stabilization_ms: float = Field(default=500.0, ge=0.0)
    """After a hard flush, suppress *all* input from a returning hand for this long while its
    coordinates are tracked to re-establish a neutral origin — prevents a violent camera snap
    on re-entry. ``0`` disables the stabilization window."""


class FaceTrackerSettings(BaseModel):
    """MediaPipe FaceLandmarker parameters."""

    model_config = {"extra": "forbid"}

    enabled: bool = True
    model_path: str = "models/face_landmarker.task"
    device: Literal["auto", "mps", "cuda", "cpu"] = "cpu"
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class GestureThresholds(BaseModel):
    """Schmitt-trigger thresholds for one discrete pinch gesture (right hand).

    Attributes:
        finger: Which fingertip's thumb-pinch drives this gesture.
        t_engage: Normalized distance below which the gesture engages (KEY_DOWN).
        t_release: Normalized distance above which it releases (KEY_UP). Must be
            strictly greater than ``t_engage`` (the hysteresis band that swallows jitter).
    """

    model_config = {"extra": "forbid"}

    finger: FingerName
    t_engage: float = Field(gt=0.0)
    t_release: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check_hysteresis(self) -> GestureThresholds:
        if not self.t_release > self.t_engage:
            raise ValueError(
                f"t_release ({self.t_release}) must be strictly greater than "
                f"t_engage ({self.t_engage}); equal/inverted thresholds reintroduce the "
                "chatter the Schmitt trigger exists to prevent (hard invariant #1)."
            )
        return self


class ExtensionThresholds(BaseModel):
    """Schmitt-trigger thresholds for one finger-extension gesture (left hand).

    Extension gestures detect which fingers are extended (straightened) vs. curled
    (closed into fist). Unlike pinch gestures where *lower* distance engages, extension
    gestures engage when the extension ratio goes *above* the engage threshold.

    Attributes:
        type: The finger combination pattern that triggers this gesture.
        t_engage: Extension ratio above which the gesture engages (KEY_DOWN).
        t_release: Extension ratio below which it releases (KEY_UP). Must be
            strictly less than ``t_engage`` (inverted Schmitt band for extension).
        pulse: If True, this gesture fires a single key tap on engage rather than
            a sustained hold. Used for toggle/one-shot actions.
    """

    model_config = {"extra": "forbid"}

    type: ExtensionGestureType
    t_engage: float = Field(gt=0.0)
    t_release: float = Field(gt=0.0)
    pulse: bool = False

    @model_validator(mode="after")
    def _check_hysteresis(self) -> ExtensionThresholds:
        if not self.t_engage > self.t_release:
            raise ValueError(
                f"t_engage ({self.t_engage}) must be strictly greater than "
                f"t_release ({self.t_release}) for extension gestures; the Schmitt band "
                "must have a gap to prevent chatter (hard invariant #1)."
            )
        return self


class GestureDetectorSettings(BaseModel):
    """Config-driven gesture detector entry.

    The virtual dual-thumbstick rewrite treats every discrete gesture as detector-backed.
    ``pinch``, ``curl_only``, and ``curl_combo`` use lower-is-engaged Schmitt semantics:
    ``t_release > t_engage``. ``extension_combo`` is the opposite: every listed
    ``extension_finger`` must extend above ``t_engage`` and release below ``t_release``.
    For ``pinch`` the signal is normalized thumb-to-fingertip distance; for ``curl_only`` it
    is the curled finger's extension ratio, gated by the listed ``open_fingers`` staying open;
    for ``curl_combo`` it is the highest extension ratio among all required curled fingers, so
    every listed finger must be down.
    """

    model_config = {"extra": "forbid"}

    detector: GestureDetectorName
    finger: DetectorFingerName
    t_engage: float = Field(gt=0.0)
    t_release: float = Field(gt=0.0)
    mode: GestureMode = "hold"
    open_fingers: tuple[DetectorFingerName, ...] = ()
    open_threshold: float = Field(default=1.1, gt=0.0)
    curl_fingers: tuple[CurlFingerName, ...] = ()
    extension_fingers: tuple[DetectorFingerName, ...] = ()
    conflict_group: str | None = None
    suppresses: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_detector(self) -> GestureDetectorSettings:
        if self.detector == "extension_combo":
            if not self.t_engage > self.t_release:
                raise ValueError(
                    f"t_engage ({self.t_engage}) must be strictly greater than "
                    f"t_release ({self.t_release}) for extension_combo gestures."
                )
        elif not self.t_release > self.t_engage:
            raise ValueError(
                f"t_release ({self.t_release}) must be strictly greater than "
                f"t_engage ({self.t_engage}) for {self.detector!r} gestures."
            )
        if self.detector == "pinch" and self.finger == "thumb":
            raise ValueError("pinch gestures target index/middle/ring/pinky, not thumb")
        if self.detector == "curl_only" and self.finger in self.open_fingers:
            raise ValueError("curl_only target finger cannot also be required open")
        if self.detector == "curl_combo":
            required = self.curl_fingers or (self.finger,)
            if self.finger == "thumb" and not self.curl_fingers:
                raise ValueError("curl_combo gestures require non-thumb curl_fingers")
            overlap = set(required).intersection(self.open_fingers)
            if overlap:
                raise ValueError(
                    f"curl_combo fingers cannot also be required open: {sorted(overlap)}"
                )
        if self.detector == "extension_combo":
            required = self.extension_fingers or (self.finger,)
            overlap = set(required).intersection(self.curl_fingers)
            if overlap:
                raise ValueError(
                    f"extension_combo fingers cannot also be required curled: {sorted(overlap)}"
                )
        return self


def _default_left_gestures() -> dict[str, ExtensionThresholds]:
    return {
        "jump": ExtensionThresholds(type="thumb_out", t_engage=1.2, t_release=0.9),
        "sneak": ExtensionThresholds(type="pinky_only", t_engage=1.15, t_release=1.05),
        "inventory": ExtensionThresholds(
            type="index_middle", t_engage=1.15, t_release=1.05, pulse=True
        ),
        "throw_item": ExtensionThresholds(
            type="ring_only", t_engage=1.15, t_release=1.05, pulse=True
        ),
    }


def _default_right_gestures() -> dict[str, GestureThresholds]:
    return {
        "attack": GestureThresholds(finger="index", t_engage=0.30, t_release=0.45),
        "use": GestureThresholds(finger="middle", t_engage=0.30, t_release=0.45),
        "hotbar_next": GestureThresholds(finger="ring", t_engage=0.30, t_release=0.45),
        "hotbar_prev": GestureThresholds(finger="pinky", t_engage=0.30, t_release=0.45),
    }


def _default_left_detector_gestures() -> dict[str, GestureDetectorSettings]:
    # Left hand is pure movement: thumb-to-fingertip pinches drive WASD. No conflict
    # groups, so two simultaneous pinches produce a diagonal (e.g. middle+index -> W+D).
    return {
        "move_right": GestureDetectorSettings(
            detector="pinch", finger="index", t_engage=0.30, t_release=0.45
        ),
        "move_forward": GestureDetectorSettings(
            detector="pinch", finger="middle", t_engage=0.30, t_release=0.45
        ),
        "move_left": GestureDetectorSettings(
            detector="pinch", finger="ring", t_engage=0.30, t_release=0.45
        ),
        "move_back": GestureDetectorSettings(
            detector="pinch", finger="pinky", t_engage=0.30, t_release=0.45
        ),
    }

def _default_right_detector_gestures() -> dict[str, GestureDetectorSettings]:
    return {
        "attack": GestureDetectorSettings(
            detector="pinch",
            finger="index",
            t_engage=0.30,
            t_release=0.45,
            conflict_group="primary_click",
        ),
        "use": GestureDetectorSettings(
            detector="pinch",
            finger="middle",
            t_engage=0.30,
            t_release=0.45,
            conflict_group="primary_click",
        ),
        "jump": GestureDetectorSettings(
            detector="pinch",
            finger="ring",
            t_engage=0.30,
            t_release=0.45,
            conflict_group="jump_sneak",
        ),
        "sneak": GestureDetectorSettings(
            detector="pinch",
            finger="pinky",
            t_engage=0.30,
            t_release=0.45,
            conflict_group="jump_sneak",
        ),
        "recenter": GestureDetectorSettings(
            detector="extension_combo",
            finger="index",
            t_engage=1.15,
            t_release=1.05,
            mode="hold",
            extension_fingers=("index", "middle"),
            curl_fingers=("ring", "pinky"),
            suppresses=("attack", "use", "jump", "sneak"),
        ),
    }


class FaceGestureDetectorSettings(BaseModel):
    """Face gesture detector settings."""

    model_config = {"extra": "forbid"}

    blendshape: str
    t_engage: float = Field(default=0.5, gt=0.0, le=1.0)
    t_release: float = Field(default=0.3, gt=0.0, le=1.0)
    engage_frames: int = Field(default=3, ge=1)
    release_frames: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _check_hysteresis(self) -> FaceGestureDetectorSettings:
        if not self.t_engage > self.t_release:
            raise ValueError(
                f"t_engage ({self.t_engage}) must be strictly greater than "
                f"t_release ({self.t_release}) for face gestures."
            )
        return self


def _default_face_gestures() -> dict[str, FaceGestureDetectorSettings]:
    return {
        "inventory": FaceGestureDetectorSettings(
            blendshape="browInnerUp",
            t_engage=0.5,
            t_release=0.3,
            engage_frames=3,
        ),
        "throw_item": FaceGestureDetectorSettings(
            blendshape="jawOpen",  # jawOpen is the actual mediapipe blendshape name for mouth open
            t_engage=0.6,
            t_release=0.35,
            engage_frames=2,
        ),
        "swap_offhand": FaceGestureDetectorSettings(
            blendshape="eyeBlinkLeft",  # blink left eye -> F (swap offhand)
            t_engage=0.5,
            t_release=0.3,
            engage_frames=2,
        ),
    }


class HeadRollGestureSettings(BaseModel):
    """Head-roll (ear-to-shoulder tilt) gesture thresholds, in degrees.

    The signal is the roll angle of the eye-corner line in the image plane (0 = upright;
    positive = head rolled toward the user's left shoulder). Two sign-gated Schmitt states
    fire ``left_gesture`` / ``right_gesture``, which are bound to scroll up / down. Hysteresis
    requires ``engage_deg > release_deg`` so a head held just past the threshold doesn't chatter.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = True
    left_gesture: str = "hotbar_next"  # roll toward left shoulder  -> scroll up
    right_gesture: str = "hotbar_prev"  # roll toward right shoulder -> scroll down
    engage_deg: float = Field(default=12.0, gt=0.0)
    release_deg: float = Field(default=7.0, gt=0.0)
    engage_frames: int = Field(default=2, ge=1)
    release_frames: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _check_hysteresis(self) -> HeadRollGestureSettings:
        if not self.engage_deg > self.release_deg:
            raise ValueError(
                f"engage_deg ({self.engage_deg}) must be strictly greater than "
                f"release_deg ({self.release_deg}) for head-roll gestures."
            )
        return self


class GestureSettings(BaseModel):
    """Per-hand/face maps of gesture-name -> detector-backed hold gesture config."""

    model_config = {"extra": "forbid"}

    left_hand: dict[str, GestureDetectorSettings] = Field(
        default_factory=_default_left_detector_gestures
    )
    right_hand: dict[str, GestureDetectorSettings] = Field(
        default_factory=_default_right_detector_gestures
    )
    face: dict[str, FaceGestureDetectorSettings] = Field(
        default_factory=_default_face_gestures
    )
    head_tilt: HeadRollGestureSettings = Field(default_factory=HeadRollGestureSettings)


class JoystickSettings(BaseModel):
    """Absolute screen-space joystick parameters (zero calibration)."""

    model_config = {"extra": "forbid"}

    deadzone: float = Field(default=0.04, ge=0.0)
    left_sensitivity: float = Field(default=5.0, gt=0.0)
    right_sensitivity: float = Field(default=40.0, gt=0.0)
    look_accel_exponent: float = Field(default=1.25, gt=0.0)
    """Exponential ease-in exponent applied to the right-hand mouse-look output."""
    smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    """EMA smoothing factor on the tracked position (0 = none, ->1 = heavy)."""
    right_smoothing: float | None = Field(default=0.55, ge=0.0, lt=1.0)
    """Optional right-hand-only EMA smoothing. Keeps mouse look steady without slowing WASD."""
    fixed_left_neutral: tuple[float, float] | None = Field(default=(0.25, 0.5))
    """Optional fixed screen-space anchor (x, y) for the left joystick (WASD)."""
    fixed_right_neutral: tuple[float, float] | None = Field(default=(0.75, 0.5))
    """Optional fixed screen-space anchor (x, y) for the right joystick (mouse look)."""

    # --- Mouse-look smoothing ----------------------------------------------------------
    look_filter: Literal["ema", "one_euro"] = "one_euro"
    one_euro_min_cutoff: float = Field(default=0.65, gt=0.0)
    one_euro_beta: float = Field(default=0.035, ge=0.0)
    one_euro_d_cutoff: float = Field(default=1.0, gt=0.0)


class InputSettings(BaseModel):
    """OS-input emission parameters. ``enabled`` is False by default (NullEmitter)."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    mouse_delta_scale: float = Field(default=58.0, gt=0.0)
    """Multiplier from normalized thumb movement to relative mouse pixels."""
    scroll_repeat_rate_hz: float = Field(default=8.0, gt=0.0)
    key_repeat_guard_ms: float = Field(default=50.0, ge=0.0)


class DebugSettings(BaseModel):
    """Debug overlay + logging. Never INFO-per-frame; rate-limited counters only."""

    model_config = {"extra": "forbid"}

    overlay: bool = False
    log_level: str = "WARNING"
    overlay_every: int = Field(default=1, ge=1)
    """Render the debug overlay on every Nth processed frame (Task 4 frame-dropping). ``1``
    draws every frame; ``2`` draws every other frame, etc. Tracking/gesture/input still run
    every frame — only the (non-free) HighGUI draw + ``imshow`` is decimated to protect the
    real-time loop. Ignored when ``overlay`` is False."""


def _default_bindings() -> dict[str, str]:
    return {
        # Left-hand pinch-WASD movement.
        "move_forward": "w",
        "move_back": "s",
        "move_left": "a",
        "move_right": "d",
        # Right-hand pinch gestures.
        "attack": "mouse_left",
        "use": "mouse_right",
        "jump": "space",
        "sneak": "shift",
        # Face gestures.
        "throw_item": "q",
        "inventory": "e",
        "swap_offhand": "f",
        # Head-roll -> hotbar scroll.
        "hotbar_next": "scroll_up",
        "hotbar_prev": "scroll_down",
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Settings(BaseSettings):
    """Root configuration model.

    Construct via :meth:`load` to read ``config.yaml`` (with optional CLI overrides).
    Environment variables prefixed ``MCV_`` (nested delimiter ``__``) override fields,
    e.g. ``MCV_INPUT__ENABLED=true``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MCV_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    camera: CameraSettings = Field(default_factory=CameraSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    face_tracking: FaceTrackerSettings = Field(default_factory=FaceTrackerSettings)
    gestures: GestureSettings = Field(default_factory=GestureSettings)
    joystick: JoystickSettings = Field(default_factory=JoystickSettings)
    input: InputSettings = Field(default_factory=InputSettings)
    bindings: dict[str, str] = Field(default_factory=_default_bindings)
    debug: DebugSettings = Field(default_factory=DebugSettings)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Settings:
        """Load settings from a YAML file, then apply CLI/programmatic overrides.

        Args:
            path: Path to a ``config.yaml``-style file. If None, only defaults +
                environment variables are used.
            overrides: A nested dict deep-merged over the file data (e.g. CLI flags like
                ``{"input": {"enabled": True}}``). Takes precedence over the file.

        Returns:
            A validated ``Settings`` instance.

        Raises:
            FileNotFoundError: If ``path`` is given but does not exist.
            pydantic.ValidationError: If any value (e.g. a bad threshold) is invalid.
        """
        data: dict[str, Any] = {}
        if path is not None:
            p = Path(path)
            if not p.is_file():
                raise FileNotFoundError(f"Config file not found: {p}")
            data = yaml.safe_load(p.read_text()) or {}
        if overrides:
            data = _deep_merge(data, overrides)
        return cls(**data)
