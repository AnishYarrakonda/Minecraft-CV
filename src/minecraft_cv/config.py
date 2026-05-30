"""Typed configuration for minecraft_cv.

Loads ``config.yaml`` into a validated pydantic ``Settings`` model. Every tunable value
lives here so gesture/joystick/input code never holds magic numbers. The most important
guarantee enforced at construction time is hard-invariant #1: ``t_release > t_engage`` for
every configured pinch gesture (see ``.claude/rules/gestures.md``).

All distance thresholds are **unitless ratios** (thumb-to-fingertip distance normalized by
hand scale), so they are invariant to how far the hand is from the camera.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

FingerName = Literal["index", "middle", "ring", "pinky", "fist"]
Anchor = Literal["wrist", "middle_mcp"]
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
            a sustained hold. Used for toggle/one-shot actions (E, Q, F).
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


def _default_left_gestures() -> dict[str, ExtensionThresholds]:
    return {
        "jump": ExtensionThresholds(type="thumb_out", t_engage=1.2, t_release=0.9),
        "sneak": ExtensionThresholds(type="index_only", t_engage=1.15, t_release=1.05),
        "sprint": ExtensionThresholds(type="middle_only", t_engage=1.15, t_release=1.05),
        "inventory": ExtensionThresholds(
            type="index_middle", t_engage=1.15, t_release=1.05, pulse=True
        ),
        "throw_item": ExtensionThresholds(
            type="ring_only", t_engage=1.15, t_release=1.05, pulse=True
        ),
        "switch_offhand": ExtensionThresholds(
            type="pinky_only", t_engage=1.15, t_release=1.05, pulse=True
        ),
    }


def _default_right_gestures() -> dict[str, GestureThresholds]:
    return {
        "attack": GestureThresholds(finger="index", t_engage=0.30, t_release=0.45),
        "use": GestureThresholds(finger="middle", t_engage=0.30, t_release=0.45),
        "hotbar_next": GestureThresholds(finger="ring", t_engage=0.30, t_release=0.45),
        "hotbar_prev": GestureThresholds(finger="pinky", t_engage=0.30, t_release=0.45),
    }


class GestureSettings(BaseModel):
    """Per-hand maps of gesture-name -> Schmitt thresholds.

    Left hand uses extension-based gestures (finger combinations from closed fist).
    Right hand uses pinch-based gestures (thumb-to-fingertip distances).
    """

    model_config = {"extra": "forbid"}

    left_hand: dict[str, ExtensionThresholds] = Field(
        default_factory=_default_left_gestures
    )
    right_hand: dict[str, GestureThresholds] = Field(
        default_factory=_default_right_gestures
    )


class JoystickSettings(BaseModel):
    """Spatial-joystick deadzone, sensitivity, and acceleration parameters.

    Units are normalized landmark coordinates ([0, 1] in frame space). The deadzone is a
    sphere radius, not a box half-width, so diagonal directions are not biased.
    """

    model_config = {"extra": "forbid"}

    deadzone_radius: float = Field(default=0.05, ge=0.0)
    sensitivity: float = Field(default=2.0, gt=0.0)
    accel_exponent: float = Field(default=2.0, gt=0.0)
    anchor: Anchor = "wrist"
    max_output: float = Field(default=1.0, gt=0.0)
    smoothing: float = Field(default=0.6, ge=0.0, lt=1.0)
    """EMA smoothing factor on the anchor position (0 = none, ->1 = heavy).

    Cuts MediaPipe landmark jitter before it reaches the deadzone/accel curve so the
    joystick output flows instead of chattering. Higher = calmer but slightly more lag.
    """
    recenter_grace_frames: int = Field(default=3, ge=0)
    """Consecutive missing-hand frames tolerated before the neutral is recentered.

    A single dropped frame should not snap the neutral to a new position (a jarring jump on
    the next frame). The hand's keys are still released immediately on any miss; only the
    recenter macro waits for a sustained dropout.
    """
    cardinal_half_width: float = Field(default=35.0, ge=0.0, le=45.0)
    """Half-width (degrees) of each pure cardinal direction zone.

    With the default of 35°, each axis has a 70° zone (±35°) where only that axis's key is
    pressed (pure W, pure D, etc.). The remaining 20° between zones produces diagonal
    movement (W+D, etc.). Set to 45° for no diagonals; set to 0° for the old behavior
    where any non-zero component always fires.
    """


class InputSettings(BaseModel):
    """OS-input emission parameters. ``enabled`` is False by default (NullEmitter)."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    mouse_delta_scale: float = Field(default=15.0, gt=0.0)
    scroll_repeat_rate_hz: float = Field(default=8.0, gt=0.0)
    key_repeat_guard_ms: float = Field(default=50.0, ge=0.0)


class DebugSettings(BaseModel):
    """Debug overlay + logging. Never INFO-per-frame; rate-limited counters only."""

    model_config = {"extra": "forbid"}

    overlay: bool = False
    log_level: str = "WARNING"


def _default_bindings() -> dict[str, str]:
    return {
        # Left-hand discrete gestures.
        "jump": "space",
        "sneak": "shift",
        "sprint": "ctrl",
        "inventory": "e",
        "throw_item": "q",
        "switch_offhand": "f",
        # Right-hand discrete gestures.
        "attack": "mouse_left",
        "use": "mouse_right",
        "hotbar_next": "scroll_up",
        "hotbar_prev": "scroll_down",
        # Left-hand spatial-joystick translation -> WASD.
        "forward": "w",
        "back": "s",
        "left": "a",
        "right": "d",
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
