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


class CameraSettings(BaseModel):
    """Hardware capture parameters for ``cv2.VideoCapture``."""

    model_config = {"extra": "forbid"}

    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "avfoundation"


class TrackingSettings(BaseModel):
    """Hand-tracking backend selection and inference parameters."""

    model_config = {"extra": "forbid"}

    backend: str = "mediapipe"
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    input_resolution: tuple[int, int] = (256, 256)
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_num_hands: int = Field(default=2, ge=1, le=2)


class GestureThresholds(BaseModel):
    """Schmitt-trigger thresholds for one discrete pinch gesture.

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


def _default_left_gestures() -> dict[str, GestureThresholds]:
    return {
        "jump": GestureThresholds(finger="index", t_engage=0.30, t_release=0.45),
        "sneak": GestureThresholds(finger="middle", t_engage=0.30, t_release=0.45),
    }


def _default_right_gestures() -> dict[str, GestureThresholds]:
    return {
        "attack": GestureThresholds(finger="index", t_engage=0.30, t_release=0.45),
        "use": GestureThresholds(finger="middle", t_engage=0.30, t_release=0.45),
    }


class GestureSettings(BaseModel):
    """Per-hand maps of gesture-name -> Schmitt thresholds."""

    model_config = {"extra": "forbid"}

    left_hand: dict[str, GestureThresholds] = Field(default_factory=_default_left_gestures)
    right_hand: dict[str, GestureThresholds] = Field(default_factory=_default_right_gestures)


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


class InputSettings(BaseModel):
    """OS-input emission parameters. ``enabled`` is False by default (NullEmitter)."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    mouse_delta_scale: float = Field(default=5.0, gt=0.0)
    scroll_repeat_rate_hz: float = Field(default=8.0, gt=0.0)
    key_repeat_guard_ms: float = Field(default=50.0, ge=0.0)


class DebugSettings(BaseModel):
    """Debug overlay + logging. Never INFO-per-frame; rate-limited counters only."""

    model_config = {"extra": "forbid"}

    overlay: bool = False
    log_level: str = "WARNING"


def _default_bindings() -> dict[str, str]:
    return {
        # Discrete pinch gestures.
        "jump": "space",
        "sneak": "shift",
        "attack": "mouse_left",
        "use": "mouse_right",
        "inventory": "e",
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
