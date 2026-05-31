"""Tests for the pydantic Settings model and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from minecraft_cv.config import (
    ExtensionThresholds,
    GestureDetectorSettings,
    GestureThresholds,
    Settings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config.yaml"


def test_defaults_construct_without_yaml() -> None:
    s = Settings()
    assert s.camera.index == 0
    assert s.input.enabled is False  # NullEmitter by default (invariant #2)
    assert s.joystick.mode == "palm_tilt"
    assert s.joystick.tilt.left_neutral is None
    assert s.joystick.palm_normal.left_neutral is None
    assert s.joystick.anchor == "wrist"
    assert s.joystick.cardinal_half_width == 35.0
    assert s.camera.mirror is True  # mirror view by default (also fixes handedness)
    assert s.tracking.swap_handedness is True
    assert s.inventory.enabled is False
    assert set(s.gestures.left_hand) == {
        "jump",
        "sneak",
        "inventory",
        "throw_item",
        "switch_offhand",
    }
    assert set(s.gestures.right_hand) == {"attack", "use", "hotbar_next", "hotbar_prev", "sprint"}

    # Check new bindings exist
    assert s.bindings["sprint"] == "ctrl"
    assert s.bindings["throw_item"] == "q"
    assert s.bindings["switch_offhand"] == "f"


def test_load_project_config_yaml() -> None:
    s = Settings.load(CONFIG_YAML)
    assert s.camera.fps == 30
    assert s.tracking.backend == "mediapipe"
    assert s.joystick.mode == "palm_tilt"
    assert s.gestures.left_hand["jump"].detector == "pinch"
    assert s.gestures.left_hand["sneak"].detector == "curl_combo"
    assert s.gestures.left_hand["sneak"].mode == "hold"
    assert s.gestures.left_hand["sneak"].curl_fingers == ("ring", "pinky")
    assert s.gestures.right_hand["sprint"].curl_fingers == ("ring", "pinky")
    assert s.gestures.right_hand["sprint"].suppresses == ("hotbar_next", "hotbar_prev")
    assert s.gestures.right_hand["attack"].finger == "index"
    # input_resolution list in YAML is coerced to a tuple.
    assert s.tracking.input_resolution == (256, 256)


def test_every_gesture_satisfies_hysteresis_invariant() -> None:
    s = Settings.load(CONFIG_YAML)

    # Detector-backed gestures: lower is engaged -> t_release > t_engage
    for name, g in s.gestures.right_hand.items():
        assert g.t_release > g.t_engage, f"{name} violates t_release > t_engage"

    for name, g in s.gestures.left_hand.items():
        assert g.t_release > g.t_engage, f"{name} violates t_release > t_engage"


def test_pinch_inverted_thresholds_raise() -> None:
    # Pinch requires t_release > t_engage
    with pytest.raises(ValidationError):
        GestureThresholds(finger="index", t_engage=0.45, t_release=0.30)


def test_extension_inverted_thresholds_raise() -> None:
    # Extension requires t_engage > t_release
    with pytest.raises(ValidationError):
        ExtensionThresholds(type="index_only", t_engage=1.05, t_release=1.15)


def test_detector_inverted_thresholds_raise() -> None:
    with pytest.raises(ValidationError):
        GestureDetectorSettings(
            detector="pinch", finger="index", t_engage=0.45, t_release=0.30
        )


def test_equal_thresholds_raise() -> None:
    with pytest.raises(ValidationError):
        GestureThresholds(finger="index", t_engage=0.30, t_release=0.30)
    with pytest.raises(ValidationError):
        ExtensionThresholds(type="index_only", t_engage=1.15, t_release=1.15)


def test_bad_threshold_in_yaml_raises(tmp_path: Path) -> None:
    bad = {
        "gestures": {
            "left_hand": {
                # Detector gestures require release > engage; this is backward
                "jump": {
                    "detector": "pinch",
                    "finger": "index",
                    "t_engage": 0.45,
                    "t_release": 0.3,
                },
            },
            "right_hand": {
                # Same detector invariant on the right hand
                "attack": {
                    "detector": "pinch",
                    "finger": "index",
                    "t_engage": 0.45,
                    "t_release": 0.3,
                },
            },
        }
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        Settings.load(p)


def test_overrides_take_precedence() -> None:
    s = Settings.load(CONFIG_YAML, overrides={"input": {"enabled": True}})
    assert s.input.enabled is True
    # Unrelated fields are untouched by the override.
    assert s.camera.fps == 30


def test_overrides_deep_merge_preserves_siblings() -> None:
    s = Settings.load(
        CONFIG_YAML,
        overrides={"debug": {"overlay": True}},
    )
    assert s.debug.overlay is True
    assert s.debug.log_level == "WARNING"  # sibling preserved, not wiped


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Settings.load(tmp_path / "does_not_exist.yaml")


def test_extra_keys_forbidden(tmp_path: Path) -> None:
    p = tmp_path / "extra.yaml"
    p.write_text(yaml.safe_dump({"camera": {"index": 0, "bogus_field": 1}}))
    with pytest.raises(ValidationError):
        Settings.load(p)


# ---------------------------------------------------------------------------
# New field tests: look_accel_exponent + sensitivity_neg
# ---------------------------------------------------------------------------


def test_look_accel_exponent_default() -> None:
    """look_accel_exponent should default to 1.6."""
    s = Settings()
    assert s.joystick.look_accel_exponent == pytest.approx(1.6)


def test_look_accel_exponent_from_yaml(tmp_path: Path) -> None:
    """look_accel_exponent is read from YAML and validated."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"joystick": {"look_accel_exponent": 2.5}}))
    s = Settings.load(cfg)
    assert s.joystick.look_accel_exponent == pytest.approx(2.5)


def test_look_accel_exponent_must_be_positive(tmp_path: Path) -> None:
    """look_accel_exponent <= 0 must raise ValidationError."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"joystick": {"look_accel_exponent": 0.0}}))
    with pytest.raises(ValidationError):
        Settings.load(cfg)


def test_sensitivity_neg_defaults_none() -> None:
    """left_sensitivity_neg and right_sensitivity_neg should default to None."""
    s = Settings()
    assert s.joystick.tilt.left_sensitivity_neg is None
    assert s.joystick.tilt.right_sensitivity_neg is None
    assert s.joystick.palm_normal.left_sensitivity_neg is None
    assert s.joystick.palm_normal.right_sensitivity_neg is None


def test_sensitivity_neg_round_trips_via_yaml(tmp_path: Path) -> None:
    """sensitivity_neg values survive a YAML round-trip."""
    data = {
        "joystick": {
            "mode": "palm_tilt",
            "tilt": {
                "left_neutral": [0.0, 0.0],
                "right_neutral": [0.0, 0.0],
                "deadzone": 0.05,
                "left_sensitivity": [4.0, 3.0],
                "right_sensitivity": [4.0, 3.0],
                "left_sensitivity_neg": [3.5, 10.0],
                "right_sensitivity_neg": [3.5, 9.5],
            },
        }
    }
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(data))
    s = Settings.load(cfg)
    assert s.joystick.tilt.left_sensitivity_neg == pytest.approx((3.5, 10.0))
    assert s.joystick.tilt.right_sensitivity_neg == pytest.approx((3.5, 9.5))


def test_project_config_yaml_loads_new_fields() -> None:
    """The project config.yaml must load cleanly with the new fields."""
    s = Settings.load(CONFIG_YAML)
    assert s.joystick.smoothing == pytest.approx(0.4)
    assert s.joystick.cardinal_half_width == pytest.approx(40.0)
    assert s.joystick.look_accel_exponent == pytest.approx(1.6)
    assert s.input.mouse_delta_scale == pytest.approx(28.0)

