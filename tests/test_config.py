"""Tests for the pydantic Settings model and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from minecraft_cv.config import ExtensionThresholds, GestureThresholds, Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config.yaml"


def test_defaults_construct_without_yaml() -> None:
    s = Settings()
    assert s.camera.index == 0
    assert s.input.enabled is False  # NullEmitter by default (invariant #2)
    assert s.joystick.anchor == "wrist"
    assert s.joystick.cardinal_half_width == 35.0
    assert s.camera.mirror is True  # mirror view by default (also fixes handedness)
    assert s.tracking.swap_handedness is True
    assert set(s.gestures.left_hand) == {"jump", "sneak", "sprint", "inventory", "throw_item", "switch_offhand"}
    assert set(s.gestures.right_hand) == {"attack", "use", "hotbar_next", "hotbar_prev"}
    
    # Check new bindings exist
    assert s.bindings["sprint"] == "ctrl"
    assert s.bindings["throw_item"] == "q"
    assert s.bindings["switch_offhand"] == "f"


def test_load_project_config_yaml() -> None:
    s = Settings.load(CONFIG_YAML)
    assert s.camera.fps == 30
    assert s.tracking.backend == "mediapipe"
    assert s.gestures.left_hand["jump"].type == "thumb_out"
    assert s.gestures.right_hand["attack"].finger == "index"
    # input_resolution list in YAML is coerced to a tuple.
    assert s.tracking.input_resolution == (256, 256)


def test_every_gesture_satisfies_hysteresis_invariant() -> None:
    s = Settings.load(CONFIG_YAML)
    
    # Right hand (pinch): lower is engaged -> t_release > t_engage
    for name, g in s.gestures.right_hand.items():
        assert g.t_release > g.t_engage, f"{name} (pinch) violates t_release > t_engage"
        
    # Left hand (extension): higher is engaged -> t_engage > t_release
    for name, g in s.gestures.left_hand.items():
        assert g.t_engage > g.t_release, f"{name} (extension) violates t_engage > t_release"


def test_pinch_inverted_thresholds_raise() -> None:
    # Pinch requires t_release > t_engage
    with pytest.raises(ValidationError):
        GestureThresholds(finger="index", t_engage=0.45, t_release=0.30)


def test_extension_inverted_thresholds_raise() -> None:
    # Extension requires t_engage > t_release
    with pytest.raises(ValidationError):
        ExtensionThresholds(type="index_only", t_engage=1.05, t_release=1.15)


def test_equal_thresholds_raise() -> None:
    with pytest.raises(ValidationError):
        GestureThresholds(finger="index", t_engage=0.30, t_release=0.30)
    with pytest.raises(ValidationError):
        ExtensionThresholds(type="index_only", t_engage=1.15, t_release=1.15)


def test_bad_threshold_in_yaml_raises(tmp_path: Path) -> None:
    bad = {
        "gestures": {
            "left_hand": {
                # Extension should be engage > release, this is backward
                "jump": {"type": "thumb_out", "t_engage": 0.4, "t_release": 0.5},
            },
            "right_hand": {
                # Pinch should be release > engage, this is backward
                "attack": {"finger": "index", "t_engage": 0.45, "t_release": 0.3},
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
