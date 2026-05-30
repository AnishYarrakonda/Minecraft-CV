"""Tests for the pydantic Settings model and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from minecraft_cv.config import GestureThresholds, Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config.yaml"


def test_defaults_construct_without_yaml() -> None:
    s = Settings()
    assert s.camera.index == 0
    assert s.input.enabled is False  # NullEmitter by default (invariant #2)
    assert s.joystick.anchor == "wrist"
    assert set(s.gestures.left_hand) == {"jump", "sneak"}
    assert set(s.gestures.right_hand) == {"attack", "use"}


def test_load_project_config_yaml() -> None:
    s = Settings.load(CONFIG_YAML)
    assert s.camera.fps == 30
    assert s.tracking.backend == "mediapipe"
    assert s.gestures.left_hand["jump"].finger == "index"
    assert s.gestures.right_hand["attack"].finger == "index"
    # input_resolution list in YAML is coerced to a tuple.
    assert s.tracking.input_resolution == (256, 256)


def test_every_gesture_satisfies_hysteresis_invariant() -> None:
    s = Settings.load(CONFIG_YAML)
    for hand in (s.gestures.left_hand, s.gestures.right_hand):
        for name, g in hand.items():
            assert g.t_release > g.t_engage, f"{name} violates t_release > t_engage"


def test_inverted_thresholds_raise() -> None:
    with pytest.raises(ValidationError):
        GestureThresholds(finger="index", t_engage=0.45, t_release=0.30)


def test_equal_thresholds_raise() -> None:
    with pytest.raises(ValidationError):
        GestureThresholds(finger="index", t_engage=0.30, t_release=0.30)


def test_bad_threshold_in_yaml_raises(tmp_path: Path) -> None:
    bad = {
        "gestures": {
            "left_hand": {
                "jump": {"finger": "index", "t_engage": 0.5, "t_release": 0.4},
            },
            "right_hand": {
                "attack": {"finger": "index", "t_engage": 0.3, "t_release": 0.45},
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
