"""Deterministic tests for the auto-calibration math + safe config persistence (Task 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minecraft_cv.calibration import (
    REACH_POSES,
    compute_calibration,
    load_config_data,
    merge_calibration,
    save_config_data,
)
from minecraft_cv.config import Settings


def _neutral_cloud(center=(0.5, 0.5), jitter=0.005, n=120, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.normal(0.0, jitter, size=(n, 2))


def _reach(center=(0.5, 0.5), reach=0.3) -> dict[str, np.ndarray]:
    return {
        "forward": np.array([[center[0], center[1] - reach]]),
        "back": np.array([[center[0], center[1] + reach]]),
        "left": np.array([[center[0] - reach, center[1]]]),
        "right": np.array([[center[0] + reach, center[1]]]),
    }


def test_neutral_estimates_center() -> None:
    r = compute_calibration(_neutral_cloud(), _reach())
    assert r.neutral[0] == pytest.approx(0.5, abs=0.01)
    assert r.neutral[1] == pytest.approx(0.5, abs=0.01)


def test_deadzone_scales_with_jitter() -> None:
    steady = compute_calibration(_neutral_cloud(jitter=0.002), _reach())
    shaky = compute_calibration(_neutral_cloud(jitter=0.02), _reach())
    assert shaky.deadzone_radius > steady.deadzone_radius


def test_deadzone_respects_floor() -> None:
    """A perfectly still hand still gets at least the floor deadzone."""
    still = np.full((50, 2), 0.5)
    r = compute_calibration(still, _reach(), deadzone_floor=0.01)
    assert r.deadzone_radius == pytest.approx(0.01)


def test_sensitivity_saturates_at_reach() -> None:
    """Output should reach saturation right at the user's measured full reach."""
    r = compute_calibration(_neutral_cloud(jitter=0.002), _reach(reach=0.3))
    # travel = mean_reach - deadzone; sensitivity == 1 / travel.
    expected = 1.0 / (r.mean_reach - r.deadzone_radius)
    assert r.sensitivity == pytest.approx(expected, rel=1e-6)


def test_sensitivity_clamped_when_reach_tiny() -> None:
    r = compute_calibration(
        np.full((10, 2), 0.5), {"forward": np.array([[0.5, 0.5]])}, max_sensitivity=50.0
    )
    assert r.sensitivity == pytest.approx(50.0)


def test_empty_neutral_raises() -> None:
    with pytest.raises(ValueError):
        compute_calibration([], _reach())


def test_reach_poses_constant_matches_directions() -> None:
    assert set(REACH_POSES) == {"forward", "back", "left", "right"}


# --- persistence ------------------------------------------------------------


def test_merge_preserves_other_settings() -> None:
    existing = {
        "camera": {"index": 2, "fps": 60},
        "joystick": {"deadzone_radius": 0.05, "sensitivity": 2.0, "anchor": "wrist"},
    }
    r = compute_calibration(_neutral_cloud(jitter=0.002), _reach())
    merged = merge_calibration(existing, r)
    # joystick values updated...
    assert merged["joystick"]["deadzone_radius"] == r.joystick_overrides()["deadzone_radius"]
    assert merged["joystick"]["sensitivity"] == r.joystick_overrides()["sensitivity"]
    # ...other joystick keys and other sections untouched.
    assert merged["joystick"]["anchor"] == "wrist"
    assert merged["camera"] == {"index": 2, "fps": 60}
    # original not mutated
    assert existing["joystick"]["deadzone_radius"] == 0.05


def test_save_load_roundtrip_is_atomic_and_valid(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    original = {"camera": {"index": 1}, "joystick": {"deadzone_radius": 0.05, "sensitivity": 2.0}}
    save_config_data(cfg, original)
    r = compute_calibration(_neutral_cloud(jitter=0.002), _reach())
    merged = merge_calibration(load_config_data(cfg), r)
    save_config_data(cfg, merged)
    # No stray temp file left behind.
    assert not (tmp_path / "config.yaml.tmp").exists()
    # Reloads, preserves camera, and validates through the real Settings model.
    reloaded = load_config_data(cfg)
    assert reloaded["camera"] == {"index": 1}
    settings = Settings.load(cfg)
    assert settings.joystick.deadzone_radius == r.joystick_overrides()["deadzone_radius"]
    assert settings.camera.index == 1
