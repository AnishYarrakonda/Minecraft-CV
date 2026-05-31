"""Deterministic tests for the auto-calibration math + safe config persistence (Task 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minecraft_cv.calibration import (
    REACH_POSES,
    compute_calibration,
    compute_palm_normal_calibration,
    compute_palm_normal_hand_calibration,
    load_config_data,
    merge_calibration,
    merge_palm_normal_calibration,
    merge_tilt_calibration,
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


def _normal_samples() -> dict[str, dict[str, np.ndarray]]:
    neutral = np.zeros((40, 2))
    return {
        hand: {
            "neutral": neutral,
            "up": np.array([[0.0, -0.3]]),
            "down": np.array([[0.0, 0.3]]),
            "left": np.array([[-0.2, 0.0]]),
            "right": np.array([[0.2, 0.0]]),
        }
        for hand in ("left", "right")
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


def test_palm_normal_calibration_writes_neutrals_and_axis_gains() -> None:
    result = compute_palm_normal_calibration(_normal_samples(), deadzone_floor=0.01)
    assert result.left.neutral == pytest.approx((0.0, 0.0))
    assert result.right.neutral == pytest.approx((0.0, 0.0))
    assert result.left.sensitivity[0] == pytest.approx(1.0 / (0.2 - result.left.deadzone))
    assert result.left.sensitivity[1] == pytest.approx(1.0 / (0.3 - result.left.deadzone))


def test_palm_normal_merge_preserves_other_settings() -> None:
    existing = {"camera": {"index": 2}, "joystick": {"smoothing": 0.6}}
    result = compute_palm_normal_calibration(_normal_samples())
    merged = merge_palm_normal_calibration(existing, result)
    assert merged["joystick"]["mode"] == "palm_normal"
    assert merged["joystick"]["palm_normal"]["left_neutral"] == [0.0, 0.0]
    assert merged["joystick"]["smoothing"] == 0.6
    assert merged["camera"] == {"index": 2}


def test_tilt_merge_writes_tilt_block_and_mode() -> None:
    existing = {
        "camera": {"index": 2},
        "joystick": {"smoothing": 0.6, "palm_normal": {"left_neutral": [9.0, 9.0]}},
    }
    result = compute_palm_normal_calibration(_normal_samples())
    merged = merge_tilt_calibration(existing, result)
    assert merged["joystick"]["mode"] == "palm_tilt"
    assert merged["joystick"]["tilt"]["left_neutral"] == [0.0, 0.0]
    assert merged["joystick"]["tilt"]["right_sensitivity"][1] == pytest.approx(
        1.0 / (0.3 - result.right.deadzone), rel=1e-4
    )
    # Other settings and the legacy palm_normal block are left untouched.
    assert merged["joystick"]["smoothing"] == 0.6
    assert merged["joystick"]["palm_normal"]["left_neutral"] == [9.0, 9.0]
    assert merged["camera"] == {"index": 2}
    # The merged config must validate (tilt is a real Settings field).
    Settings(**merged)


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


# ---------------------------------------------------------------------------
# Asymmetric (signed) gain tests
# ---------------------------------------------------------------------------


def _asymmetric_samples() -> dict[str, dict[str, np.ndarray]]:
    """Samples where up-reach (back) is much smaller than down-reach (forward)."""
    neutral = np.zeros((40, 2))
    return {
        hand: {
            "neutral": neutral,
            "up": np.array([[0.0, -0.08]]),    # small back reach
            "down": np.array([[0.0, 0.30]]),   # large forward reach
            "left": np.array([[-0.20, 0.0]]),
            "right": np.array([[0.20, 0.0]]),
        }
        for hand in ("left", "right")
    }


def test_four_signed_gains_computed() -> None:
    """compute_palm_normal_hand_calibration emits both sensitivity and sensitivity_neg."""
    samples = _asymmetric_samples()["left"]
    result = compute_palm_normal_hand_calibration(
        samples["neutral"],
        {p: samples.get(p, []) for p in ("up", "down", "left", "right")},
        deadzone_floor=0.01,
    )
    assert len(result.sensitivity) == 2
    assert len(result.sensitivity_neg) == 2


def test_back_gain_exceeds_forward_gain_when_up_reach_smaller() -> None:
    """When up-reach (back) is smaller than down-reach (forward), back_gain > forward_gain."""
    samples = _asymmetric_samples()["left"]
    result = compute_palm_normal_hand_calibration(
        samples["neutral"],
        {p: samples.get(p, []) for p in ("up", "down", "left", "right")},
        deadzone_floor=0.01,
    )
    forward_gain = result.sensitivity[1]     # positive-y / forward
    back_gain = result.sensitivity_neg[1]    # negative-y / back
    assert back_gain > forward_gain, (
        f"back_gain ({back_gain:.3f}) should exceed forward_gain ({forward_gain:.3f}) "
        "when the back reach is smaller"
    )


def test_symmetric_reaches_produce_equal_gains() -> None:
    """Equal up and down reach -> forward_gain ≈ back_gain."""
    samples = {
        "neutral": np.zeros((40, 2)),
        "up": np.array([[0.0, -0.25]]),
        "down": np.array([[0.0, 0.25]]),
        "left": np.array([[-0.2, 0.0]]),
        "right": np.array([[0.2, 0.0]]),
    }
    result = compute_palm_normal_hand_calibration(
        samples["neutral"],
        {p: samples[p] for p in ("up", "down", "left", "right")},
        deadzone_floor=0.01,
    )
    assert result.sensitivity[1] == pytest.approx(result.sensitivity_neg[1], rel=1e-4)


def test_joystick_overrides_serializes_sensitivity_neg() -> None:
    """joystick_overrides() must include left_sensitivity_neg and right_sensitivity_neg."""
    result = compute_palm_normal_calibration(_asymmetric_samples(), deadzone_floor=0.01)
    overrides = result.joystick_overrides(mode="palm_tilt", block="tilt")
    block = overrides["tilt"]
    assert "left_sensitivity_neg" in block
    assert "right_sensitivity_neg" in block
    assert len(block["left_sensitivity_neg"]) == 2
    assert len(block["right_sensitivity_neg"]) == 2


def test_tilt_merge_with_sensitivity_neg_validates() -> None:
    """Merged config including sensitivity_neg must validate through Settings."""
    result = compute_palm_normal_calibration(_asymmetric_samples(), deadzone_floor=0.01)
    merged = merge_tilt_calibration({}, result)
    # Must not raise ValidationError
    s = Settings(**merged)
    assert s.joystick.tilt.left_sensitivity_neg is not None
    assert s.joystick.tilt.right_sensitivity_neg is not None
