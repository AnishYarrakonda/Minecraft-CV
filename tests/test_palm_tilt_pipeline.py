"""Pipeline wiring for the default ``palm_tilt`` mode and its tilt-to-pointer cursor."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from conftest import make_tilt_calibrated_settings
from minecraft_cv.config import Settings
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.joystick.palm_tilt import palm_tilt_xy
from minecraft_cv.pipeline import Pipeline


def test_default_mode_is_palm_tilt() -> None:
    assert Settings().joystick.mode == "palm_tilt"


def test_uncalibrated_palm_tilt_requires_calibration() -> None:
    """Live build must fail loudly when tilt neutrals are missing (safety)."""
    with pytest.raises(ValueError, match="[Pp]alm-tilt"):
        Pipeline.from_settings(Settings())


def test_uncalibrated_palm_tilt_preview_builds() -> None:
    pipe = Pipeline.from_settings(Settings(), allow_uncalibrated_palm_normal=True)
    assert pipe.joystick_mode == "palm_tilt"
    assert pipe.joystick_signal is palm_tilt_xy


def test_from_settings_wires_tilt_signal_and_mode() -> None:
    pipe = Pipeline.from_settings(make_tilt_calibrated_settings())
    assert pipe.joystick_mode == "palm_tilt"
    assert pipe.joystick_signal is palm_tilt_xy


def test_inventory_cursor_tracks_tilt_from_center(
    null_emitter: NullEmitter,
    make_tilt_landmarks: Callable[..., np.ndarray],
) -> None:
    """In palm_tilt mode the inventory cursor is a calibrated tilt-to-absolute pointer.

    Neutral (resting) tilt -> screen center; tilt right -> cursor right of center; tilt up ->
    cursor above center (smaller normalized y).
    """
    settings = make_tilt_calibrated_settings(
        inventory={"enabled": True, "hold_frames": 1, "cooldown_frames": 0, "cursor_gain": 1.0},
    )
    pipe = Pipeline.from_settings(settings, emitter=null_emitter)

    # Resting hand (tilt at the calibrated neutral of 0,0) sits at screen center.
    pipe._update_cursor(make_tilt_landmarks(tilt=(0.0, 0.0)))
    center = null_emitter.log[-1]
    assert center[0] == "mouse_move_abs"
    assert float(center[1]) == pytest.approx(0.5, abs=1e-6)
    assert float(center[2]) == pytest.approx(0.5, abs=1e-6)

    # Tilt right -> x > 0.5; tilt up (knuckles higher in frame -> negative image y) -> y < 0.5.
    pipe._update_cursor(make_tilt_landmarks(tilt=(0.3, -0.3)))
    moved = null_emitter.log[-1]
    sig = palm_tilt_xy(make_tilt_landmarks(tilt=(0.3, -0.3)))
    expected = np.clip(0.5 + np.clip(sig * np.array([2.0, 2.0]), -1.0, 1.0) * 0.5, 0.0, 1.0)
    assert float(moved[1]) == pytest.approx(expected[0], abs=1e-6)
    assert float(moved[2]) == pytest.approx(expected[1], abs=1e-6)
    assert float(moved[1]) > 0.5
    assert float(moved[2]) < 0.5
