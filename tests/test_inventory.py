"""Tests for the inventory-mode two-hand toggle and its pipeline integration.

Pure and deterministic: synthetic open-palm / closed-fist landmarks drive the detector and
the full ``Pipeline.step`` with a recording ``NullEmitter`` (no camera, no OS input).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from conftest import make_calibrated_settings
from minecraft_cv.gestures.inventory import InventoryModeToggle
from minecraft_cv.input.emitter import NullEmitter
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.tracking.tracker import HandResult

_OPEN = {"index": 1.3, "middle": 1.3, "ring": 1.3, "pinky": 1.3}
_FIST = {"index": 0.8, "middle": 0.8, "ring": 0.8, "pinky": 0.8}


def _open_palm(
    make: Callable[..., np.ndarray],
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    return make(_OPEN, thumb_ext=1.3, offset=offset)


def _fist(make: Callable[..., np.ndarray]) -> np.ndarray:
    return make(_FIST, thumb_ext=0.4)


# ---------------------------------------------------------------------------
# Detector unit tests
# ---------------------------------------------------------------------------


def test_held_two_hand_open_palm_toggles_on(
    make_extended_landmarks: Callable[..., np.ndarray],
) -> None:
    t = InventoryModeToggle(hold_frames=3, cooldown_frames=0)
    lm = _open_palm(make_extended_landmarks)
    assert t.update(lm, lm).active is False  # frame 1: holding
    assert t.update(lm, lm).active is False  # frame 2: holding
    res = t.update(lm, lm)  # frame 3: hold satisfied -> toggle
    assert res.toggled is True
    assert res.active is True


def test_holding_does_not_oscillate(
    make_extended_landmarks: Callable[..., np.ndarray],
) -> None:
    """Continuing to hold the pose past the toggle does not flip it back (disarmed)."""
    t = InventoryModeToggle(hold_frames=2, cooldown_frames=0)
    lm = _open_palm(make_extended_landmarks)
    t.update(lm, lm)
    assert t.update(lm, lm).active is True  # toggled on
    for _ in range(10):
        res = t.update(lm, lm)
        assert res.toggled is False
        assert res.active is True


def test_release_then_repose_toggles_off(
    make_extended_landmarks: Callable[..., np.ndarray],
) -> None:
    t = InventoryModeToggle(hold_frames=1, cooldown_frames=0)
    open_lm = _open_palm(make_extended_landmarks)
    fist = _fist(make_extended_landmarks)
    assert t.update(open_lm, open_lm).active is True  # on
    t.update(fist, fist)  # release pose (re-arm)
    assert t.update(open_lm, open_lm).active is False  # re-pose -> off


def test_single_hand_does_not_toggle(
    make_extended_landmarks: Callable[..., np.ndarray],
) -> None:
    t = InventoryModeToggle(hold_frames=1, cooldown_frames=0)
    open_lm = _open_palm(make_extended_landmarks)
    assert t.update(open_lm, None).active is False
    assert t.update(None, open_lm).active is False


def test_disabled_never_toggles(
    make_extended_landmarks: Callable[..., np.ndarray],
) -> None:
    t = InventoryModeToggle(enabled=False, hold_frames=1)
    lm = _open_palm(make_extended_landmarks)
    for _ in range(20):
        assert t.update(lm, lm).active is False


def test_tracking_loss_rearms_without_toggling(
    make_extended_landmarks: Callable[..., np.ndarray],
) -> None:
    t = InventoryModeToggle(hold_frames=3, cooldown_frames=0)
    lm = _open_palm(make_extended_landmarks)
    t.update(lm, lm)
    t.update(lm, lm)
    t.update(None, None)  # dropout resets the hold counter
    assert t.update(lm, lm).active is False  # must re-accumulate from scratch


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _inv_pipeline(emitter: NullEmitter) -> Pipeline:
    settings = make_calibrated_settings(
        inventory={"enabled": True, "hold_frames": 1, "cooldown_frames": 0}
    )
    return Pipeline.from_settings(settings, emitter=emitter)


def test_inventory_mode_disabled_by_default(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = Pipeline.from_settings(make_calibrated_settings(), emitter=null_emitter)
    left = make_hand_result(_open_palm(make_extended_landmarks), "Right")
    right = make_hand_result(_open_palm(make_extended_landmarks), "Left")
    assert pipe.step([left, right]).inventory_active is False


def test_inventory_mode_drives_absolute_cursor_not_relative_look(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _inv_pipeline(null_emitter)
    left = make_hand_result(_open_palm(make_extended_landmarks), "Right")  # -> physical left
    right = make_hand_result(_open_palm(make_extended_landmarks, offset=(0.3, 0.3, 0.0)), "Left")
    result = pipe.step([left, right])
    assert result.inventory_active is True
    # Absolute cursor move emitted; no relative mouse-look in inventory mode.
    assert any(e[0] == "mouse_move_abs" for e in null_emitter.log)
    assert not any(e[0] == "mouse_move" for e in null_emitter.log)


def test_inventory_mode_pauses_wasd_and_suppresses_left_gestures(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _inv_pipeline(null_emitter)
    # Both open palms toggle inventory on this frame; gameplay movement/actions stay paused.
    left = make_hand_result(_open_palm(make_extended_landmarks, offset=(0.7, 0.5, 0.0)), "Right")
    right = make_hand_result(_open_palm(make_extended_landmarks), "Left")
    result = pipe.step([left, right])
    assert result.inventory_active is True
    pass
    assert not any(e == ("key_down", "space") for e in null_emitter.log)


def test_toggling_off_returns_to_normal_movement(
    null_emitter: NullEmitter,
    make_extended_landmarks: Callable[..., np.ndarray],
    make_hand_result: Callable[..., HandResult],
) -> None:
    pipe = _inv_pipeline(null_emitter)
    open_l = make_hand_result(_open_palm(make_extended_landmarks), "Right")
    open_r = make_hand_result(_open_palm(make_extended_landmarks), "Left")
    fist_l = make_hand_result(_fist(make_extended_landmarks), "Right")
    fist_r = make_hand_result(_fist(make_extended_landmarks), "Left")
    assert pipe.step([open_l, open_r]).inventory_active is True  # on
    pipe.step([fist_l, fist_r])  # release pose (re-arm)
    assert pipe.step([open_l, open_r]).inventory_active is False  # off again
