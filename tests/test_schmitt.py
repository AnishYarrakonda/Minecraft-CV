"""Unit tests for the Schmitt trigger — pure, deterministic, no camera."""

from __future__ import annotations

import pytest

from minecraft_cv.gestures.schmitt import (
    KEY_DOWN,
    KEY_UP,
    PinchState,
    SchmittTrigger,
)

# Canonical band used across cases.
ENGAGE = 0.30
RELEASE = 0.45


def _trigger() -> SchmittTrigger:
    return SchmittTrigger(t_engage=ENGAGE, t_release=RELEASE, engage_frames=2, release_frames=2)


def test_clean_engage_then_release() -> None:
    t = _trigger()
    assert t.update(0.20) is None
    assert t.update(0.20) == KEY_DOWN  # below engage for 2 frames -> HOLDING
    assert t.state is PinchState.HOLDING
    assert t.update(0.50) is None
    assert t.update(0.50) == KEY_UP  # above release for 2 frames -> RELEASED
    assert t.state is PinchState.RELEASED


def test_hover_just_above_engage_no_transition() -> None:
    t = _trigger()
    # Just above engage from RELEASED: nothing should fire.
    for _ in range(20):
        assert t.update(ENGAGE + 0.01) is None
    assert t.state is PinchState.RELEASED


def test_hover_inside_band_does_not_chatter() -> None:
    t = _trigger()
    assert t.update(0.10) is None
    assert t.update(0.10) == KEY_DOWN  # engage first
    # Now sit inside the hysteresis band (between engage and release): must hold steady.
    for d in [0.31, 0.40, 0.44, 0.35, 0.31, 0.44]:
        assert t.update(d) is None
    assert t.state is PinchState.HOLDING


def test_jittery_sequence_exact_transitions() -> None:
    t = _trigger()
    # distance, expected transition
    sequence = [
        (0.50, None),  # released, far
        (0.40, None),  # in band but still released -> stays released
        (0.29, None),  # dips below engage (1 frame)
        (0.31, None),  # back into band -> resets counter, stays released
        (0.29, None),  # dips below engage
        (0.29, KEY_DOWN),  # dips below engage (2 frames)
        (0.44, None),  # near release edge -> still holding
        (0.46, None),  # crosses release (1 frame)
        (0.44, None),  # back in band -> resets counter, stays holding
        (0.46, None),  # crosses release
        (0.46, KEY_UP),  # crosses release (2 frames)
        (0.10, None),  # re-engage
        (0.10, KEY_DOWN),  # re-engage
    ]
    for distance, expected in sequence:
        assert t.update(distance) == expected


def test_debounce_spike() -> None:
    t = _trigger()
    # 1 frame spike
    assert t.update(0.29) is None
    assert t.update(0.31) is None
    assert t.state is PinchState.RELEASED

    # 2 frame spike
    assert t.update(0.29) is None
    assert t.update(0.29) == KEY_DOWN
    assert t.state is PinchState.HOLDING


def test_inverted_thresholds_raise() -> None:
    with pytest.raises(ValueError):
        SchmittTrigger(t_engage=0.45, t_release=0.30)


def test_equal_thresholds_raise() -> None:
    with pytest.raises(ValueError):
        SchmittTrigger(t_engage=0.30, t_release=0.30)


def test_reset_releases_when_holding() -> None:
    t = _trigger()
    t.update(0.10)
    t.update(0.10)  # engage
    assert t.state is PinchState.HOLDING
    assert t.reset() == KEY_UP
    assert t.state is PinchState.RELEASED


def test_reset_is_noop_when_released() -> None:
    t = _trigger()
    assert t.reset() is None
    assert t.state is PinchState.RELEASED


def test_multiple_cycles() -> None:
    t = _trigger()
    for _ in range(5):
        assert t.update(0.10) is None
        assert t.update(0.10) == KEY_DOWN
        assert t.update(0.20) is None  # still holding
        assert t.update(0.60) is None
        assert t.update(0.60) == KEY_UP
        assert t.update(0.50) is None  # still released
