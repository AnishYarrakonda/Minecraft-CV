"""Deterministic tests for the depth-velocity Sprint trigger (Task 2).

The trigger is pure: it takes a stream of ``(z, timestamp)`` samples (the left-hand anchor's
relative-depth coordinate) and emits ``ENGAGE`` / ``RELEASE`` tokens. No camera, no landmarks,
no clock — timestamps are supplied explicitly so the velocity math is reproducible.

Frame of reference: MediaPipe ``z`` is depth relative to the wrist; **negative = closer to
the camera**. A fast forward push therefore makes ``z`` *decrease*; forward velocity is
``-dz/dt``.
"""

from __future__ import annotations

from minecraft_cv.joystick.sprint_velocity import ENGAGE, RELEASE, SprintVelocityTrigger


def _trigger(**kw: float | int | bool) -> SprintVelocityTrigger:
    params: dict[str, float | int | bool] = dict(
        v_sprint=1.0,  # normalized z units / second
        trigger_frames=3,
        release_margin=0.02,
        enabled=True,
    )
    params.update(kw)
    return SprintVelocityTrigger(**params)  # type: ignore[arg-type]


def test_first_sample_seeds_neutral_no_event() -> None:
    t = _trigger()
    assert t.update(0.0, 0.0) is None
    assert not t.active


def test_slow_forward_motion_does_not_engage() -> None:
    """Pushing forward gently (below v_sprint) never trips the sprint."""
    t = _trigger(v_sprint=1.0)
    t.update(0.0, 0.0)
    z = 0.0
    for i in range(1, 10):
        z -= 0.005  # 0.005 units per 0.1 s = 0.05 u/s « 1.0 threshold
        assert t.update(z, i * 0.1) is None
    assert not t.active


def test_fast_forward_push_engages_after_trigger_frames() -> None:
    """A fast forward push sustained for ``trigger_frames`` frames engages sprint."""
    t = _trigger(v_sprint=1.0, trigger_frames=3)
    t.update(0.0, 0.0)
    # dz = -0.2 over dt = 0.1 s -> forward velocity = 2.0 u/s > 1.0 threshold.
    z = 0.0
    events = []
    for i in range(1, 4):
        z -= 0.2
        events.append(t.update(z, i * 0.1))
    # First two frames count up; the third reaches trigger_frames and engages.
    assert events[0] is None
    assert events[1] is None
    assert events[2] == ENGAGE
    assert t.active


def test_engage_requires_consecutive_frames() -> None:
    """A single fast frame followed by a slow frame resets the counter (no engage)."""
    t = _trigger(v_sprint=1.0, trigger_frames=3)
    t.update(0.0, 0.0)
    assert t.update(-0.2, 0.1) is None  # fast (2.0 u/s) -> count 1
    assert t.update(-0.205, 0.2) is None  # slow (0.05 u/s) -> count reset to 0
    assert t.update(-0.405, 0.3) is None  # fast again -> count 1, not yet 3
    assert not t.active


def test_sprint_holds_while_forward_then_releases_on_retreat() -> None:
    t = _trigger(v_sprint=1.0, trigger_frames=3, release_margin=0.02)
    t.update(0.0, 0.0)  # neutral z = 0.0
    z = 0.0
    for i in range(1, 4):
        z -= 0.2
        t.update(z, i * 0.1)
    assert t.active
    # Hand stays pushed forward (z well below neutral) -> still sprinting, no event.
    assert t.update(-0.6, 0.4) is None
    assert t.active
    # Hand retreats back toward neutral (z rises above neutral - release_margin) -> release.
    assert t.update(0.0, 0.5) == RELEASE
    assert not t.active


def test_disabled_trigger_never_engages() -> None:
    t = _trigger(enabled=False)
    t.update(0.0, 0.0)
    z = 0.0
    for i in range(1, 6):
        z -= 0.3
        assert t.update(z, i * 0.1) is None
    assert not t.active


def test_reset_releases_when_active() -> None:
    """reset() is the fail-safe used on tracking loss; returns RELEASE if sprinting."""
    t = _trigger()
    t.update(0.0, 0.0)
    z = 0.0
    for i in range(1, 4):
        z -= 0.2
        t.update(z, i * 0.1)
    assert t.active
    assert t.reset() == RELEASE
    assert not t.active
    assert t.reset() is None  # idempotent


def test_nonadvancing_clock_is_ignored() -> None:
    """A duplicate timestamp must not divide by zero or spuriously engage."""
    t = _trigger()
    t.update(0.0, 0.0)
    assert t.update(-0.5, 0.0) is None  # dt == 0 -> ignored
    assert not t.active


def test_reset_neutral_reanchors_depth() -> None:
    """After reset_neutral the next sample reseeds the neutral depth."""
    t = _trigger()
    t.update(0.0, 0.0)
    z = 0.0
    for i in range(1, 4):
        z -= 0.2
        t.update(z, i * 0.1)
    assert t.active
    t.reset_neutral()
    assert not t.active  # reset_neutral also disengages (fail-safe)
    # New neutral seeds at the deep position; a small jitter does not re-engage.
    assert t.update(-0.6, 0.5) is None
    assert t.update(-0.605, 0.6) is None
