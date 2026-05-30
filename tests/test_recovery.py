"""Deterministic tests for the per-hand tracking-loss recovery controller (Task 5).

Pure time-driven state machine: ``update(present, now)`` where ``now`` is supplied explicitly
(seconds), so the >100 ms hard-flush and 500 ms re-entry stabilization windows are tested
without a real clock.
"""

from __future__ import annotations

from minecraft_cv.recovery import PHASE_ABSENT, PHASE_NORMAL, PHASE_STABILIZING, HandRecovery


def _rec() -> HandRecovery:
    return HandRecovery(dropout_flush_ms=100.0, stabilization_ms=500.0)


def test_present_hand_is_normal() -> None:
    r = _rec()
    d = r.update(present=True, now=0.0)
    assert d.phase == PHASE_NORMAL
    assert d.present and d.emit and d.track and not d.flush


def test_brief_absence_no_flush_no_stabilization() -> None:
    """A sub-100 ms blip never flushes, and re-entry resumes emitting immediately."""
    r = _rec()
    r.update(present=True, now=0.0)
    d_absent = r.update(present=False, now=0.010)  # 10 ms absent
    assert d_absent.phase == PHASE_ABSENT
    assert not d_absent.flush
    d_back = r.update(present=True, now=0.020)
    assert d_back.phase == PHASE_NORMAL
    assert d_back.emit


def test_long_absence_flushes_once_then_stays_absent() -> None:
    r = _rec()
    r.update(present=True, now=0.0)
    r.update(present=False, now=0.050)  # 50 ms -> no flush yet
    d_flush = r.update(present=False, now=0.150)  # 150 ms > 100 ms -> flush
    assert d_flush.flush and d_flush.phase == PHASE_ABSENT
    # Flush is one-shot: subsequent absent frames do not re-flush.
    d_more = r.update(present=False, now=0.300)
    assert not d_more.flush


def test_reentry_after_flush_enters_stabilization() -> None:
    r = _rec()
    r.update(present=True, now=0.0)
    r.update(present=False, now=0.150)  # flush
    d_back = r.update(present=True, now=0.200)
    # Returning hand is tracked (to re-seed neutral) but must NOT emit yet.
    assert d_back.phase == PHASE_STABILIZING
    assert d_back.present and d_back.track and not d_back.emit and not d_back.flush


def test_stabilization_window_expires_then_emits() -> None:
    r = _rec()
    r.update(present=True, now=0.0)
    r.update(present=False, now=0.150)  # flush at 150 ms
    r.update(present=True, now=0.200)  # return -> stabilize until 0.700
    assert r.update(present=True, now=0.600).phase == PHASE_STABILIZING
    assert r.update(present=True, now=0.699).phase == PHASE_STABILIZING
    d = r.update(present=True, now=0.701)  # window elapsed
    assert d.phase == PHASE_NORMAL and d.emit


def test_zero_stabilization_emits_immediately_after_flush() -> None:
    r = HandRecovery(dropout_flush_ms=100.0, stabilization_ms=0.0)
    r.update(present=True, now=0.0)
    r.update(present=False, now=0.150)  # flush
    d = r.update(present=True, now=0.200)
    assert d.phase == PHASE_NORMAL and d.emit


def test_absence_during_stabilization_can_flush_again() -> None:
    """If the hand drops out again mid-stabilization long enough, it re-flushes."""
    r = _rec()
    r.update(present=True, now=0.0)
    r.update(present=False, now=0.150)  # flush #1
    r.update(present=True, now=0.200)  # last-present 0.200, stabilizing until 0.700
    # Drops out again; 50 ms later is still under the flush window...
    assert not r.update(present=False, now=0.250).flush
    # ...but 200 ms after the last-present frame it crosses the flush threshold again.
    assert r.update(present=False, now=0.400).flush
