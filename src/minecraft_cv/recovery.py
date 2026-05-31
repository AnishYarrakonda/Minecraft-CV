"""Per-hand tracking-loss recovery: time-based hard-flush + re-entry stabilization (Task 5).

The MVP fail-safe (``gestures/safety.py``) already releases a hand's held keys the instant the
tracker reports no hand. That is necessary but not sufficient for two failure modes the user
actually hits:

1. **Long dropout** (a hand leaves the frame while ``W`` is held). Releasing keys per frame
   already covers stuck keys, but a long absence should also recenter the joystick neutral and
   drop stale look-velocity so the controller returns to a clean slate.
2. **Violent re-entry snap.** When the hand reappears, its first coordinates are far from the
   old neutral, so the very next frame would command a huge WASD/look output — the camera
   snaps. The hand needs a brief window to re-establish a neutral origin before any input is
   emitted.

This controller encodes both as a small, clock-driven state machine, one instance per hand:

  * Continuously absent for longer than ``dropout_flush_ms`` -> emit a one-shot ``flush``
    signal (the pipeline recenters the joystick and resets the look filter).
  * Returning from such a flushed dropout -> a ``stabilization_ms`` window where the hand is
    *tracked* (coordinates feed the joystick so a neutral re-seeds) but **no input is emitted**.

It is pure: timestamps are passed in explicitly, so the 100 ms / 500 ms windows are unit-
testable without a real clock (see ``tests/test_recovery.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Per-frame phase for one hand.
PHASE_NORMAL = "NORMAL"
PHASE_ABSENT = "ABSENT"
PHASE_STABILIZING = "STABILIZING"


@dataclass(frozen=True)
class RecoveryDecision:
    """What the pipeline should do with one hand this frame.

    Attributes:
        phase: ``NORMAL`` / ``ABSENT`` / ``STABILIZING``.
        present: Whether the hand is present (post-confidence) this frame.
        emit: Whether the pipeline may emit OS input derived from this hand. False while
            absent and while stabilizing.
        track: Whether the pipeline should feed this hand's coordinates to its joystick (to
            re-establish a neutral). True whenever the hand is present (including stabilizing).
        flush: One-shot True on the frame a sustained absence first crosses
            ``dropout_flush_ms`` — the pipeline performs the hard flush (recenter + reset).
    """

    phase: str
    present: bool
    emit: bool
    track: bool
    flush: bool


class HandRecovery:
    """Time-based dropout-flush + re-entry-stabilization state machine for one hand."""

    def __init__(self, dropout_flush_ms: float = 100.0, stabilization_ms: float = 500.0) -> None:
        """Configure the recovery windows.

        Args:
            dropout_flush_ms: Continuous absence (milliseconds) after which a hard flush fires.
            stabilization_ms: Window (milliseconds) after a flushed dropout during which a
                returning hand is tracked but emits no input. ``0`` disables stabilization.
        """
        self.dropout_flush_s = float(dropout_flush_ms) / 1000.0
        self.stabilization_s = float(stabilization_ms) / 1000.0
        self._last_present_t: float | None = None
        self._flushed = False
        self._stabilize_until: float | None = None

    def reset(self) -> None:
        """Clear all timing state (e.g. on inventory-mode toggle or shutdown)."""
        self._last_present_t = None
        self._flushed = False
        self._stabilize_until = None

    def update(self, present: bool, now: float) -> RecoveryDecision:
        """Advance the state machine by one frame.

        Args:
            present: True if the tracker reported this hand (above the confidence floor).
            now: Monotonic time of this frame in seconds (non-decreasing).

        Returns:
            A :class:`RecoveryDecision` directing the pipeline for this hand.
        """
        now = float(now)
        if not present:
            # Absence is measured from the last frame the hand *was* present, so a long gap
            # between samples still counts as a sustained dropout (frames arrive ~16-33 ms
            # apart; a hand can be gone across a single missing frame for >100 ms).
            flush = False
            if (
                self._last_present_t is not None
                and not self._flushed
                and (now - self._last_present_t) >= self.dropout_flush_s
            ):
                self._flushed = True
                flush = True
            return RecoveryDecision(
                phase=PHASE_ABSENT, present=False, emit=False, track=False, flush=flush
            )

        # Present this frame.
        returned_from_flush = self._flushed
        self._last_present_t = now
        self._flushed = False
        if returned_from_flush:
            # Open a stabilization window so the neutral can re-seed before we emit anything.
            self._stabilize_until = now + self.stabilization_s if self.stabilization_s > 0 else None

        if self._stabilize_until is not None:
            if now < self._stabilize_until:
                return RecoveryDecision(
                    phase=PHASE_STABILIZING, present=True, emit=False, track=True, flush=False
                )
            self._stabilize_until = None

        return RecoveryDecision(
            phase=PHASE_NORMAL, present=True, emit=True, track=True, flush=False
        )


__all__ = [
    "PHASE_ABSENT",
    "PHASE_NORMAL",
    "PHASE_STABILIZING",
    "HandRecovery",
    "RecoveryDecision",
]
