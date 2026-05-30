"""Depth-velocity Sprint trigger (Task 2).

Holding a static posture to sprint is exhausting (Gorilla Arm). Instead, a quick *forward
push* of the left hand toward the camera engages Sprint, which is then held as long as the
hand stays pushed forward and released when the hand retreats. The pipeline maps an engage to
``Ctrl`` held alongside ``W`` (forward), matching Minecraft's sprint.

The trigger consumes the left-hand anchor's relative-depth coordinate ``z`` plus a timestamp.

Frame of reference: MediaPipe ``z`` is depth relative to the wrist, in the same scale as ``x``;
**negative = closer to the camera**. A forward push makes ``z`` *decrease*, so forward velocity
is ``-dz/dt`` (positive when pushing toward the camera). ``z`` is the least reliable landmark
axis, so this gate is deliberately conservative (sustained-velocity engage + positional hold
with hysteresis) and is **disabled by default** in config.

This module is pure and allocation-free on the hot path: it holds three scalars of state and
performs no per-frame allocation. Timestamps are supplied explicitly, so it is fully
unit-testable without a clock (see ``tests/test_sprint_velocity.py``).
"""

from __future__ import annotations

# Transition tokens (mirrors the KEY_DOWN/KEY_UP vocabulary of the Schmitt machines).
ENGAGE = "ENGAGE"
RELEASE = "RELEASE"


class SprintVelocityTrigger:
    """Engages on a sustained forward push; holds while forward; releases on retreat.

    The engage condition is a *velocity* gate (forward speed above ``v_sprint`` for
    ``trigger_frames`` consecutive frames). The hold/release condition is a *positional* gate
    relative to a stored neutral depth, with hysteresis so a brief wobble does not drop sprint.
    """

    def __init__(
        self,
        v_sprint: float,
        trigger_frames: int = 3,
        release_margin: float = 0.02,
        enabled: bool = False,
    ) -> None:
        """Configure the trigger.

        Args:
            v_sprint: Forward-velocity threshold in normalized-``z`` units per second. The
                forward push must exceed this to count toward engaging.
            trigger_frames: Consecutive above-threshold frames required to engage (>=1). This
                is the "over N frames" debounce from the task spec (default 3).
            release_margin: While sprinting, the hand is considered to have retreated (and
                sprint releases) once ``z`` rises back above ``neutral_z - release_margin``.
                Units: normalized ``z``. Provides the hysteresis band against depth jitter.
            enabled: If False, :meth:`update` is inert and the trigger never engages.
        """
        self.v_sprint = float(v_sprint)
        self.trigger_frames = max(1, int(trigger_frames))
        self.release_margin = float(release_margin)
        self.enabled = bool(enabled)
        self._neutral_z: float | None = None
        self._prev_z: float | None = None
        self._prev_t: float | None = None
        self._count = 0
        self._active = False

    @property
    def active(self) -> bool:
        """True while sprint is engaged."""
        return self._active

    def reset_neutral(self) -> None:
        """Forget the neutral depth and disengage (recenter macro / dropout fail-safe).

        The next :meth:`update` reseeds the neutral from its sample, exactly like the spatial
        joystick's recenter. Disengaging here is a fail-safe so a dropout never leaves Ctrl
        latched on a stale neutral.
        """
        self._neutral_z = None
        self._prev_z = None
        self._prev_t = None
        self._count = 0
        self._active = False

    def reset(self) -> str | None:
        """Force release (tracking-loss fail-safe). Returns ``RELEASE`` if it was active.

        Idempotent: a no-op (returns ``None``) when already released. The velocity history is
        cleared so re-entry starts fresh.
        """
        self._prev_z = None
        self._prev_t = None
        self._count = 0
        if self._active:
            self._active = False
            return RELEASE
        return None

    def update(self, z: float, timestamp: float) -> str | None:
        """Feed one frame's anchor depth; return a transition token on a state change.

        Args:
            z: The left-hand anchor's relative-depth coordinate this frame (normalized ``z``,
                negative = closer to camera).
            timestamp: Monotonic time of this sample in seconds. Must be non-decreasing.

        Returns:
            ``ENGAGE`` on a release->sprint transition, ``RELEASE`` on sprint->release, else
            ``None``.
        """
        z = float(z)
        if not self.enabled:
            return None

        if self._neutral_z is None or self._prev_z is None or self._prev_t is None:
            # First sample after construction / reset seeds the neutral and velocity history.
            self._neutral_z = z
            self._prev_z = z
            self._prev_t = float(timestamp)
            return None

        dt = float(timestamp) - self._prev_t
        if dt <= 0.0:
            # Non-advancing clock: cannot estimate velocity. Ignore (no div-by-zero).
            return None

        forward_velocity = -(z - self._prev_z) / dt  # positive = pushing toward camera
        self._prev_z = z
        self._prev_t = float(timestamp)

        if not self._active:
            if forward_velocity > self.v_sprint:
                self._count += 1
            else:
                self._count = 0
            if self._count >= self.trigger_frames:
                self._count = 0
                self._active = True
                return ENGAGE
            return None

        # Active: hold while the hand stays pushed forward (z below neutral - release_margin);
        # release once it retreats back across that line.
        if z > self._neutral_z - self.release_margin:
            self._active = False
            return RELEASE
        return None


__all__ = ["ENGAGE", "RELEASE", "SprintVelocityTrigger"]
