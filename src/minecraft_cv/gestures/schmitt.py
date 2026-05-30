"""Schmitt trigger — the hysteresis gate behind every discrete pinch gesture.

This is the single most important correctness primitive in the project. Each pinch state
machine is a Schmitt trigger operating on a **normalized** thumb-to-fingertip distance
(unitless ratio, scale-invariant). Two thresholds with a gap between them swallow CV frame
jitter so a finger hovering near the engage point does not chatter.

Hard invariant #1: ``t_release > t_engage`` strictly. Asserted at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PinchState(Enum):
    """State of a single pinch Schmitt trigger."""

    RELEASED = "RELEASED"
    HOLDING = "HOLDING"


# Transition tokens returned by ``update``/``reset``.
KEY_DOWN = "KEY_DOWN"
KEY_UP = "KEY_UP"


@dataclass
class SchmittTrigger:
    """Hysteresis gate for one discrete pinch gesture.

    The thresholds operate on the normalized thumb-to-fingertip distance (distance divided
    by the wrist->middle-MCP hand span). Because the input is already scale-invariant, the
    thresholds are unitless ratios that never need recalibration when the user moves closer
    to or farther from the camera.

    Attributes:
        t_engage: Distance strictly below which a RELEASED trigger engages (KEY_DOWN).
        t_release: Distance strictly above which a HOLDING trigger releases (KEY_UP). Must
            be strictly greater than ``t_engage`` (the hysteresis band).
        state: Current :class:`PinchState`. Starts RELEASED.

    Raises:
        ValueError: If ``t_release <= t_engage`` (would reintroduce jitter chatter).
    """

    t_engage: float
    t_release: float
    engage_frames: int = 2
    release_frames: int = 2
    state: PinchState = PinchState.RELEASED

    def __post_init__(self) -> None:
        if not self.t_release > self.t_engage:
            raise ValueError(
                f"t_release ({self.t_release}) must be strictly greater than "
                f"t_engage ({self.t_engage}); equal/inverted thresholds reintroduce the "
                "jitter chatter the Schmitt trigger exists to prevent (hard invariant #1)."
            )
        self._consecutive_engage: int = 0
        self._consecutive_release: int = 0

    def update(self, distance: float) -> str | None:
        """Feed one normalized distance; return a transition token on a state change.

        Args:
            distance: Normalized (unitless) thumb-to-fingertip distance for this frame.

        Returns:
            ``KEY_DOWN`` on RELEASED->HOLDING, ``KEY_UP`` on HOLDING->RELEASED, else None.
        """
        if self.state is PinchState.RELEASED:
            if distance < self.t_engage:
                self._consecutive_engage += 1
                if self._consecutive_engage >= self.engage_frames:
                    self.state = PinchState.HOLDING
                    self._consecutive_engage = 0
                    self._consecutive_release = 0
                    return KEY_DOWN
            else:
                self._consecutive_engage = 0
        elif self.state is PinchState.HOLDING:
            if distance > self.t_release:
                self._consecutive_release += 1
                if self._consecutive_release >= self.release_frames:
                    self.state = PinchState.RELEASED
                    self._consecutive_release = 0
                    self._consecutive_engage = 0
                    return KEY_UP
            else:
                self._consecutive_release = 0
        return None

    def reset(self) -> str | None:
        """Force the trigger to RELEASED, returning ``KEY_UP`` if it had been holding.

        Used by the tracking-loss fail-safe so a dropout can never leave a key stuck down
        (no bunny-hopping or sneak-lock). Idempotent: a no-op (returns None) if already
        RELEASED.
        """
        self._consecutive_engage = 0
        self._consecutive_release = 0
        if self.state is PinchState.HOLDING:
            self.state = PinchState.RELEASED
            return KEY_UP
        return None
