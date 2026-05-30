"""Extension-based gesture state machine for the left hand.

Maps finger extension patterns from a closed-fist default to discrete button events.
Each gesture is defined by which fingers must be extended and uses a Schmitt trigger
on the extension ratio for hysteresis-based engage/release detection.

Unlike the right hand's pinch-based system (lower distance = engaged), extension
gestures engage when a finger's extension ratio goes *above* the threshold (the finger
straightens out from the fist). The Schmitt trigger is therefore "inverted": engage
fires when the signal rises above ``t_engage``, and release fires when it drops below
``t_release`` (where ``t_release < t_engage``).

Gesture types:
    - ``thumb_out``: Thumb extended laterally (independent of other fingers).
    - ``index_only``: Index finger extended, others curled.
    - ``middle_only``: Middle finger extended, others curled.
    - ``index_middle``: Both index and middle extended (peace sign).
    - ``ring_only``: Ring finger extended, others curled.
    - ``pinky_only``: Pinky finger extended, others curled.

Pulse gestures (inventory, throw_item, switch_offhand) fire a single KEY_DOWN on
engage; the pipeline converts these to key taps (immediate key_down + key_up).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from minecraft_cv.gestures.finger_state import FingerState, finger_extensions
from minecraft_cv.gestures.schmitt import KEY_DOWN, KEY_UP

Hand = str  # "left" | "right"


class ExtensionThresholdSpec(Protocol):
    """Duck-typed extension gesture spec; satisfied by config models."""

    @property
    def type(self) -> str:
        """The finger combination pattern (e.g. 'thumb_out', 'index_only')."""

    @property
    def t_engage(self) -> float:
        """Extension ratio above which the gesture engages."""

    @property
    def t_release(self) -> float:
        """Extension ratio below which the gesture releases."""

    @property
    def pulse(self) -> bool:
        """If True, this gesture fires a single tap rather than a sustained hold."""


@dataclass(frozen=True)
class GestureEvent:
    """A discrete button transition produced by an extension state machine.

    Attributes:
        gesture: Logical gesture name, e.g. ``"jump"``, ``"sneak"``.
        action: ``"KEY_DOWN"`` or ``"KEY_UP"``.
        hand: ``"left"`` or ``"right"``.
    """

    gesture: str
    action: str
    hand: Hand


class _ExtensionState(Enum):
    """State of a single extension Schmitt trigger."""

    RELEASED = "RELEASED"
    HOLDING = "HOLDING"


# Maps gesture type to a function that extracts the relevant signal from FingerState.
# For single-finger gestures, the signal is the extension ratio of that finger.
# For multi-finger gestures (e.g. index_middle), the signal is the minimum of the
# required fingers' ratios — all must be extended for the gesture to engage.
_SIGNAL_EXTRACTORS: dict[str, list[str]] = {
    "thumb_out": ["thumb_ext"],
    "index_only": ["index_ext"],
    "middle_only": ["middle_ext"],
    "index_middle": ["index_ext", "middle_ext"],
    "ring_only": ["ring_ext"],
    "pinky_only": ["pinky_ext"],
}

# For "only" gestures, these are the fingers that must remain CURLED (below release
# threshold). If any of these are extended, the gesture is suppressed to prevent
# false positives (e.g., a full open hand shouldn't trigger every single-finger gesture).
_EXCLUSION_FINGERS: dict[str, list[str]] = {
    "thumb_out": [],  # thumb is independent — no exclusions
    "index_only": ["middle_ext", "ring_ext", "pinky_ext"],
    "middle_only": ["index_ext", "ring_ext", "pinky_ext"],
    "index_middle": ["ring_ext", "pinky_ext"],
    "ring_only": ["index_ext", "middle_ext", "pinky_ext"],
    "pinky_only": ["index_ext", "middle_ext", "ring_ext"],
}


@dataclass
class _GestureTrigger:
    """Internal state for one extension gesture's Schmitt trigger."""

    name: str
    gesture_type: str
    t_engage: float
    t_release: float
    pulse: bool
    state: _ExtensionState = _ExtensionState.RELEASED

    def update(self, finger_state: FingerState) -> str | None:
        """Feed one frame of finger state; return a transition token on state change.

        Args:
            finger_state: The current frame's finger extension ratios.

        Returns:
            ``KEY_DOWN`` on engage, ``KEY_UP`` on release, else ``None``.
        """
        signal = self._extract_signal(finger_state)
        excluded = self._check_exclusion(finger_state)

        if self.state is _ExtensionState.RELEASED:
            if signal > self.t_engage and not excluded:
                self.state = _ExtensionState.HOLDING
                return KEY_DOWN
        elif self.state is _ExtensionState.HOLDING:
            # Release if signal drops below release threshold OR exclusion fingers fire.
            if signal < self.t_release or excluded:
                self.state = _ExtensionState.RELEASED
                return KEY_UP
        return None

    def reset(self) -> str | None:
        """Force to RELEASED, returning KEY_UP if was holding."""
        if self.state is _ExtensionState.HOLDING:
            self.state = _ExtensionState.RELEASED
            return KEY_UP
        return None

    def _extract_signal(self, fs: FingerState) -> float:
        """Get the primary signal value from finger state."""
        fields = _SIGNAL_EXTRACTORS[self.gesture_type]
        values = [getattr(fs, f) for f in fields]
        # For multi-finger gestures, ALL must be extended -> use min.
        return min(values)

    def _check_exclusion(self, fs: FingerState) -> bool:
        """Check if any exclusion fingers are extended (would suppress this gesture)."""
        excl_fields = _EXCLUSION_FINGERS[self.gesture_type]
        if not excl_fields:
            return False
        # Gesture is excluded if any non-required finger is well above the release
        # threshold (indicating it's clearly extended, not just noise).
        for f in excl_fields:
            if getattr(fs, f) > self.t_engage:
                return True
        return False


class ExtensionStateMachine:
    """Resolves one hand's extension-based gestures into :class:`GestureEvent` lists."""

    def __init__(
        self, hand: Hand, gestures: Mapping[str, ExtensionThresholdSpec]
    ) -> None:
        """Build a state machine for one hand.

        Args:
            hand: ``"left"`` or ``"right"`` — stamped onto every emitted event.
            gestures: Map of gesture name -> threshold spec.
        """
        self.hand = hand
        self._triggers: list[_GestureTrigger] = []
        self._pulse_gestures: set[str] = set()
        for name, spec in gestures.items():
            if spec.type not in _SIGNAL_EXTRACTORS:
                raise ValueError(
                    f"Gesture {name!r} has unsupported type {spec.type!r}; "
                    f"supported types: {sorted(_SIGNAL_EXTRACTORS)}"
                )
            self._triggers.append(
                _GestureTrigger(
                    name=name,
                    gesture_type=spec.type,
                    t_engage=spec.t_engage,
                    t_release=spec.t_release,
                    pulse=spec.pulse,
                )
            )
            if spec.pulse:
                self._pulse_gestures.add(name)

    def update(self, landmarks: np.ndarray) -> list[GestureEvent]:
        """Advance every gesture trigger with one frame of landmarks.

        Args:
            landmarks: ``(21, 3)`` float landmark array for this hand.

        Returns:
            The list of transitions that fired this frame (possibly empty).
        """
        fs = finger_extensions(landmarks)
        events: list[GestureEvent] = []
        for trigger in self._triggers:
            transition = trigger.update(fs)
            if transition is not None:
                events.append(
                    GestureEvent(gesture=trigger.name, action=transition, hand=self.hand)
                )
        return events

    def reset(self) -> list[GestureEvent]:
        """Release every currently-held gesture (fail-safe on tracking loss).

        Returns:
            One ``KEY_UP`` event per gesture that had been holding.
        """
        events: list[GestureEvent] = []
        for trigger in self._triggers:
            transition = trigger.reset()
            if transition is not None:
                events.append(
                    GestureEvent(gesture=trigger.name, action=transition, hand=self.hand)
                )
        return events

    @property
    def held(self) -> set[str]:
        """Names of gestures currently in the HOLDING state."""
        return {
            t.name for t in self._triggers if t.state is _ExtensionState.HOLDING
        }

    @property
    def pulse_gestures(self) -> frozenset[str]:
        """Names of gestures configured as pulse (one-shot tap) actions."""
        return frozenset(self._pulse_gestures)


__all__ = [
    "ExtensionStateMachine",
    "ExtensionThresholdSpec",
    "GestureEvent",
]
