"""Config-driven gesture detector registry.

This module lets the pipeline resolve both hands through the same state-machine surface. Each
gesture config names a detector implementation (currently ``pinch``, ``curl_only``,
``curl_combo``, or ``extension_combo``), its thresholds, mode, and optional conflict group.
The pipeline only sees logical gesture events; changing which detector drives a binding is a
config/model change rather than a pipeline branch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from minecraft_cv.gestures.finger_state import FingerState, finger_extensions
from minecraft_cv.gestures.pinch import normalized_distances
from minecraft_cv.gestures.schmitt import KEY_DOWN, KEY_UP, PinchState, SchmittTrigger

Hand = str
_PINCH_FINGERS = ("index", "middle", "ring", "pinky")
_FINGER_FIELDS = {
    "thumb": "thumb_ext",
    "index": "index_ext",
    "middle": "middle_ext",
    "ring": "ring_ext",
    "pinky": "pinky_ext",
}


class GestureDetectorSpec(Protocol):
    """Duck-typed config for one detector-backed gesture."""

    @property
    def detector(self) -> str:
        """Detector name, e.g. ``"pinch"``, ``"curl_combo"``, or ``"extension_combo"``."""

    @property
    def finger(self) -> str:
        """Primary finger for the detector."""

    @property
    def t_engage(self) -> float:
        """Signal value below which the detector engages."""

    @property
    def t_release(self) -> float:
        """Signal value above which the detector releases."""

    @property
    def mode(self) -> str:
        """Gesture emission mode: ``"hold"`` or ``"toggle"``."""

    @property
    def open_fingers(self) -> tuple[str, ...]:
        """For curl detectors, optional fingers that must remain open."""

    @property
    def open_threshold(self) -> float:
        """Extension ratio above which an open finger counts as open."""

    @property
    def curl_fingers(self) -> tuple[str, ...]:
        """For ``curl_combo``, every listed finger must remain curled."""

    @property
    def extension_fingers(self) -> tuple[str, ...]:
        """For ``extension_combo``, every listed finger must remain extended."""

    @property
    def conflict_group(self) -> str | None:
        """Optional group in which only the strongest active gesture may hold."""

    @property
    def suppresses(self) -> tuple[str, ...]:
        """Other gesture names to suppress while this gesture is held."""


@dataclass(frozen=True)
class GestureEvent:
    """A logical gesture transition from the registry state machine."""

    gesture: str
    action: str
    hand: Hand


@dataclass
class _DetectorState:
    """Internal detector state for one configured gesture."""

    name: str
    spec: GestureDetectorSpec
    trigger: SchmittTrigger
    output_held: bool = False

    def signal(self, landmarks: np.ndarray, fs: FingerState | None) -> float:
        """Return the lower-is-more-engaged signal for this detector."""
        if self.spec.detector == "pinch":
            return normalized_distances(landmarks)[self.spec.finger]
        if self.spec.detector == "curl_only":
            if fs is None:
                fs = finger_extensions(landmarks)
            return float(getattr(fs, _FINGER_FIELDS[self.spec.finger]))
        if self.spec.detector == "curl_combo":
            if fs is None:
                fs = finger_extensions(landmarks)
            fingers = self.spec.curl_fingers or (self.spec.finger,)
            return max(float(getattr(fs, _FINGER_FIELDS[finger])) for finger in fingers)
        if self.spec.detector == "extension_combo":
            if fs is None:
                fs = finger_extensions(landmarks)
            fingers = self.spec.extension_fingers or (self.spec.finger,)
            # Convert the high-is-engaged extension signal into the lower-is-engaged
            # convention used by SchmittTrigger.
            return -min(float(getattr(fs, _FINGER_FIELDS[finger])) for finger in fingers)
        raise ValueError(f"Unsupported gesture detector: {self.spec.detector!r}")

    def gate_open(self, fs: FingerState | None) -> bool:
        """Return whether non-primary conditions allow this detector to engage/hold."""
        if self.spec.detector == "extension_combo":
            if fs is None:
                return False
            for finger in self.spec.curl_fingers:
                if float(getattr(fs, _FINGER_FIELDS[finger])) >= self.spec.t_release:
                    return False
            return True
        if self.spec.detector not in ("curl_only", "curl_combo"):
            return True
        if fs is None:
            return False
        for finger in self.spec.open_fingers:
            if getattr(fs, _FINGER_FIELDS[finger]) <= self.spec.open_threshold:
                return False
        return True

    def apply_transition(self, transition: str | None) -> GestureEvent | None:
        """Translate physical detector transitions into configured output events."""
        if transition is None:
            return None
        if self.spec.mode == "hold":
            return self._apply_hold_transition(transition)
        if self.spec.mode == "toggle":
            return self._apply_toggle_transition(transition)
        raise ValueError(f"Unsupported gesture mode: {self.spec.mode!r}")

    def _apply_hold_transition(self, transition: str) -> GestureEvent | None:
        if transition == KEY_DOWN and not self.output_held:
            self.output_held = True
            return GestureEvent(self.name, KEY_DOWN, "")
        if transition != KEY_DOWN and self.output_held:
            self.output_held = False
            return GestureEvent(self.name, KEY_UP, "")
        return None

    def _apply_toggle_transition(self, transition: str) -> GestureEvent | None:
        if transition != KEY_DOWN:
            return None
        self.output_held = not self.output_held
        action = KEY_DOWN if self.output_held else KEY_UP
        return GestureEvent(self.name, action, "")

    def force_release(self) -> str | None:
        """Reset the physical trigger and release the logical output if it is down."""
        self.trigger.reset()
        if self.output_held:
            self.output_held = False
            return KEY_UP
        return None

    @property
    def required_curl_count(self) -> int:
        """Number of fingers this detector requires curled; used as tie-break specificity."""
        if self.spec.detector == "curl_combo":
            return len(self.spec.curl_fingers or (self.spec.finger,))
        if self.spec.detector == "extension_combo":
            return len(self.spec.extension_fingers or (self.spec.finger,)) + len(
                self.spec.curl_fingers
            )
        return 1

    @property
    def physically_held(self) -> bool:
        """Whether this detector's Schmitt trigger is holding."""
        return self.trigger.state is PinchState.HOLDING

    @property
    def held(self) -> bool:
        """Whether this detector's output is currently holding a game action."""
        return self.output_held


class GestureStateMachine:
    """Resolve one hand's detector-backed gesture map into transition events."""

    def __init__(self, hand: Hand, gestures: Mapping[str, GestureDetectorSpec]) -> None:
        """Build the state machine.

        Args:
            hand: ``"left"`` or ``"right"`` label stamped on emitted events.
            gestures: Map of logical gesture name to detector config.
        """
        self.hand = hand
        self._detectors: list[_DetectorState] = []
        self._conflict_winners: dict[str, str] = {}
        for name, spec in gestures.items():
            if spec.detector == "pinch" and spec.finger not in _PINCH_FINGERS:
                raise ValueError(
                    f"Pinch gesture {name!r} targets unsupported finger {spec.finger!r}"
                )
            if spec.detector == "curl_only" and spec.finger not in _FINGER_FIELDS:
                raise ValueError(
                    f"Curl gesture {name!r} targets unsupported finger {spec.finger!r}"
                )
            if spec.detector == "curl_combo":
                fingers = spec.curl_fingers or (spec.finger,)
                bad = [
                    finger
                    for finger in fingers
                    if finger not in _FINGER_FIELDS or finger == "thumb"
                ]
                if bad:
                    raise ValueError(
                        f"Curl-combo gesture {name!r} targets unsupported fingers {bad!r}"
                    )
            if spec.detector == "extension_combo":
                extension_fingers = spec.extension_fingers or (spec.finger,)
                bad_extension = [
                    finger for finger in extension_fingers if finger not in _FINGER_FIELDS
                ]
                bad_curl = [
                    finger
                    for finger in spec.curl_fingers
                    if finger not in _FINGER_FIELDS or finger == "thumb"
                ]
                if bad_extension or bad_curl:
                    raise ValueError(
                        f"Extension-combo gesture {name!r} has unsupported fingers: "
                        f"extend={bad_extension!r} curl={bad_curl!r}"
                    )
            trigger_engage = spec.t_engage
            trigger_release = spec.t_release
            if spec.detector == "extension_combo":
                trigger_engage = -spec.t_engage
                trigger_release = -spec.t_release
            self._detectors.append(
                _DetectorState(
                    name=name,
                    spec=spec,
                    trigger=SchmittTrigger(
                        t_engage=trigger_engage,
                        t_release=trigger_release,
                    ),
                )
            )

    def update(self, landmarks: np.ndarray) -> list[GestureEvent]:
        """Advance every detector with one frame of landmarks."""
        fs = finger_extensions(landmarks)
        events: list[GestureEvent] = []
        signals: dict[str, float] = {}

        for detector in self._detectors:
            signal = detector.signal(landmarks, fs)
            signals[detector.name] = signal
            if not detector.gate_open(fs):
                transition = detector.trigger.reset()
            else:
                transition = detector.trigger.update(signal)
            event = detector.apply_transition(transition)
            if event is not None:
                events.append(GestureEvent(event.gesture, event.action, self.hand))

        output = self._resolve_conflicts(events, signals)
        return self._resolve_suppressions(output)

    def _resolve_conflicts(
        self, events: list[GestureEvent], signals: dict[str, float]
    ) -> list[GestureEvent]:
        """Enforce conflict groups by holding only the strongest active detector."""
        output = list(events)
        groups = {
            detector.spec.conflict_group
            for detector in self._detectors
            if detector.spec.conflict_group is not None
        }
        for group in groups:
            if group is None:
                continue
            members = [
                detector
                for detector in self._detectors
                if detector.spec.conflict_group == group and detector.physically_held
            ]
            if not members:
                self._conflict_winners.pop(group, None)
                continue
            winner = self._choose_winner(group, members, signals)
            self._conflict_winners[group] = winner.name
            for detector in members:
                if detector.name == winner.name:
                    continue
                had_key_down_this_frame = any(
                    event.gesture == detector.name
                    and event.action == KEY_DOWN
                    and event.hand == self.hand
                    for event in output
                )
                transition = detector.force_release()
                output = [
                    event
                    for event in output
                    if not (
                        event.gesture == detector.name
                        and event.action == KEY_DOWN
                        and event.hand == self.hand
                    )
                ]
                if transition is not None and not had_key_down_this_frame:
                    output.append(GestureEvent(detector.name, transition, self.hand))
        return output

    def _resolve_suppressions(self, events: list[GestureEvent]) -> list[GestureEvent]:
        """Release configured lower-priority gestures while suppressors are held."""
        output = list(events)
        detectors = {detector.name: detector for detector in self._detectors}
        for suppressor in self._detectors:
            if not suppressor.held:
                continue
            for target_name in suppressor.spec.suppresses:
                target = detectors.get(target_name)
                if target is None:
                    continue
                had_key_down_this_frame = any(
                    event.gesture == target.name
                    and event.action == KEY_DOWN
                    and event.hand == self.hand
                    for event in output
                )
                transition = target.force_release()
                output = [
                    event
                    for event in output
                    if not (
                        event.gesture == target.name
                        and event.action == KEY_DOWN
                        and event.hand == self.hand
                    )
                ]
                if transition is not None and not had_key_down_this_frame:
                    output.append(GestureEvent(target.name, transition, self.hand))
        return output

    def _choose_winner(
        self, group: str, members: list[_DetectorState], signals: dict[str, float]
    ) -> _DetectorState:
        """Pick the strongest active detector; prefer current holder on equal strength."""
        current = self._conflict_winners.get(group)
        best_signal = min(signals[m.name] for m in members)
        tied = [m for m in members if abs(signals[m.name] - best_signal) <= 1e-9]
        best_specificity = max(m.required_curl_count for m in tied)
        tied = [m for m in tied if m.required_curl_count == best_specificity]
        if current is not None:
            for detector in tied:
                if detector.name == current:
                    return detector
        return tied[0]

    def reset(self) -> list[GestureEvent]:
        """Release all held gestures and clear conflict state."""
        events: list[GestureEvent] = []
        for detector in self._detectors:
            transition = detector.force_release()
            if transition is not None:
                events.append(GestureEvent(detector.name, transition, self.hand))
        self._conflict_winners.clear()
        return events

    @property
    def held(self) -> set[str]:
        """Names of gestures currently holding."""
        return {detector.name for detector in self._detectors if detector.held}

    @property
    def pulse_gestures(self) -> frozenset[str]:
        """The registry rewrite uses held gestures only; kept for pipeline compatibility."""
        return frozenset()


__all__ = ["GestureDetectorSpec", "GestureEvent", "GestureStateMachine"]
