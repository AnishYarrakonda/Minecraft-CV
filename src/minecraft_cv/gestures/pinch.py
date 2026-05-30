"""Per-hand pinch-bitmask gesture resolver.

Maps a stream of MediaPipe hand landmarks to discrete button events via one independent
Schmitt trigger per configured gesture. Distances are computed once per frame with a single
vectorized NumPy call (no Python loop over landmarks) and normalized by hand scale so the
thresholds are camera-distance invariant.

MVP gestures (per ``.claude/rules/gestures.md``): Left hand jump (thumb->index) + sneak
(thumb->middle); Right hand attack (thumb->index) + use (thumb->middle). Ring/pinky hotbar
and the full-fist inventory mode switch are V2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from minecraft_cv.gestures.schmitt import KEY_DOWN, KEY_UP, SchmittTrigger

# --- MediaPipe Hands landmark indices (single source of truth) ----------------
WRIST = 0
THUMB_TIP = 4
MIDDLE_MCP = 9

# Fingertip landmark indices in canonical finger order.
_FINGER_ORDER: tuple[str, ...] = ("index", "middle", "ring", "pinky")
_TIP_INDICES = np.array([8, 12, 16, 20], dtype=np.intp)  # index, middle, ring, pinky

Hand = str  # "left" | "right"


class ThresholdSpec(Protocol):
    """Duck-typed gesture spec; satisfied by both :class:`GestureSpec` and config models.

    Declared with read-only properties (not writable attributes) so the value type stays
    covariant — i.e. ``config.GestureThresholds`` whose ``finger`` is a ``Literal[...]`` is
    accepted where a ``str`` is expected.
    """

    @property
    def finger(self) -> str: ...

    @property
    def t_engage(self) -> float: ...

    @property
    def t_release(self) -> float: ...


@dataclass(frozen=True)
class GestureSpec:
    """A single gesture's finger + Schmitt thresholds (decoupled from config/pydantic)."""

    finger: str
    t_engage: float
    t_release: float


@dataclass(frozen=True)
class GestureEvent:
    """A discrete button transition produced by a pinch state machine.

    Attributes:
        gesture: Logical gesture name, e.g. ``"jump"``, ``"attack"``.
        action: ``"KEY_DOWN"`` or ``"KEY_UP"``.
        hand: ``"left"`` or ``"right"``.
    """

    gesture: str
    action: str
    hand: Hand


def normalized_distances(landmarks: np.ndarray) -> dict[str, float]:
    """Thumb-to-fingertip distances normalized by hand scale.

    The distances are divided by the wrist->middle-MCP span, making them unitless ratios
    that are invariant to how far the hand is from the camera.

    Args:
        landmarks: ``(21, 3)`` float array of ``(x, y, z)`` hand keypoints. ``x``/``y`` are
            normalized to ``[0, 1]`` in frame space; ``z`` is relative depth.

    Returns:
        Mapping of finger name (``"index"``/``"middle"``/``"ring"``/``"pinky"``) to its
        normalized thumb-pinch distance ratio.
    """
    scale = float(np.linalg.norm(landmarks[MIDDLE_MCP] - landmarks[WRIST])) or 1e-6
    thumb = landmarks[THUMB_TIP]
    tips = landmarks[_TIP_INDICES]  # (4, 3)
    raw = np.linalg.norm(tips - thumb, axis=1) / scale  # (4,) — one vectorized call
    return {name: float(d) for name, d in zip(_FINGER_ORDER, raw)}


class PinchStateMachine:
    """Resolves one hand's configured pinch gestures into :class:`GestureEvent` lists."""

    def __init__(self, hand: Hand, gestures: Mapping[str, ThresholdSpec]) -> None:
        """Build a state machine for one hand.

        Args:
            hand: ``"left"`` or ``"right"`` — stamped onto every emitted event.
            gestures: Map of gesture name -> threshold spec. Each spec's ``finger`` must be
                one of ``"index"``/``"middle"``/``"ring"``/``"pinky"`` (the full-fist
                ``"inventory"`` gesture is a V2 mode switch handled elsewhere).

        Raises:
            ValueError: If a gesture targets an unsupported finger, or a spec violates
                ``t_release > t_engage`` (raised by :class:`SchmittTrigger`).
        """
        self.hand = hand
        self._fingers: dict[str, str] = {}
        self._triggers: dict[str, SchmittTrigger] = {}
        for name, spec in gestures.items():
            if spec.finger not in _FINGER_ORDER:
                raise ValueError(
                    f"Gesture {name!r} targets unsupported finger {spec.finger!r}; "
                    f"MVP pinch gestures use one of {_FINGER_ORDER}."
                )
            self._fingers[name] = spec.finger
            self._triggers[name] = SchmittTrigger(
                t_engage=spec.t_engage, t_release=spec.t_release
            )

    @classmethod
    def from_thresholds(
        cls, hand: Hand, gestures: Mapping[str, ThresholdSpec]
    ) -> PinchStateMachine:
        """Alias constructor for building from config ``GestureThresholds`` objects."""
        return cls(hand, gestures)

    def update(self, landmarks: np.ndarray) -> list[GestureEvent]:
        """Advance every gesture trigger with one frame of landmarks.

        Args:
            landmarks: ``(21, 3)`` float landmark array for this hand.

        Returns:
            The list of transitions that fired this frame (possibly empty). Multiple
            gestures may transition in the same frame (e.g. independent fingers).
        """
        dists = normalized_distances(landmarks)
        events: list[GestureEvent] = []
        for name, trigger in self._triggers.items():
            transition = trigger.update(dists[self._fingers[name]])
            if transition is not None:
                events.append(GestureEvent(gesture=name, action=transition, hand=self.hand))
        return events

    def reset(self) -> list[GestureEvent]:
        """Release every currently-held gesture (fail-safe on tracking loss).

        Returns:
            One ``KEY_UP`` :class:`GestureEvent` per gesture that had been holding; empty
            if nothing was held. Idempotent.
        """
        events: list[GestureEvent] = []
        for name, trigger in self._triggers.items():
            transition = trigger.reset()
            if transition is not None:
                events.append(GestureEvent(gesture=name, action=transition, hand=self.hand))
        return events

    @property
    def held(self) -> set[str]:
        """Names of gestures currently in the HOLDING state."""
        from minecraft_cv.gestures.schmitt import PinchState

        return {
            name
            for name, t in self._triggers.items()
            if t.state is PinchState.HOLDING
        }


__all__ = [
    "KEY_DOWN",
    "KEY_UP",
    "GestureEvent",
    "GestureSpec",
    "PinchStateMachine",
    "ThresholdSpec",
    "normalized_distances",
]
