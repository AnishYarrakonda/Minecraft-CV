"""Tracking-loss fail-safe for the discrete-gesture layer.

Hard requirement (``.claude/rules/gestures.md`` -> "Tracking loss safety"): if MediaPipe
returns no hand for a side, every key held by that hand must be released. A crash or dropout
must never leave ``Space`` held (bunny-hopping forever) or ``Left Shift`` held (sneak-lock).

This module owns that guarantee for the pinch state machines. Zeroing the joystick output on
dropout is the pipeline's responsibility (it simply emits a zero vector for the absent hand).
"""

from __future__ import annotations

import numpy as np

from minecraft_cv.gestures.pinch import GestureEvent, PinchStateMachine


class TrackingLossGuard:
    """Routes per-frame landmarks to each hand's state machine, fail-safing on dropout.

    For each hand, every frame:
      * landmarks present -> :meth:`PinchStateMachine.update` (normal gesture resolution).
      * landmarks absent (``None``) -> :meth:`PinchStateMachine.reset` (release held keys).

    ``reset`` is idempotent, so a hand that stays out of frame emits a ``KEY_UP`` only on the
    first absent frame and nothing thereafter.
    """

    def __init__(self, left: PinchStateMachine, right: PinchStateMachine) -> None:
        """Wrap the left- and right-hand pinch state machines."""
        self._left = left
        self._right = right

    def process(
        self,
        left_landmarks: np.ndarray | None,
        right_landmarks: np.ndarray | None,
    ) -> list[GestureEvent]:
        """Resolve both hands for one frame, releasing keys for any absent hand.

        Args:
            left_landmarks: ``(21, 3)`` landmark array for the left hand, or ``None`` if the
                tracker reported no left hand this frame.
            right_landmarks: As above for the right hand.

        Returns:
            All gesture events produced this frame across both hands (transitions on present
            hands plus ``KEY_UP`` releases for newly-absent hands).
        """
        events: list[GestureEvent] = []
        events.extend(self._resolve(self._left, left_landmarks))
        events.extend(self._resolve(self._right, right_landmarks))
        return events

    @staticmethod
    def _resolve(
        machine: PinchStateMachine, landmarks: np.ndarray | None
    ) -> list[GestureEvent]:
        if landmarks is None:
            return machine.reset()
        return machine.update(landmarks)

    def release_all(self) -> list[GestureEvent]:
        """Release every held gesture on both hands (shutdown / crash safety).

        Returns:
            A ``KEY_UP`` event per gesture that had been holding. The pipeline emits these
            and then also calls ``emitter.release_all()`` as the OS-level backstop.
        """
        return self._left.reset() + self._right.reset()
