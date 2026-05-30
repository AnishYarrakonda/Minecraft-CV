"""Two-hand pose detector that toggles Inventory Mode.

Opening the Minecraft inventory needs a *modal* shift: the cursor must roam the GUI freely
without the camera rotating, and WASD must stop. That mode is entered/left by a deliberate,
rarely-accidental two-hand pose — **both palms fully open, held briefly** — rather than a
single-hand gesture that could collide with the normal extension/pinch maps.

The detector is a small edge-triggered state machine:

  * The pose must be held for ``hold_frames`` consecutive frames before it toggles (debounce
    against momentary open hands).
  * After a toggle it disarms until the pose is *released*, so holding both palms open does
    not oscillate the mode on/off.
  * A ``cooldown_frames`` guard adds a second debounce against rapid re-toggles.

Both hands must be present for the pose to register, so a tracking dropout simply cannot
toggle the mode (it never auto-exits inventory mode on its own — the user toggles back out).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minecraft_cv.gestures.finger_state import FingerState, finger_extensions


@dataclass(frozen=True)
class InventoryToggleResult:
    """Outcome of one :meth:`InventoryModeToggle.update`.

    Attributes:
        active: The inventory-mode state *after* this frame.
        toggled: True only on the frame where the state flipped.
    """

    active: bool
    toggled: bool


def _is_open_palm(fs: FingerState, open_threshold: float, thumb_threshold: float) -> bool:
    """True if all four fingers and the thumb read as extended (a flat open palm)."""
    return (
        fs.index_ext > open_threshold
        and fs.middle_ext > open_threshold
        and fs.ring_ext > open_threshold
        and fs.pinky_ext > open_threshold
        and fs.thumb_ext > thumb_threshold
    )


class InventoryModeToggle:
    """Edge-triggered detector flipping inventory mode on a held two-hand open-palm pose."""

    def __init__(
        self,
        enabled: bool = True,
        open_threshold: float = 1.1,
        thumb_open_threshold: float = 0.9,
        hold_frames: int = 8,
        cooldown_frames: int = 20,
    ) -> None:
        """Configure the detector.

        Args:
            enabled: If False, :meth:`update` never toggles (mode stays Off).
            open_threshold: Finger extension ratio above which a finger counts as extended.
            thumb_open_threshold: Thumb lateral-extension ratio above which the thumb is open.
            hold_frames: Consecutive both-open frames required to toggle (>=1).
            cooldown_frames: Minimum frames between successive toggles (>=0).
        """
        self.enabled = bool(enabled)
        self.open_threshold = float(open_threshold)
        self.thumb_open_threshold = float(thumb_open_threshold)
        self.hold_frames = max(1, int(hold_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self._active = False
        self._pose_frames = 0
        self._armed = True
        self._cooldown = 0

    @property
    def active(self) -> bool:
        """Whether inventory mode is currently engaged."""
        return self._active

    def reset(self) -> None:
        """Clear the pose-hold/arming state (e.g. on tracking loss). Mode state is preserved."""
        self._pose_frames = 0
        self._armed = True

    def update(
        self, left_landmarks: np.ndarray | None, right_landmarks: np.ndarray | None
    ) -> InventoryToggleResult:
        """Advance the detector by one frame.

        Args:
            left_landmarks: ``(21, 3)`` left-hand landmarks, or ``None`` if absent this frame.
            right_landmarks: As above for the right hand.

        Returns:
            An :class:`InventoryToggleResult` with the post-frame mode state and a one-frame
            ``toggled`` edge flag.
        """
        if self._cooldown > 0:
            self._cooldown -= 1

        if not self.enabled or left_landmarks is None or right_landmarks is None:
            # Pose impossible this frame; re-arm so a future complete pose can toggle.
            self._pose_frames = 0
            self._armed = True
            return InventoryToggleResult(active=self._active, toggled=False)

        both_open = _is_open_palm(
            finger_extensions(left_landmarks), self.open_threshold, self.thumb_open_threshold
        ) and _is_open_palm(
            finger_extensions(right_landmarks), self.open_threshold, self.thumb_open_threshold
        )

        if not both_open:
            self._pose_frames = 0
            self._armed = True  # pose released -> ready to detect the next deliberate hold
            return InventoryToggleResult(active=self._active, toggled=False)

        self._pose_frames += 1
        if self._armed and self._pose_frames >= self.hold_frames and self._cooldown == 0:
            self._active = not self._active
            self._armed = False  # disarm until the pose is released (no oscillation)
            self._cooldown = self.cooldown_frames
            return InventoryToggleResult(active=self._active, toggled=True)

        return InventoryToggleResult(active=self._active, toggled=False)


__all__ = ["InventoryModeToggle", "InventoryToggleResult"]
