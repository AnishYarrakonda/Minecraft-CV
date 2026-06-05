"""Face gesture detector using blendshape scores and Schmitt triggers."""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

from minecraft_cv.gestures.registry import GestureEvent
from minecraft_cv.gestures.schmitt import KEY_DOWN, KEY_UP
from minecraft_cv.tracking.face_tracker import FaceResult

logger = logging.getLogger(__name__)

# MediaPipe FaceMesh landmark indices for the outer eye corners. The image-plane line
# between them gives a stable head-roll (ear-to-shoulder tilt) signal.
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# State machine for each face gesture
class FaceGestureDetector:
    """Schmitt-trigger state machine for a single face blendshape gesture."""

    def __init__(
        self,
        name: str,
        blendshape_name: str,
        t_engage: float,
        t_release: float,
        engage_frames: int = 3,
        release_frames: int = 2,
    ) -> None:
        self.name = name
        self.blendshape_name = blendshape_name
        self.t_engage = t_engage
        self.t_release = t_release
        self.engage_frames = engage_frames
        self.release_frames = release_frames

        self._is_active = False
        self._consecutive_above = 0
        self._consecutive_below = 0

    def update(self, result: FaceResult) -> list[GestureEvent]:
        """Update detector state with new blendshape scores."""
        score = result.blendshapes.get(self.blendshape_name, 0.0)
        events: list[GestureEvent] = []

        if self._is_active:
            if score <= self.t_release:
                self._consecutive_below += 1
                if self._consecutive_below >= self.release_frames:
                    self._is_active = False
                    events.append(GestureEvent(self.name, KEY_UP, "face"))
            else:
                self._consecutive_below = 0
        else:
            if score >= self.t_engage:
                self._consecutive_above += 1
                if self._consecutive_above >= self.engage_frames:
                    self._is_active = True
                    events.append(GestureEvent(self.name, KEY_DOWN, "face"))
            else:
                self._consecutive_above = 0

        return events


def _head_roll_deg(result: FaceResult) -> float | None:
    """Head-roll angle in degrees from the eye-corner line, or None if no face landmarks.

    Uses the 478-landmark FaceMesh array carried on ``result.landmarks`` (x/y normalized to
    ``[0, 1]`` in frame space, y growing downward). Roll is ``atan2(dy, dx)`` of the vector
    from the left to the right outer eye corner: 0 = upright, positive = the eye line rotates
    so the right-eye corner drops below the left. The physical tilt direction that yields a
    positive angle depends on whether the camera feed is mirrored; the left/right gesture names
    are configurable so the mapping can be swapped without code changes.
    """
    landmarks = result.landmarks
    if landmarks is None or len(landmarks) <= RIGHT_EYE_OUTER:
        return None
    left = landmarks[LEFT_EYE_OUTER]
    right = landmarks[RIGHT_EYE_OUTER]
    dx = float(right[0] - left[0])
    dy = float(right[1] - left[1])
    return math.degrees(math.atan2(dy, dx))


class HeadRollDetector:
    """Sign-gated Schmitt trigger over head-roll angle, emitting left/right scroll gestures.

    Two mutually-exclusive directions share one angle signal: rolling past ``+engage_deg``
    fires ``left_gesture``; rolling past ``-engage_deg`` fires ``right_gesture``. Each releases
    when the angle returns inside its ``release_deg`` band. Frame-count debounce mirrors
    :class:`FaceGestureDetector`. Emitted KEY_DOWN/KEY_UP events drive the pipeline's existing
    scroll-repeat path, so a held tilt scrolls continuously at ``scroll_repeat_rate_hz``.
    """

    def __init__(
        self,
        left_gesture: str,
        right_gesture: str,
        engage_deg: float,
        release_deg: float,
        engage_frames: int = 2,
        release_frames: int = 2,
    ) -> None:
        """Initialize with the left/right gesture ids and degree thresholds."""
        self.left_gesture = left_gesture
        self.right_gesture = right_gesture
        self.engage_deg = engage_deg
        self.release_deg = release_deg
        self.engage_frames = engage_frames
        self.release_frames = release_frames

        # "active" is None, "left", or "right" — only one direction can hold at a time.
        self._active: Literal["left", "right"] | None = None
        self._consecutive_above = 0
        self._consecutive_below = 0

    @property
    def names(self) -> tuple[str, str]:
        """The (left, right) gesture ids this detector can emit."""
        return (self.left_gesture, self.right_gesture)

    def _name(self, direction: Literal["left", "right"]) -> str:
        return self.left_gesture if direction == "left" else self.right_gesture

    def update(self, result: FaceResult) -> list[GestureEvent]:
        """Update with the latest face result and emit scroll-gesture transitions."""
        roll = _head_roll_deg(result)
        events: list[GestureEvent] = []

        if self._active is not None:
            # Release when the angle falls back inside the release band for this direction.
            inside = roll is None or abs(roll) <= self.release_deg
            wrong_side = (
                roll is not None
                and (
                    (self._active == "left" and roll < 0)
                    or (self._active == "right" and roll > 0)
                )
            )
            if inside or wrong_side:
                self._consecutive_below += 1
                if self._consecutive_below >= self.release_frames:
                    events.append(GestureEvent(self._name(self._active), KEY_UP, "face"))
                    self._active = None
                    self._consecutive_below = 0
                    self._consecutive_above = 0
            else:
                self._consecutive_below = 0
            return events

        # Inactive: look for a sustained roll past the engage threshold.
        if roll is None or abs(roll) < self.engage_deg:
            self._consecutive_above = 0
            return events
        direction: Literal["left", "right"] = "left" if roll > 0 else "right"
        self._consecutive_above += 1
        if self._consecutive_above >= self.engage_frames:
            events.append(GestureEvent(self._name(direction), KEY_DOWN, "face"))
            self._active = direction
            self._consecutive_above = 0
            self._consecutive_below = 0
        return events

    def reset(self) -> list[GestureEvent]:
        """Force-release the held direction (fail-safe on face dropout)."""
        events: list[GestureEvent] = []
        if self._active is not None:
            events.append(GestureEvent(self._name(self._active), KEY_UP, "face"))
            self._active = None
        self._consecutive_above = 0
        self._consecutive_below = 0
        return events


def _head_pitch_ratio(result: FaceResult) -> float | None:
    """Ratio of (nose-to-chin) / (nasion-to-nose) 2D distance.
    
    Drops significantly when the user nods their head down.
    """
    landmarks = result.landmarks
    if landmarks is None or len(landmarks) <= 168:
        return None
    nose_y = landmarks[1][1]
    chin_y = landmarks[152][1]
    nasion_y = landmarks[168][1]

    top_dist = nose_y - nasion_y
    bottom_dist = chin_y - nose_y
    if top_dist <= 0:
        return None
    return bottom_dist / top_dist


class HeadPitchDetector:
    """Schmitt trigger over 2D pitch ratio, emitting a downward-nod gesture.
    
    The ratio drops when looking down. Engages when ratio drops below `engage_ratio`.
    Releases when ratio rises above `release_ratio`.
    """
    def __init__(
        self,
        gesture: str,
        engage_ratio: float,
        release_ratio: float,
        engage_frames: int = 3,
        release_frames: int = 2,
    ) -> None:
        self.gesture = gesture
        self.engage_ratio = engage_ratio
        self.release_ratio = release_ratio
        self.engage_frames = engage_frames
        self.release_frames = release_frames

        self._is_active = False
        self._consecutive_above = 0
        self._consecutive_below = 0
        self._baseline_ratio: float | None = None

    @property
    def name(self) -> str:
        return self.gesture

    def update(self, result: FaceResult) -> list[GestureEvent]:
        raw_ratio = _head_pitch_ratio(result)
        events: list[GestureEvent] = []

        if raw_ratio is None:
            return events

        if self._baseline_ratio is None:
            self._baseline_ratio = raw_ratio
        elif not self._is_active:
            if raw_ratio > self._baseline_ratio:
                self._baseline_ratio = self._baseline_ratio * 0.5 + raw_ratio * 0.5
            else:
                self._baseline_ratio = self._baseline_ratio * 0.99 + raw_ratio * 0.01

        ratio = raw_ratio / self._baseline_ratio

        if self._is_active:
            if ratio is None or ratio >= self.release_ratio:
                self._consecutive_above += 1
                if self._consecutive_above >= self.release_frames:
                    self._is_active = False
                    events.append(GestureEvent(self.gesture, KEY_UP, "face"))
            else:
                self._consecutive_above = 0
        else:
            if ratio is not None and ratio <= self.engage_ratio:
                self._consecutive_below += 1
                if self._consecutive_below >= self.engage_frames:
                    self._is_active = True
                    events.append(GestureEvent(self.gesture, KEY_DOWN, "face"))
            else:
                self._consecutive_below = 0

        return events

    def reset(self) -> list[GestureEvent]:
        events = []
        if self._is_active:
            events.append(GestureEvent(self.gesture, KEY_UP, "face"))
            self._is_active = False
        self._consecutive_above = 0
        self._consecutive_below = 0
        return events


class FaceGestureStateMachine:
    """Manages all face gestures (blendshape gestures + head-roll scroll)."""

    def __init__(
        self,
        settings: dict[str, Any],
        head_roll: Any | None = None,
        head_pitch: Any | None = None,
    ) -> None:
        """Initialize from face gesture settings and optional head-roll/pitch settings."""
        self._detectors: list[FaceGestureDetector] = []
        for name, config in settings.items():
            detector = FaceGestureDetector(
                name=name,
                blendshape_name=config.blendshape,
                t_engage=config.t_engage,
                t_release=config.t_release,
                engage_frames=config.engage_frames,
                release_frames=config.release_frames,
            )
            self._detectors.append(detector)

        self._head_roll: HeadRollDetector | None = None
        if head_roll is not None and getattr(head_roll, "enabled", True):
            self._head_roll = HeadRollDetector(
                left_gesture=head_roll.left_gesture,
                right_gesture=head_roll.right_gesture,
                engage_deg=head_roll.engage_deg,
                release_deg=head_roll.release_deg,
                engage_frames=head_roll.engage_frames,
                release_frames=head_roll.release_frames,
            )

        self._head_pitch: HeadPitchDetector | None = None
        if head_pitch is not None and getattr(head_pitch, "enabled", True):
            self._head_pitch = HeadPitchDetector(
                gesture=head_pitch.gesture,
                engage_ratio=head_pitch.engage_ratio,
                release_ratio=head_pitch.release_ratio,
                engage_frames=head_pitch.engage_frames,
                release_frames=head_pitch.release_frames,
            )

        self._last_result = FaceResult()

    def update(self, result: FaceResult) -> list[GestureEvent]:
        """Process a face result and emit events."""
        self._last_result = result
        events = []
        for detector in self._detectors:
            events.extend(detector.update(result))
        if self._head_roll is not None:
            events.extend(self._head_roll.update(result))
        if self._head_pitch is not None:
            events.extend(self._head_pitch.update(result))
        return events

    def active_gestures(self) -> frozenset[str]:
        """The set of currently-held face gesture ids (blendshape gestures + head roll)."""
        names = {d.name for d in self._detectors if d._is_active}
        if self._head_roll is not None and self._head_roll._active is not None:
            names.add(self._head_roll._name(self._head_roll._active))
        if self._head_pitch is not None and self._head_pitch._is_active:
            names.add(self._head_pitch.name)
        return frozenset(names)

    def status(self) -> Literal["tracking", "absent"]:
        """Return tracking status based on latest result."""
        if self._last_result.blendshapes or self._last_result.landmarks is not None:
            return "tracking"
        return "absent"

    def reset(self) -> list[GestureEvent]:
        """Force release all active gestures."""
        events = []
        for detector in self._detectors:
            if detector._is_active:
                detector._is_active = False
                detector._consecutive_above = 0
                detector._consecutive_below = 0
                events.append(GestureEvent(detector.name, KEY_UP, "face"))
        if self._head_roll is not None:
            events.extend(self._head_roll.reset())
        if self._head_pitch is not None:
            events.extend(self._head_pitch.reset())
        return events
