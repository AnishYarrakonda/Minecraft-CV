"""Face gesture detector using blendshape scores and Schmitt triggers."""

from __future__ import annotations

import logging
from typing import Any, Literal

from minecraft_cv.gestures.registry import GestureEvent
from minecraft_cv.gestures.schmitt import KEY_DOWN, KEY_UP
from minecraft_cv.tracking.face_tracker import FaceResult

logger = logging.getLogger(__name__)

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


class FaceGestureStateMachine:
    """Manages all face gestures."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Initialize from face gesture settings."""
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
        
        self._last_result = FaceResult()

    def update(self, result: FaceResult) -> list[GestureEvent]:
        """Process a face result and emit events."""
        self._last_result = result
        events = []
        for detector in self._detectors:
            events.extend(detector.update(result))
        return events
    
    def status(self) -> Literal["tracking", "absent"]:
        """Return tracking status based on latest result."""
        if self._last_result.blendshapes:
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
        return events
