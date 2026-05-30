"""Discrete pinch-bitmask gesture layer: Schmitt triggers, per-hand resolver, fail-safe."""

from __future__ import annotations

from minecraft_cv.gestures.pinch import (
    GestureEvent,
    GestureSpec,
    PinchStateMachine,
    ThresholdSpec,
    normalized_distances,
)
from minecraft_cv.gestures.safety import TrackingLossGuard
from minecraft_cv.gestures.schmitt import (
    KEY_DOWN,
    KEY_UP,
    PinchState,
    SchmittTrigger,
)

__all__ = [
    "KEY_DOWN",
    "KEY_UP",
    "GestureEvent",
    "GestureSpec",
    "PinchState",
    "PinchStateMachine",
    "SchmittTrigger",
    "ThresholdSpec",
    "TrackingLossGuard",
    "normalized_distances",
]
