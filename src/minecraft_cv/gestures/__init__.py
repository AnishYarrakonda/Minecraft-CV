"""Discrete gesture layer: detector registry, pinch, curled-finger, and legacy helpers.

Includes Schmitt triggers, per-hand resolvers, finger-extension detection, and fail-safe
tracking-loss guard.
"""

from __future__ import annotations

from minecraft_cv.gestures.extension import (
    ExtensionStateMachine,
    ExtensionThresholdSpec,
)
from minecraft_cv.gestures.finger_state import FingerState, finger_extensions
from minecraft_cv.gestures.pinch import (
    GestureEvent,
    GestureSpec,
    PinchStateMachine,
    ThresholdSpec,
    normalized_distances,
)
from minecraft_cv.gestures.registry import (
    GestureDetectorSpec,
    GestureStateMachine,
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
    "ExtensionStateMachine",
    "ExtensionThresholdSpec",
    "FingerState",
    "GestureEvent",
    "GestureDetectorSpec",
    "GestureSpec",
    "GestureStateMachine",
    "PinchState",
    "PinchStateMachine",
    "SchmittTrigger",
    "ThresholdSpec",
    "TrackingLossGuard",
    "finger_extensions",
    "normalized_distances",
]
