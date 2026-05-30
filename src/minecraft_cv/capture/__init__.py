"""Capture layer: frame sources (camera/clip) + threaded newest-frame-wins buffer."""

from __future__ import annotations

from minecraft_cv.capture.buffer import FrameBuffer
from minecraft_cv.capture.source import (
    AVFoundationSource,
    ClipSource,
    FrameSource,
    enumerate_devices,
)

__all__ = [
    "AVFoundationSource",
    "ClipSource",
    "FrameBuffer",
    "FrameSource",
    "enumerate_devices",
]
