"""Tracking layer: HandTracker ABC + HandResult. Backends imported lazily via create()."""

from __future__ import annotations

from minecraft_cv.tracking.tracker import HandResult, HandTracker

__all__ = ["HandResult", "HandTracker"]
