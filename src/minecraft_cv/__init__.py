"""minecraft_cv — real-time webcam gesture -> Minecraft input controller.

Pipeline: camera capture -> MediaPipe hand pose -> (spatial joysticks + pinch-bitmask
Schmitt-trigger gestures) -> OS keyboard/mouse input. See ``.claude/CLAUDE.md`` and the
rules in ``.claude/rules/`` for the full design contract.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
