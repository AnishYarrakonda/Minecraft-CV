"""Polished PySide6 desktop front-end for minecraft_cv.

The camera feed is "the painting" (clean, with a subtle glowing hand skeleton); the
surrounding window chrome is "the frame" (live keymap, tracking health, controls).

Submodules :mod:`keymap` and :mod:`skeleton` are intentionally **Qt-free** so the pure HUD
logic can be unit-tested without installing PySide6. Everything that touches Qt
(``app``, ``camera_view``, ``panels``, ``widgets``, ``worker``, ``theme``) imports PySide6
lazily, so ``import minecraft_cv.ui`` itself stays light.
"""

from __future__ import annotations
