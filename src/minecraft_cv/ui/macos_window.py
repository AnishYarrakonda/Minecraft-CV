"""Native macOS window-level helpers for Qt windows that pin in front of games.

Qt's ``WindowStaysOnTopHint`` only raises a window to ``NSFloatingWindowLevel`` and,
for ``Qt.Tool`` windows, AppKit hides the window when its app is deactivated. Neither
keeps a HUD reliably in front of a focused or fullscreen game. This module reaches the
underlying ``NSWindow`` (via pyobjc) to pin it above other apps, including separate
fullscreen Spaces (:func:`keep_window_in_front`), and to undo that pinning
(:func:`reset_window_level`). All pyobjc imports are lazy and guarded so the module is a
no-op on non-macOS platforms and when the native handles are unavailable.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


def keep_window_in_front(widget: QWidget) -> None:
    """Pin a Qt window above other apps on macOS, including fullscreen Spaces.

    Raises the native window level above normal/floating app windows, opts it into every
    Space plus fullscreen-auxiliary, and disables hide-on-deactivate so the window stays
    visible and in front when another app (e.g. Minecraft) is focused.

    The widget's native window must already exist, so call this from ``showEvent`` or after
    ``show()`` — ``widget.winId()`` is only valid once the window is realized.

    No-op on non-macOS platforms or if pyobjc / the native window handle is unavailable; the
    window then falls back to its existing Qt ``WindowStaysOnTopHint`` behavior.

    Args:
        widget: A shown top-level Qt widget whose native ``NSWindow`` should be pinned.
    """
    if sys.platform != "darwin":
        return
    try:
        import objc
        from AppKit import (  # type: ignore[import-untyped]
            NSStatusWindowLevel,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowCollectionBehaviorStationary,
        )

        view = objc.objc_object(c_void_p=int(widget.winId()))  # NSView*
        ns_window = view.window()
        if ns_window is None:
            return
        # NSStatusWindowLevel (25) floats above normal/floating app windows (windowed and
        # typical borderless games) while staying below the menu bar and system alerts.
        ns_window.setLevel_(NSStatusWindowLevel)
        ns_window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )
        ns_window.setHidesOnDeactivate_(False)
    except Exception:  # pragma: no cover - defensive native bridge
        pass


def reset_window_level(widget: QWidget) -> None:
    """Undo :func:`keep_window_in_front`: return the window to its normal level and behavior.

    Restores the default ``NSWindowLevel`` and collection behavior so an unpinned window stops
    floating above other apps and fullscreen Spaces. No-op on non-macOS platforms or if pyobjc /
    the native window handle is unavailable.

    Args:
        widget: A shown top-level Qt widget whose native ``NSWindow`` should be reset.
    """
    if sys.platform != "darwin":
        return
    try:
        import objc
        from AppKit import (  # type: ignore[import-untyped]
            NSNormalWindowLevel,
            NSWindowCollectionBehaviorDefault,
        )

        view = objc.objc_object(c_void_p=int(widget.winId()))  # NSView*
        ns_window = view.window()
        if ns_window is None:
            return
        ns_window.setLevel_(NSNormalWindowLevel)
        ns_window.setCollectionBehavior_(NSWindowCollectionBehaviorDefault)
        ns_window.setHidesOnDeactivate_(False)
    except Exception:  # pragma: no cover - defensive native bridge
        pass


__all__ = ["keep_window_in_front", "reset_window_level"]
