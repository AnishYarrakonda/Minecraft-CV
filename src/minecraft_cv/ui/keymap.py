"""Keymap model for the HUD: turn the config into displayable gesture -> key rows.

Qt-free and pure so it can be unit-tested without PySide6. The desktop sidebar renders one
:class:`KeyRow` per bound gesture and lights its indicator from the live
``StepResult.left_gestures`` / ``right_gestures`` sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minecraft_cv.config import Settings

# Human-readable gesture names (fallback is a title-cased gesture id).
_DISPLAY_NAMES: dict[str, str] = {
    "jump": "Jump",
    "inventory": "Inventory",
    "throw_item": "Throw Item",
    "sneak": "Sneak",
    "sprint": "Sprint",
    "attack": "Attack",
    "use": "Use",
    "hotbar_next": "Hotbar Next",
    "hotbar_prev": "Hotbar Prev",
}

# Pretty key-cap labels for raw binding names (fallback handles single letters / titles).
_KEY_LABELS: dict[str, str] = {
    "space": "Space",
    "shift": "Shift",
    "ctrl": "Ctrl",
    "alt": "Alt",
    "tab": "Tab",
    "esc": "Esc",
    "enter": "Enter",
    "mouse_left": "LMB",
    "mouse_right": "RMB",
    "mouse_middle": "MMB",
    "scroll_up": "Scroll ↑",
    "scroll_down": "Scroll ↓",
}

# Pretty finger descriptors for the row subtitle.
_FINGER_LABELS: dict[str, str] = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "pinky": "Pinky",
}


@dataclass(frozen=True)
class KeyRow:
    """One displayable gesture -> key binding.

    Attributes:
        gesture: Logical gesture id (matches ``StepResult.left_gestures`` / ``right_gestures``).
        name: Human-readable gesture name, e.g. ``"Jump"``.
        key: Pretty key-cap label, e.g. ``"Space"`` / ``"LMB"`` / ``"Scroll ↑"``.
        hand: ``"left"`` or ``"right"``.
        finger: Pretty finger descriptor for the trigger (e.g. ``"Index"``), or ``""``.
    """

    gesture: str
    name: str
    key: str
    hand: str
    finger: str = ""


def display_name(gesture: str) -> str:
    """Return a human-readable name for a gesture id."""
    return _DISPLAY_NAMES.get(gesture, gesture.replace("_", " ").title())


def key_label(binding: str) -> str:
    """Return a pretty key-cap label for a raw binding name.

    Args:
        binding: Raw binding (``"space"``, ``"mouse_left"``, ``"scroll_up"``, ``"e"`` ...).

    Returns:
        A short display label suitable for a key-cap (``"Space"``, ``"LMB"``, ``"E"`` ...).
    """
    if binding in _KEY_LABELS:
        return _KEY_LABELS[binding]
    if len(binding) == 1:
        return binding.upper()
    return binding.replace("_", " ").title()


def _finger_label(finger: str) -> str:
    return _FINGER_LABELS.get(finger, finger.title() if finger else "")


def build_keymap(settings: Settings) -> list[KeyRow]:
    """Build the ordered list of key rows to display, in config order, left hand then right.

    Only gestures that have an entry in ``settings.bindings`` are included, so non-key macros
    such as ``recenter`` are excluded.

    Args:
        settings: Loaded configuration (gestures + bindings).

    Returns:
        Left-hand, right-hand, and face rows, each in the order configured.
    """
    bindings = settings.bindings
    rows: list[KeyRow] = []
    for hand, gestures in (
        ("left", settings.gestures.left_hand),
        ("right", settings.gestures.right_hand),
        ("face", settings.gestures.face) if hasattr(settings.gestures, "face") else None,
    ):
        if hand is None:
            continue
        for gesture, spec in gestures.items():
            binding = bindings.get(gesture)
            if binding is None:
                continue
            rows.append(
                KeyRow(
                    gesture=gesture,
                    name=display_name(gesture),
                    key=key_label(binding),
                    hand=hand,
                    finger=_finger_label(getattr(spec, "finger", "")),
                )
            )
    return rows


__all__ = ["KeyRow", "build_keymap", "display_name", "key_label"]
