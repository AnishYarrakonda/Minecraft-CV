"""Pure tests for the HUD keymap model (no PySide6 needed)."""
from __future__ import annotations

from minecraft_cv.config import Settings
from minecraft_cv.ui.keymap import build_keymap, display_name, key_label


def test_build_keymap_covers_all_bound_gestures() -> None:
    rows = {(r.hand, r.gesture): r for r in build_keymap(Settings())}

    # Left-hand actions with their pretty key labels and trigger fingers.
    assert rows[("left", "jump")].key == "Space"
    assert rows[("left", "jump")].finger == "Index"
    assert rows[("left", "inventory")].key == "E"
    assert rows[("left", "throw_item")].key == "Q"
    assert rows[("left", "sneak")].key == "Shift"

    # Right-hand combat with mouse / scroll specials.
    assert rows[("right", "attack")].key == "LMB"
    assert rows[("right", "use")].key == "RMB"
    assert rows[("right", "hotbar_next")].key == "Scroll ↑"
    assert rows[("right", "hotbar_prev")].key == "Scroll ↓"


def test_build_keymap_excludes_unbound_macros() -> None:
    rows = {(r.hand, r.gesture) for r in build_keymap(Settings())}
    # ``recenter`` has no binding in settings.bindings, so it must not appear.
    assert ("left", "recenter") not in rows
    assert ("right", "recenter") not in rows


def test_build_keymap_orders_left_then_right() -> None:
    hands = [r.hand for r in build_keymap(Settings())]
    assert "left" in hands and "right" in hands
    # All left rows precede all right rows.
    assert hands == sorted(hands, key=lambda h: 0 if h == "left" else 1)
    assert hands.index("right") == hands.count("left")


def test_key_label_specials_and_fallback() -> None:
    assert key_label("space") == "Space"
    assert key_label("shift") == "Shift"
    assert key_label("mouse_left") == "LMB"
    assert key_label("mouse_right") == "RMB"
    assert key_label("scroll_up") == "Scroll ↑"
    assert key_label("scroll_down") == "Scroll ↓"
    assert key_label("e") == "E"  # single letter
    assert key_label("page_up") == "Page Up"  # fallback title-case


def test_display_name_known_and_fallback() -> None:
    assert display_name("throw_item") == "Throw Item"
    assert display_name("hotbar_next") == "Hotbar Next"
    assert display_name("some_new_gesture") == "Some New Gesture"
