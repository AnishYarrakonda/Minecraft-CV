"""Mocked backend tests for :class:`MacInputEmitter` (Task 5).

Hard invariant #2 forbids moving the real mouse or pressing real keys in tests. We uphold it
*while still testing the real backend code* by injecting fake ``pynput`` / ``Quartz`` /
``ApplicationServices`` modules into ``sys.modules`` before the emitter imports them. Every
"emitted" event lands in an in-memory recorder; nothing reaches the OS, and the genuine
pynput/Quartz libraries are never imported.

This is the one place ``MacInputEmitter`` is instantiated — and only against fakes.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from minecraft_cv.config import Settings
from minecraft_cv.gestures.pinch import PinchStateMachine

# --- fake backend modules ------------------------------------------------------------------


class _Recorder:
    """Captures every fake OS call so tests can assert on them."""

    def __init__(self) -> None:
        self.kbd: list[tuple[str, Any]] = []
        self.mouse: list[tuple[str, Any]] = []
        self.posted: list[dict[str, Any]] = []


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, trusted: bool = True) -> _Recorder:
    rec = _Recorder()

    # pynput.keyboard
    class FakeKeyboard:
        def press(self, key: Any) -> None:
            rec.kbd.append(("press", key))

        def release(self, key: Any) -> None:
            rec.kbd.append(("release", key))

    class FakeKeyCode:
        def __init__(self, char: str) -> None:
            self.char = char

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeKeyCode) and other.char == self.char

        def __hash__(self) -> int:
            return hash(("kc", self.char))

        @staticmethod
        def from_char(char: str) -> FakeKeyCode:
            return FakeKeyCode(char)

    keyboard_mod = ModuleType("pynput.keyboard")
    keyboard_mod.Controller = lambda: FakeKeyboard()  # type: ignore[attr-defined]
    keyboard_mod.Key = SimpleNamespace(  # type: ignore[attr-defined]
        space="K.space",
        shift="K.shift",
        ctrl="K.ctrl",
        alt="K.alt",
        cmd="K.cmd",
        enter="K.enter",
        tab="K.tab",
        esc="K.esc",
    )
    keyboard_mod.KeyCode = FakeKeyCode  # type: ignore[attr-defined]

    # pynput.mouse
    class FakeMouse:
        def press(self, button: Any) -> None:
            rec.mouse.append(("press", button))

        def release(self, button: Any) -> None:
            rec.mouse.append(("release", button))

        def scroll(self, dx: int, dy: int) -> None:
            rec.mouse.append(("scroll", (dx, dy)))

    mouse_mod = ModuleType("pynput.mouse")
    mouse_mod.Controller = lambda: FakeMouse()  # type: ignore[attr-defined]
    mouse_mod.Button = SimpleNamespace(left="B.left", right="B.right", middle="B.middle")  # type: ignore[attr-defined]

    pynput_mod = ModuleType("pynput")
    pynput_mod.keyboard = keyboard_mod  # type: ignore[attr-defined]
    pynput_mod.mouse = mouse_mod  # type: ignore[attr-defined]

    # Quartz
    quartz = ModuleType("Quartz")

    def _create_event(_a: Any, etype: Any, pos: Any, _b: Any) -> dict[str, Any]:
        return {"type": etype, "pos": pos, "fields": {}}

    def _set_field(event: dict[str, Any], field: Any, value: int) -> None:
        event["fields"][field] = value

    def _post(_tap: Any, event: dict[str, Any]) -> None:
        rec.posted.append(event)

    quartz.CGEventCreateMouseEvent = _create_event  # type: ignore[attr-defined]
    quartz.CGEventSetIntegerValueField = _set_field  # type: ignore[attr-defined]
    quartz.CGEventPost = _post  # type: ignore[attr-defined]
    quartz.CGMainDisplayID = lambda: 1  # type: ignore[attr-defined]
    quartz.CGDisplayPixelsWide = lambda _d: 1000  # type: ignore[attr-defined]
    quartz.CGDisplayPixelsHigh = lambda _d: 800  # type: ignore[attr-defined]
    quartz.kCGEventMouseMoved = "mouseMoved"  # type: ignore[attr-defined]
    quartz.kCGMouseButtonLeft = "btnLeft"  # type: ignore[attr-defined]
    quartz.kCGMouseEventDeltaX = "dX"  # type: ignore[attr-defined]
    quartz.kCGMouseEventDeltaY = "dY"  # type: ignore[attr-defined]
    quartz.kCGHIDEventTap = "hid"  # type: ignore[attr-defined]

    # ApplicationServices (permission check)
    appsvc = ModuleType("ApplicationServices")
    appsvc.AXIsProcessTrusted = lambda: trusted  # type: ignore[attr-defined]

    for name, mod in (
        ("pynput", pynput_mod),
        ("pynput.keyboard", keyboard_mod),
        ("pynput.mouse", mouse_mod),
        ("Quartz", quartz),
        ("ApplicationServices", appsvc),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return rec


@pytest.fixture
def emitter_and_recorder(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    rec = _install_fakes(monkeypatch)
    from minecraft_cv.input.mac_emitter import MacInputEmitter

    em = MacInputEmitter(mouse_delta_scale=10.0, key_repeat_guard_ms=10_000.0)
    return em, rec


# --- key / button events -------------------------------------------------------------------


def test_key_down_presses_special_key(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder
    em.key_down("space")
    assert ("press", "K.space") in rec.kbd
    assert em.held_keys == frozenset({"space"})


def test_key_down_is_deduplicated(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder
    em.key_down("shift")
    em.key_down("shift")
    assert [c for c in rec.kbd if c[0] == "press"] == [("press", "K.shift")]


def test_mouse_button_routed_to_mouse_controller(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder
    em.key_down("mouse_left")
    em.key_up("mouse_left")
    assert ("press", "B.left") in rec.mouse
    assert ("release", "B.left") in rec.mouse


def test_key_tap_presses_then_releases(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder
    em.key_tap("e")
    kinds = [c[0] for c in rec.kbd]
    assert kinds == ["press", "release"]
    # key_tap is not tracked as held.
    assert em.held_keys == frozenset()


def test_release_all_flushes_keyboard_and_mouse(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder
    em.key_down("space")
    em.key_down("mouse_left")
    em.release_all()
    assert ("release", "K.space") in rec.kbd
    assert ("release", "B.left") in rec.mouse
    assert em.held_keys == frozenset()


# --- mouse motion --------------------------------------------------------------------------


def test_relative_move_accumulates_subpixel_then_posts(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder  # mouse_delta_scale=10
    em.mouse_move(0.05, 0.0)  # 0.5 px -> below 1, carried, nothing posted
    assert rec.posted == []
    em.mouse_move(0.05, 0.0)  # +0.5 px -> 1.0 px -> posts deltaX == 1
    assert len(rec.posted) == 1
    assert rec.posted[0]["fields"]["dX"] == 1


def test_absolute_move_scales_to_display_pixels(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder  # fake display = 1000 x 800
    em.mouse_move_abs(0.5, 0.25)
    assert len(rec.posted) == 1
    assert rec.posted[0]["pos"] == (500.0, 200.0)


def test_absolute_move_clamps_out_of_range(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder
    em.mouse_move_abs(5.0, -1.0)
    assert rec.posted[0]["pos"] == (1000.0, 0.0)


def test_scroll_is_rate_limited(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder  # guard = 10 s -> second same-direction scroll suppressed
    em.scroll(1)
    em.scroll(1)
    scrolls = [c for c in rec.mouse if c[0] == "scroll"]
    assert len(scrolls) == 1


# --- permissions ---------------------------------------------------------------------------
