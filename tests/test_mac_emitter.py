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
        space="K.space", shift="K.shift", ctrl="K.ctrl", alt="K.alt",
        cmd="K.cmd", enter="K.enter", tab="K.tab", esc="K.esc",
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

    def _create_event(_a: Any, etype: Any, pos: Any, button: Any) -> dict[str, Any]:
        return {"type": etype, "pos": pos, "button": button, "fields": {}}

    def _create_generic_event(_a: Any) -> dict[str, Any]:
        return {"type": "generic", "pos": (123.0, 456.0), "fields": {}}

    def _get_location(_event: dict[str, Any]) -> tuple[float, float]:
        return (123.0, 456.0)

    def _set_field(event: dict[str, Any], field: Any, value: int) -> None:
        event["fields"][field] = value

    def _post(_tap: Any, event: dict[str, Any]) -> None:
        rec.posted.append(event)

    quartz.CGEventCreateMouseEvent = _create_event  # type: ignore[attr-defined]
    quartz.CGEventCreate = _create_generic_event  # type: ignore[attr-defined]
    quartz.CGEventGetLocation = _get_location  # type: ignore[attr-defined]
    quartz.CGEventSetIntegerValueField = _set_field  # type: ignore[attr-defined]
    quartz.CGEventPost = _post  # type: ignore[attr-defined]
    quartz.CGMainDisplayID = lambda: 1  # type: ignore[attr-defined]
    quartz.CGDisplayPixelsWide = lambda _d: 1000  # type: ignore[attr-defined]
    quartz.CGDisplayPixelsHigh = lambda _d: 800  # type: ignore[attr-defined]
    quartz.kCGEventMouseMoved = "mouseMoved"  # type: ignore[attr-defined]
    quartz.kCGMouseButtonLeft = "btnLeft"  # type: ignore[attr-defined]
    quartz.kCGMouseButtonRight = "btnRight"  # type: ignore[attr-defined]
    quartz.kCGMouseButtonCenter = "btnCenter"  # type: ignore[attr-defined]
    quartz.kCGEventLeftMouseDown = "leftDown"  # type: ignore[attr-defined]
    quartz.kCGEventLeftMouseUp = "leftUp"  # type: ignore[attr-defined]
    quartz.kCGEventRightMouseDown = "rightDown"  # type: ignore[attr-defined]
    quartz.kCGEventRightMouseUp = "rightUp"  # type: ignore[attr-defined]
    quartz.kCGEventOtherMouseDown = "otherDown"  # type: ignore[attr-defined]
    quartz.kCGEventOtherMouseUp = "otherUp"  # type: ignore[attr-defined]
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

    em = MacInputEmitter(
        mouse_delta_scale=10.0,
        key_repeat_guard_ms=10_000.0,
    )
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


def test_mouse_button_routed_to_quartz_at_current_position(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder
    em.key_down("mouse_left")
    em.key_up("mouse_left")
    assert [event["type"] for event in rec.posted] == ["leftDown", "leftUp"]
    assert [event["pos"] for event in rec.posted] == [(123.0, 456.0), (123.0, 456.0)]


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
    assert any(event["type"] == "leftUp" for event in rec.posted)
    assert em.held_keys == frozenset()


# --- mouse motion --------------------------------------------------------------------------


def test_relative_move_accumulates_subpixel_then_posts(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder  # mouse_delta_scale=10
    em.mouse_move(0.05, 0.0)  # 0.5 px -> below 1, carried, nothing posted
    assert rec.posted == []
    em.mouse_move(0.05, 0.0)  # +0.5 px -> 1.0 px -> posts deltaX == 1
    assert len(rec.posted) == 1
    assert rec.posted[0]["pos"] == (124.0, 456.0)
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


def test_missing_accessibility_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch, trusted=False)
    from minecraft_cv.input.mac_emitter import MacInputEmitter

    with pytest.raises(PermissionError):
        MacInputEmitter()


# --- integration: gesture state machine -> mocked emitter ----------------------------------


def test_pinch_sequence_drives_clean_down_up_no_stuck_key(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    """A Schmitt-gated pinch sequence produces exactly one press then one release."""
    em, rec = emitter_and_recorder
    sm = PinchStateMachine("right", {"attack": Settings().gestures.right_hand["attack"]})

    def drive(distance: float) -> None:
        import numpy as np

        # Build a single-finger landmark set with a known normalized index distance.
        lm = np.zeros((21, 3), dtype=np.float32)
        lm[9] = (0.0, 0.2, 0.0)  # hand scale = 0.2
        lm[4] = (0.5, 0.5, 0.0)  # thumb
        lm[8] = lm[4] + np.array([distance * 0.2, 0.0, 0.0], dtype=np.float32)  # index tip
        for name in ("middle", "ring", "pinky"):
            idx = {"middle": 12, "ring": 16, "pinky": 20}[name]
            lm[idx] = lm[4] + np.array([1.0 * 0.2, 0.0, 0.0], dtype=np.float32)  # far -> released
        for event in sm.update(lm):
            if event.action == "KEY_DOWN":
                em.key_down("mouse_left")
            else:
                em.key_up("mouse_left")

    drive(0.5)   # released (above engage)
    drive(0.20)  # engage -> press
    drive(0.20)  # engage -> press (debounce)
    drive(0.25)  # still in hysteresis band -> no change
    drive(0.5)   # release
    drive(0.5)   # release (debounce)
    presses = [c for c in rec.posted if c["type"] == "leftDown"]
    releases = [c for c in rec.posted if c["type"] == "leftUp"]
    assert len(presses) == 1
    assert len(releases) == 1
    assert em.held_keys == frozenset()


def test_tracking_loss_reset_flushes_held_button(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    """Holding a pinch then losing tracking must release the button (no stuck input)."""
    import numpy as np

    em, rec = emitter_and_recorder
    sm = PinchStateMachine("right", {"attack": Settings().gestures.right_hand["attack"]})
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[9] = (0.0, 0.2, 0.0)
    lm[4] = (0.5, 0.5, 0.0)
    lm[8] = lm[4] + np.array([0.20 * 0.2, 0.0, 0.0], dtype=np.float32)
    for idx in (12, 16, 20):  # middle, ring, pinky -> far/released
        lm[idx] = lm[4] + np.array([0.2, 0.0, 0.0], dtype=np.float32)
    for _ in sm.update(lm):  # engage
        em.key_down("mouse_left")
    for _ in sm.update(lm):  # engage (debounce)
        em.key_down("mouse_left")
    assert em.held_keys == frozenset({"mouse_left"})
    # Tracking lost: the machine's reset emits KEY_UP, which the emitter must honor.
    for _ in sm.reset():
        em.key_up("mouse_left")
    assert any(event["type"] == "leftUp" for event in rec.posted)
    assert em.held_keys == frozenset()


def test_context_manager_releases_on_exit(emitter_and_recorder) -> None:  # type: ignore[no-untyped-def]
    em, rec = emitter_and_recorder
    em.key_down("space")
    em.__exit__(None, None, None)
    assert ("release", "K.space") in rec.kbd


# --- config invariant at the boundary the emitter serves -----------------------------------


def test_pinch_config_upholds_release_gt_engage() -> None:
    """The emitter only ever sees clean transitions because T_release > T_engage holds."""
    for name, spec in Settings().gestures.right_hand.items():
        if spec.detector == "extension_combo":
            assert spec.t_engage > spec.t_release, name
        else:
            assert spec.t_release > spec.t_engage, name
