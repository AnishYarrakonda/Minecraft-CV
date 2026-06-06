"""Game-faithful regression test: a held modifier (sneak's Shift) must NOT corrupt WASD.

The mocked tests in ``test_mac_emitter.py`` replace pynput's ``Controller`` with a fake that
records the logical key and stops there. That fake bypasses pynput's real
``Controller._resolve`` / ``KeyCode._event`` path — the exact code that misbehaves *in
Minecraft* while Shift (sneak) is held. So those tests stay green even when the game is
unplayable.

This module instead drives the **real** ``MacInputEmitter`` + real pynput Controller, and only
intercepts the final ``Quartz.CGEventPost`` so nothing ever reaches the OS HID tap (hard
invariant #2: tests never emit real input). It asserts the property that actually matters
in-game: the physical key code posted for each of W/A/S/D is independent of whether sneak's
Shift is currently held.

macOS-only: skipped where the real pynput/Quartz darwin backend is unavailable.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="exercises the real pynput/Quartz macOS backend"
)


@pytest.fixture
def live_emitter(monkeypatch):  # type: ignore[no-untyped-def]
    """A real ``MacInputEmitter`` whose OS key posts are captured instead of emitted.

    Yields ``(emitter, posted_keycodes)`` where ``posted_keycodes`` is the list of physical
    virtual-key codes pynput would have posted (most-recent last).
    """
    pytest.importorskip("Quartz")
    pytest.importorskip("pynput.keyboard")
    import pynput.keyboard._darwin as darwin_backend
    from Quartz import CGEventGetIntegerValueField, kCGKeyboardEventKeycode

    from minecraft_cv.input.mac_emitter import MacInputEmitter

    posted: list[int] = []

    def fake_post(_tap, event):  # type: ignore[no-untyped-def]
        # Capture the key code; never forward to the real HID tap.
        posted.append(int(CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)))

    monkeypatch.setattr(darwin_backend, "CGEventPost", fake_post)
    # Orthogonal side effects we don't want during a test run.
    monkeypatch.setattr(MacInputEmitter, "_check_permissions", lambda self: None)
    monkeypatch.setattr(MacInputEmitter, "_disable_app_nap", lambda self: None)

    return MacInputEmitter(), posted


def test_held_sneak_shift_does_not_remap_wasd_keycodes(live_emitter) -> None:  # type: ignore[no-untyped-def]
    """Holding Shift (sneak) must not change which physical key each WASD press emits."""
    emitter, posted = live_emitter

    def keycode_for(key: str) -> int:
        posted.clear()
        emitter.key_down(key)
        code = posted[-1]
        emitter.key_up(key)
        return code

    # Baseline: keycode each movement key posts with no modifier held.
    baseline = {k: keycode_for(k) for k in ("w", "a", "s", "d")}

    # Now hold sneak's Shift and press the same keys.
    emitter.key_down("shift")
    with_shift = {k: keycode_for(k) for k in ("w", "a", "s", "d")}
    emitter.key_up("shift")

    assert with_shift == baseline, (
        f"Sneak's held Shift remapped WASD key codes: {baseline} -> {with_shift}. "
        "Each movement key must post the same physical key code regardless of held modifiers."
    )
    # Guard against the specific failure: all four collapsing onto one key code (e.g. A).
    assert len(set(with_shift.values())) == 4, (
        f"WASD collapsed onto fewer than four distinct keys while sneaking: {with_shift}"
    )
