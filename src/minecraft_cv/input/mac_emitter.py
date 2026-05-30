"""Real macOS input backend: pynput keyboard/scroll + Quartz CGEvent mouse-look.

Opt-in only (``input.enabled: true`` / ``--input``). Never imported during tests — all heavy
imports (pynput, Quartz, ApplicationServices) happen inside ``__init__`` / methods so that
``import minecraft_cv.input`` stays light and camera/OS-free.

Permissions: emitting synthetic input requires **Accessibility / Input Monitoring** granted
to the terminal app running Python (System Settings -> Privacy & Security). Without it, events
are silently dropped. We best-effort detect this at startup and raise a clear error.

Mouse look uses Quartz relative deltas (``kCGMouseEventDeltaX/Y``) because Minecraft reads
true relative motion; we emit small, frequent deltas and do not correct for macOS pointer
acceleration at the injection level.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from minecraft_cv.input.emitter import InputEmitter

# Logical names that map to mouse buttons rather than keyboard keys.
_MOUSE_BUTTONS = ("mouse_left", "mouse_right", "mouse_middle")


class MacInputEmitter(InputEmitter):
    """Emits real keyboard/mouse events on macOS via pynput + Quartz CGEvent."""

    def __init__(self, mouse_delta_scale: float = 5.0, key_repeat_guard_ms: float = 50.0) -> None:
        """Construct the emitter and verify input permissions.

        Args:
            mouse_delta_scale: Multiplier from joystick units to CGEvent relative pixels.
            key_repeat_guard_ms: Minimum interval between repeated emits of the same event.

        Raises:
            RuntimeError: If pynput / Quartz are not importable (not installed).
            PermissionError: If Accessibility/Input Monitoring is detectably not granted.
        """
        super().__init__()
        self.mouse_delta_scale = float(mouse_delta_scale)
        self.key_repeat_guard_s = float(key_repeat_guard_ms) / 1000.0

        try:
            from pynput import keyboard, mouse
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "pynput is required for live input emission. Install with "
                "'pip install pynput', or run with --no-input."
            ) from exc

        try:
            import Quartz
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "pyobjc-framework-Quartz is required for mouse-look emission. Install with "
                "'pip install pyobjc-framework-Quartz', or run with --no-input."
            ) from exc

        self._keyboard: Any = keyboard.Controller()
        self._mouse: Any = mouse.Controller()
        self._kbd_mod: Any = keyboard
        self._mouse_mod: Any = mouse
        self._quartz: Any = Quartz
        self._last_emit: dict[str, float] = {}

        self._check_permissions()

    # --- permission detection ----------------------------------------------
    def _check_permissions(self) -> None:
        """Best-effort check that this process may emit synthetic input."""
        trusted = True
        try:  # pragma: no cover - macOS-only, environment-dependent
            from ApplicationServices import AXIsProcessTrusted

            trusted = bool(AXIsProcessTrusted())
        except Exception:
            # Can't determine (framework missing); proceed rather than false-alarm.
            trusted = True
        if not trusted:  # pragma: no cover - depends on system grant
            raise PermissionError(
                "Accessibility / Input Monitoring is not granted to this terminal app. "
                "Open System Settings -> Privacy & Security -> Accessibility (and Input "
                "Monitoring) and enable your terminal, then restart it. Without this, input "
                "events are silently dropped."
            )

    # --- helpers ------------------------------------------------------------
    def _rate_limited(self, token: str) -> bool:
        """Return True if ``token`` was emitted within the repeat-guard window."""
        now = time.perf_counter()
        last = self._last_emit.get(token)
        if last is not None and (now - last) < self.key_repeat_guard_s:
            return True
        self._last_emit[token] = now
        return False

    def _resolve_key(self, key: str) -> Any:
        """Map a logical keyboard key name to a pynput Key / KeyCode."""
        kbd = self._kbd_mod
        specials = {
            "space": kbd.Key.space,
            "shift": kbd.Key.shift,
            "ctrl": kbd.Key.ctrl,
            "alt": kbd.Key.alt,
            "cmd": kbd.Key.cmd,
            "enter": kbd.Key.enter,
            "tab": kbd.Key.tab,
            "esc": kbd.Key.esc,
        }
        if key in specials:
            return specials[key]
        return kbd.KeyCode.from_char(key)

    def _resolve_button(self, key: str) -> Any:
        buttons = {
            "mouse_left": self._mouse_mod.Button.left,
            "mouse_right": self._mouse_mod.Button.right,
            "mouse_middle": self._mouse_mod.Button.middle,
        }
        return buttons[key]

    # --- primitives ---------------------------------------------------------
    def _emit_key_down(self, key: str) -> None:
        if key in _MOUSE_BUTTONS:
            self._mouse.press(self._resolve_button(key))
        else:
            self._keyboard.press(self._resolve_key(key))

    def _emit_key_up(self, key: str) -> None:
        if key in _MOUSE_BUTTONS:
            self._mouse.release(self._resolve_button(key))
        else:
            self._keyboard.release(self._resolve_key(key))

    def _emit_mouse_move(self, dx: float, dy: float) -> None:
        q = self._quartz
        sx = int(round(dx * self.mouse_delta_scale))
        sy = int(round(dy * self.mouse_delta_scale))
        if sx == 0 and sy == 0:
            return
        event = q.CGEventCreateMouseEvent(
            None, q.kCGEventMouseMoved, (0.0, 0.0), q.kCGMouseButtonLeft
        )
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventDeltaX, sx)
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventDeltaY, sy)
        q.CGEventPost(q.kCGHIDEventTap, event)

    def _emit_scroll(self, clicks: int) -> None:
        if self._rate_limited(f"scroll:{1 if clicks > 0 else -1}"):
            return
        self._mouse.scroll(0, clicks)

    def __enter__(self) -> MacInputEmitter:
        return self

    def __exit__(self, *exc: object) -> None:
        # Never leave a key stuck down on exit (normal or exceptional).
        try:
            self.release_all()
        except Exception:  # pragma: no cover - defensive shutdown
            print("[mac-emitter] warning: release_all failed during shutdown", file=sys.stderr)
