"""Real macOS input backend: pynput keyboard/scroll + Quartz CGEvent mouse-look.

Opt-in only (``input.enabled: true`` / ``--input``). Never imported during tests — all heavy
imports (pynput, Quartz, ApplicationServices) happen inside ``__init__`` / methods so that
``import minecraft_cv.input`` stays light and camera/OS-free.

Permissions: emitting synthetic input requires **Accessibility / Input Monitoring** granted
to the terminal app running Python (System Settings -> Privacy & Security). Without it, events
are silently dropped. We best-effort detect this at startup and raise a clear error.

Mouse look uses Quartz relative deltas (``kCGMouseEventDeltaX/Y``) because Minecraft reads
true relative motion. The default path emits the thumb's per-frame movement directly rather
than maintaining a velocity stream.
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

    def __init__(
        self,
        mouse_delta_scale: float = 58.0,
        key_repeat_guard_ms: float = 50.0,
    ) -> None:
        """Construct the emitter and verify input permissions.

        Args:
            mouse_delta_scale: Multiplier from normalized thumb delta to relative pixels.
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
        # Main-display pixel size for normalized absolute-cursor coordinates.
        try:
            main = Quartz.CGMainDisplayID()
            self._screen_w = float(Quartz.CGDisplayPixelsWide(main))
            self._screen_h = float(Quartz.CGDisplayPixelsHigh(main))
        except Exception:  # pragma: no cover - environment-dependent
            self._screen_w, self._screen_h = 1920.0, 1080.0
        # Fractional-pixel carry for mouse-look: small per-frame deltas would otherwise round
        # to 0 px every frame (slow looks never move, fast looks jump). Accumulate the residual.
        self._move_accum_x = 0.0
        self._move_accum_y = 0.0

        self._activity = None
        self._disable_app_nap()
        self._check_permissions()

    # --- app nap prevention ------------------------------------------------
    def _disable_app_nap(self) -> None:
        """Prevent macOS from throttling the pipeline when the window is occluded."""
        try:
            import Foundation

            options = (
                Foundation.NSActivityUserInitiated
                | Foundation.NSActivityLatencyCritical
            )
            self._activity = (
                Foundation.NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                    options, "minecraft_cv live tracking"
                )
            )
        except Exception:
            pass

    def _enable_app_nap(self) -> None:
        """Allow macOS to throttle the app again."""
        if self._activity is not None:
            try:
                import Foundation

                Foundation.NSProcessInfo.processInfo().endActivity_(self._activity)
            except Exception:
                pass
            self._activity = None

    # --- permission detection ----------------------------------------------
    def _check_permissions(self) -> None:
        """Best-effort check that this process may emit synthetic input. Prompts if missing."""
        trusted = True
        try:  # pragma: no cover - macOS-only, environment-dependent
            import ApplicationServices  # type: ignore[import-untyped]
            import CoreFoundation  # type: ignore[import-untyped]

            key = CoreFoundation.CFSTR("AXTrustedCheckOptionPrompt")
            options = {key: True}
            if hasattr(ApplicationServices, "AXIsProcessTrustedWithOptions"):
                trusted = bool(ApplicationServices.AXIsProcessTrustedWithOptions(options))
            elif hasattr(ApplicationServices, "AXIsProcessTrusted"):
                trusted = bool(ApplicationServices.AXIsProcessTrusted())
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

    def _current_mouse_position(self) -> tuple[float, float]:
        """Return the current cursor position, falling back to display center."""
        q = self._quartz
        try:  # pragma: no cover - exact CGPoint shape is platform-dependent
            event = q.CGEventCreate(None)
            loc = q.CGEventGetLocation(event)
            if hasattr(loc, "x") and hasattr(loc, "y"):
                return float(loc.x), float(loc.y)
            return float(loc[0]), float(loc[1])
        except Exception:
            return self._screen_w / 2.0, self._screen_h / 2.0

    def _clamp_mouse_position(self, x: float, y: float) -> tuple[float, float]:
        """Clamp a display pixel position to the main display bounds."""
        return (
            min(max(float(x), 0.0), self._screen_w),
            min(max(float(y), 0.0), self._screen_h),
        )

    def _mouse_button_event_spec(self, key: str) -> tuple[Any, Any, Any]:
        """Map logical mouse key to Quartz button/down/up constants."""
        q = self._quartz
        if key == "mouse_left":
            return q.kCGMouseButtonLeft, q.kCGEventLeftMouseDown, q.kCGEventLeftMouseUp
        if key == "mouse_right":
            return q.kCGMouseButtonRight, q.kCGEventRightMouseDown, q.kCGEventRightMouseUp
        return (
            getattr(q, "kCGMouseButtonCenter", 2),
            q.kCGEventOtherMouseDown,
            q.kCGEventOtherMouseUp,
        )

    def _emit_mouse_button(self, key: str, *, down: bool) -> None:
        """Emit a Quartz button event at the current cursor position."""
        q = self._quartz
        button, down_type, up_type = self._mouse_button_event_spec(key)
        event = q.CGEventCreateMouseEvent(
            None,
            down_type if down else up_type,
            self._current_mouse_position(),
            button,
        )
        q.CGEventPost(q.kCGHIDEventTap, event)

    def _post_relative_pixels(self, sx: int, sy: int) -> None:
        """Post one relative mouse event with integer pixel deltas."""
        if sx == 0 and sy == 0:
            return
        q = self._quartz
        cx, cy = self._current_mouse_position()
        pos = self._clamp_mouse_position(cx + sx, cy + sy)
        event = q.CGEventCreateMouseEvent(
            None, q.kCGEventMouseMoved, pos, q.kCGMouseButtonLeft
        )
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventDeltaX, sx)
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventDeltaY, sy)
        q.CGEventPost(q.kCGHIDEventTap, event)

    # --- primitives ---------------------------------------------------------
    def _emit_key_down(self, key: str) -> None:
        if key in _MOUSE_BUTTONS:
            self._emit_mouse_button(key, down=True)
        else:
            self._keyboard.press(self._resolve_key(key))

    def _emit_key_up(self, key: str) -> None:
        if key in _MOUSE_BUTTONS:
            self._emit_mouse_button(key, down=False)
        else:
            self._keyboard.release(self._resolve_key(key))

    def _emit_mouse_move(self, dx: float, dy: float) -> None:
        """Emit mouse movement immediately — smoothing is the upstream filter's job."""
        self._move_accum_x += dx * self.mouse_delta_scale
        self._move_accum_y += dy * self.mouse_delta_scale
        sx = int(self._move_accum_x)
        sy = int(self._move_accum_y)
        if sx != 0 or sy != 0:
            self._move_accum_x -= sx
            self._move_accum_y -= sy
            self._post_relative_pixels(sx, sy)

    def _emit_mouse_stop(self) -> None:
        self._move_accum_x = 0.0
        self._move_accum_y = 0.0

    def _emit_mouse_move_abs(self, x: float, y: float) -> None:
        self._emit_mouse_stop()
        q = self._quartz
        # Clamp normalized coords and scale to main-display pixels. Absolute warp (no delta
        # fields) so the cursor jumps to the GUI position without rotating the camera.
        px = min(max(x, 0.0), 1.0) * self._screen_w
        py = min(max(y, 0.0), 1.0) * self._screen_h
        event = q.CGEventCreateMouseEvent(
            None, q.kCGEventMouseMoved, (px, py), q.kCGMouseButtonLeft
        )
        q.CGEventPost(q.kCGHIDEventTap, event)

    def _emit_scroll(self, clicks: int) -> None:
        if self._rate_limited(f"scroll:{1 if clicks > 0 else -1}"):
            return
        self._mouse.scroll(0, clicks)

    def __enter__(self) -> MacInputEmitter:
        """Return this emitter for context-manager use."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release all held input on context-manager exit."""
        # Never leave a key stuck down on exit (normal or exceptional).
        try:
            self.release_all()
        except Exception:  # pragma: no cover - defensive shutdown
            print("[mac-emitter] warning: release_all failed during shutdown", file=sys.stderr)
        finally:
            self._enable_app_nap()
