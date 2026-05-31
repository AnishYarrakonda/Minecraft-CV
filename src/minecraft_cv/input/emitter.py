"""OS-input emitter interface and the default no-op implementation.

Hard invariant #2: the input emitter is a **no-op by default**. Tests and ``--no-input``
dry-runs must never move the real mouse or press real keys. The real macOS implementation
(:class:`~minecraft_cv.input.mac_emitter.MacInputEmitter`) is opt-in and imported lazily so
that pynput / Quartz are never imported during tests.

The ABC owns held-key bookkeeping (dedup + ``release_all``) so every backend shares the same
safety semantics; subclasses implement only the four ``_emit_*`` primitives.

Logical key names: keyboard keys are passed by name (``"space"``, ``"shift"``, ``"e"``,
``"w"``); mouse buttons use the ``"mouse_left"`` / ``"mouse_right"`` prefix convention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minecraft_cv.config import Settings


class InputEmitter(ABC):
    """Abstract OS-input emitter with shared held-key tracking.

    Concrete backends implement the ``_emit_*`` primitives. The public ``key_down`` /
    ``key_up`` / ``release_all`` methods deduplicate presses and guarantee that every held
    key is releasable in one call (used on shutdown / tracking loss / crash).
    """

    def __init__(self) -> None:
        """Initialize held-key reference counts."""
        self._held_keys: dict[str, int] = {}

    # --- public API ---------------------------------------------------------
    def key_down(self, key: str) -> None:
        """Press ``key``. Emits OS down if refcount transitions 0 -> 1."""
        count = self._held_keys.get(key, 0)
        self._held_keys[key] = count + 1
        if count == 0:
            self._emit_key_down(key)

    def key_up(self, key: str) -> None:
        """Release ``key``. Emits OS up if refcount transitions 1 -> 0."""
        count = self._held_keys.get(key, 0)
        if count <= 0:
            return
        self._held_keys[key] = count - 1
        if self._held_keys[key] == 0:
            self._emit_key_up(key)
            del self._held_keys[key]

    def mouse_move(self, dx: float, dy: float) -> None:
        """Emit a relative mouse-look delta in screen pixels (camera rotation)."""
        self._emit_mouse_move(dx, dy)

    def mouse_stop(self) -> None:
        """Stop any backend-maintained continuous mouse-look motion."""
        self._emit_mouse_stop()

    def mouse_move_abs(self, x: float, y: float) -> None:
        """Warp the cursor to an absolute normalized screen position.

        Args:
            x: Target screen x in ``[0, 1]`` (0 = left edge, 1 = right edge).
            y: Target screen y in ``[0, 1]`` (0 = top edge, 1 = bottom edge).

        Backends scale the normalized coords to the main display size.
        """
        self._emit_mouse_move_abs(x, y)

    def scroll(self, clicks: int) -> None:
        """Emit ``clicks`` scroll ticks (positive = up = hotbar next)."""
        if clicks:
            self._emit_scroll(clicks)

    def key_tap(self, key: str) -> None:
        """Emit a momentary key tap (down + immediate up). Not tracked in held_keys.

        Used for one-shot actions where the gesture maps to a single key press rather than
        a hold.
        """
        self._emit_key_down(key)
        self._emit_key_up(key)

    def release_all(self) -> None:
        """Release every currently-held key/button. The OS-level fail-safe backstop."""
        self.mouse_stop()
        for key, count in sorted(self._held_keys.items()):
            if count > 0:
                self._emit_key_up(key)
        self._held_keys.clear()

    @property
    def held_keys(self) -> frozenset[str]:
        """The set of keys/buttons currently held down."""
        return frozenset(k for k, v in self._held_keys.items() if v > 0)

    # --- primitives implemented by backends ---------------------------------
    @abstractmethod
    def _emit_key_down(self, key: str) -> None: ...

    @abstractmethod
    def _emit_key_up(self, key: str) -> None: ...

    @abstractmethod
    def _emit_mouse_move(self, dx: float, dy: float) -> None: ...

    def _emit_mouse_stop(self) -> None:
        """Optional backend hook for clearing backend mouse-motion state."""
        return None

    @abstractmethod
    def _emit_mouse_move_abs(self, x: float, y: float) -> None: ...

    @abstractmethod
    def _emit_scroll(self, clicks: int) -> None: ...


class NullEmitter(InputEmitter):
    """No-op emitter that records calls for assertions. Default for tests/dry-runs.

    Every primitive appends to :attr:`log` and emits nothing to the OS. Held-key tracking
    and ``release_all`` behave exactly as in a real backend, so dropout/shutdown safety can
    be tested without touching the OS.
    """

    def __init__(self) -> None:
        """Create an empty in-memory event log."""
        super().__init__()
        self.log: list[tuple[str, ...]] = []

    def _emit_key_down(self, key: str) -> None:
        self.log.append(("key_down", key))

    def _emit_key_up(self, key: str) -> None:
        self.log.append(("key_up", key))

    def _emit_mouse_move(self, dx: float, dy: float) -> None:
        self.log.append(("mouse_move", f"{dx:.6f}", f"{dy:.6f}"))

    def _emit_mouse_stop(self) -> None:
        self.log.append(("mouse_stop",))

    def _emit_mouse_move_abs(self, x: float, y: float) -> None:
        self.log.append(("mouse_move_abs", f"{x:.6f}", f"{y:.6f}"))

    def _emit_scroll(self, clicks: int) -> None:
        self.log.append(("scroll", str(clicks)))

    def key_tap(self, key: str) -> None:
        """Record and emit a momentary key tap."""
        self.log.append(("key_tap", key))
        # Still emit the actual down/up for backends that need it
        self._emit_key_down(key)
        self._emit_key_up(key)


def create_emitter(settings: Settings) -> InputEmitter:
    """Factory: return a real emitter only when input is explicitly enabled.

    Args:
        settings: Loaded configuration. ``settings.input.enabled`` gates real emission.

    Returns:
        A :class:`NullEmitter` unless ``settings.input.enabled`` is True, in which case the
        macOS :class:`MacInputEmitter` is imported lazily and returned.
    """
    if not settings.input.enabled:
        return NullEmitter()
    # Imported lazily so pynput / Quartz are never imported in tests or dry-runs.
    from minecraft_cv.input.mac_emitter import MacInputEmitter

    return MacInputEmitter(
        mouse_delta_scale=settings.input.mouse_delta_scale,
        key_repeat_guard_ms=settings.input.key_repeat_guard_ms,
    )
