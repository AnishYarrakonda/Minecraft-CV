"""Guard tests for the native macOS window-pinning helper.

These stay pure: no Qt display, no real NSWindow. They only assert the helper is
defensive — it returns None and never raises, even with a bogus widget or off-darwin.
"""

import sys

from minecraft_cv.ui.macos_window import keep_window_in_front


class _FakeWidget:
    """Minimal stand-in; winId() returns something int() can't bridge to a window."""

    def winId(self) -> int:  # noqa: N802 - mirrors Qt's QWidget.winId
        return 0


def test_keep_window_in_front_never_raises() -> None:
    assert keep_window_in_front(_FakeWidget()) is None  # type: ignore[arg-type]


def test_keep_window_in_front_is_noop_off_darwin(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "linux")
    # Off-darwin must short-circuit before touching the widget at all.
    sentinel = object()
    assert keep_window_in_front(sentinel) is None  # type: ignore[arg-type]
