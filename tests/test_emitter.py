"""Tests for the input emitter interface and NullEmitter (no OS input ever).

These tests NEVER instantiate MacInputEmitter — they only exercise NullEmitter and the
create_emitter factory, upholding hard invariant #2 (no real input in tests).
"""

from __future__ import annotations

from minecraft_cv.config import Settings
from minecraft_cv.input.emitter import InputEmitter, NullEmitter, create_emitter


def test_null_emitter_logs_calls() -> None:
    em = NullEmitter()
    em.key_down("space")
    em.mouse_move(1.5, -2.0)
    em.scroll(1)
    em.key_up("space")
    assert em.log == [
        ("key_down", "space"),
        ("mouse_move", "1.500000", "-2.000000"),
        ("scroll", "1"),
        ("key_up", "space"),
    ]


def test_key_down_is_deduplicated_and_refcounted() -> None:
    em = NullEmitter()
    em.key_down("w")
    em.key_down("w")  # count is now 2 -> no second press
    assert em.log == [("key_down", "w")]
    assert em.held_keys == frozenset({"w"})

    em.key_up("w")  # count goes to 1 -> no release yet
    assert em.log == [("key_down", "w")]
    assert em.held_keys == frozenset({"w"})

    em.key_up("w")  # count goes to 0 -> released
    assert em.log == [("key_down", "w"), ("key_up", "w")]
    assert em.held_keys == frozenset()


def test_key_up_for_unheld_key_is_noop() -> None:
    em = NullEmitter()
    em.key_up("space")  # never pressed
    assert em.log == []


def test_release_all_releases_held_keys() -> None:
    em = NullEmitter()
    em.key_down("space")
    em.key_down("mouse_left")
    em.release_all()
    # release_all emits key_up for each held key (sorted for determinism).
    assert ("key_up", "mouse_left") in em.log
    assert ("key_up", "space") in em.log
    assert em.held_keys == frozenset()


def test_scroll_zero_is_ignored() -> None:
    em = NullEmitter()
    em.scroll(0)
    assert em.log == []


def test_create_emitter_returns_null_when_disabled() -> None:
    settings = Settings()  # input.enabled is False by default
    em = create_emitter(settings)
    assert isinstance(em, NullEmitter)
    assert isinstance(em, InputEmitter)


def test_create_emitter_does_not_emit_by_default() -> None:
    settings = Settings()
    em = create_emitter(settings)
    # A freshly created default emitter has emitted nothing.
    assert isinstance(em, NullEmitter)
    assert em.log == []
