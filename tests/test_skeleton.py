"""Pure geometry tests for the hand-skeleton overlay helpers (no PySide6 needed)."""
from __future__ import annotations

import pytest

from minecraft_cv.ui.skeleton import FINGERTIPS, HAND_CONNECTIONS, fit_rect, to_widget


def test_fit_rect_letterboxes_4_3_into_16_9() -> None:
    x, y, w, h = fit_rect(640, 480, 1280, 720)
    # min(1280/640, 720/480) = min(2.0, 1.5) = 1.5 -> 960x720, centered horizontally.
    assert (w, h) == pytest.approx((960.0, 720.0))
    assert x == pytest.approx(160.0)
    assert y == pytest.approx(0.0)


def test_fit_rect_letterboxes_wide_into_square() -> None:
    x, y, w, h = fit_rect(640, 480, 480, 480)
    # scale = min(0.75, 1.0) = 0.75 -> 480x360, centered vertically.
    assert (w, h) == pytest.approx((480.0, 360.0))
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(60.0)


def test_fit_rect_nonpositive_returns_zero() -> None:
    assert fit_rect(0, 480, 1280, 720) == (0.0, 0.0, 0.0, 0.0)
    assert fit_rect(640, 480, 0, 720) == (0.0, 0.0, 0.0, 0.0)


def test_to_widget_maps_corners_and_center() -> None:
    rect = (160.0, 0.0, 960.0, 720.0)
    assert to_widget(0.0, 0.0, rect) == pytest.approx((160.0, 0.0))
    assert to_widget(1.0, 1.0, rect) == pytest.approx((1120.0, 720.0))
    assert to_widget(0.5, 0.5, rect) == pytest.approx((640.0, 360.0))


def test_hand_connections_indices_in_range() -> None:
    for a, b in HAND_CONNECTIONS:
        assert 0 <= a <= 20
        assert 0 <= b <= 20
        assert a != b


def test_fingertips_are_valid_distinct_indices() -> None:
    assert len(FINGERTIPS) == 5
    assert len(set(FINGERTIPS)) == 5
    assert all(0 <= i <= 20 for i in FINGERTIPS)
