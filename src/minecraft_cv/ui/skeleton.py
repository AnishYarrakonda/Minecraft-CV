"""Hand-skeleton geometry helpers for the camera overlay.

Qt-free and pure so the letterbox/mapping math can be unit-tested without PySide6. The
:class:`~minecraft_cv.ui.camera_view.CameraView` uses these to map MediaPipe's normalized
``[0, 1]`` landmarks onto the on-screen (letterboxed) camera image and to know which joints
to connect.
"""

from __future__ import annotations

# Standard MediaPipe Hands 21-landmark topology (wrist=0, then thumb/index/middle/ring/pinky).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                # palm base
)

# Landmark indices that are fingertips (drawn slightly larger / brighter).
FINGERTIPS: tuple[int, ...] = (4, 8, 12, 16, 20)


def fit_rect(
    img_w: float, img_h: float, widget_w: float, widget_h: float
) -> tuple[float, float, float, float]:
    """Letterbox an image into a widget, preserving aspect ratio and centering.

    Args:
        img_w: Source image width (pixels).
        img_h: Source image height (pixels).
        widget_w: Destination widget width (pixels).
        widget_h: Destination widget height (pixels).

    Returns:
        ``(x, y, w, h)`` of the drawn image rectangle within the widget, in widget pixels.
        Returns ``(0, 0, 0, 0)`` for non-positive inputs.
    """
    if img_w <= 0 or img_h <= 0 or widget_w <= 0 or widget_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    scale = min(widget_w / img_w, widget_h / img_h)
    w = img_w * scale
    h = img_h * scale
    x = (widget_w - w) / 2.0
    y = (widget_h - h) / 2.0
    return (x, y, w, h)


def to_widget(
    norm_x: float, norm_y: float, rect: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Map a normalized ``[0, 1]`` landmark to widget pixel coordinates inside ``rect``.

    Args:
        norm_x: Normalized x in ``[0, 1]`` (0 = left edge of the image).
        norm_y: Normalized y in ``[0, 1]`` (0 = top edge of the image).
        rect: The letterboxed image rectangle ``(x, y, w, h)`` from :func:`fit_rect`.

    Returns:
        ``(px, py)`` in widget pixels.
    """
    x, y, w, h = rect
    return (x + norm_x * w, y + norm_y * h)


__all__ = ["FINGERTIPS", "HAND_CONNECTIONS", "fit_rect", "to_widget"]
