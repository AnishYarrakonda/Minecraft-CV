"""Knuckle-tilt joystick signal — the default front-facing-camera steering mode.

The palm-tilt mode steers from the *image-plane direction of the hand*: the vector from the
wrist to the knuckle (MCP) centroid, projected onto the camera's ``(x, y)`` plane. Tilting a
resting hand at the wrist swings the knuckles across the frame, producing a large,
sign-stable 2D signal.

It deliberately avoids the two weak signals this project previously shipped:

- the **palm normal**'s ``(x, y)`` projection, which is near-zero and noise-dominated in the
  comfortable palm-down resting range and whose stabilizing hemisphere clamp collapses
  left-tilt and right-tilt onto the same vector, and
- the MediaPipe **depth (``z``)** axis, which is the least reliable landmark coordinate.

The tilt signal is translation-invariant (a difference of two landmarks), scale-invariant
(divided by the hand span), and immune to finger curl/pinch (MCP-based, not fingertip-based),
so the same signal drives WASD/look and the inventory cursor without re-tuning.
"""

from __future__ import annotations

import numpy as np

from minecraft_cv.joystick.wrist_rotation import palm_vector


def palm_tilt_xy(landmarks: np.ndarray) -> np.ndarray:
    """Return the image-plane ``(x, y)`` of the wrist->MCP-centroid vector.

    Args:
        landmarks: ``(21, 3)`` MediaPipe hand landmarks. ``x``/``y`` are normalized frame
            coordinates where ``y`` increases downward; ``z`` is relative depth and is ignored.

    Returns:
        A ``(2,)`` array. Index ``0`` is horizontal hand tilt (knuckles left/right of the
        wrist in the frame); index ``1`` is vertical hand tilt (larger = knuckles lower in the
        frame, i.e. the hand tilted down). Both components are normalized by the
        wrist->middle-MCP span, so they are invariant to how close the hand is to the camera.
    """
    vec = palm_vector(landmarks)
    return np.asarray([vec[0], vec[1]], dtype=np.float64)


__all__ = ["palm_tilt_xy"]
