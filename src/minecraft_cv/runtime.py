"""Shared real-time frame core: capture -> track -> :meth:`Pipeline.step`.

This module extracts the per-frame work that used to live inline in
:func:`minecraft_cv.pipeline.run_pipeline` into a reusable :class:`FrameProcessor`. Both the
legacy OpenCV debug loop (``run_pipeline``) and the PySide6 desktop app drive the same core
through :meth:`FrameProcessor.process_once`, so there is exactly one capture/tracking path.

The processor is **Qt-free** and OS-input-safe: with an injected :class:`FrameSource`, a fake
tracker, and a :class:`NullEmitter` it is fully unit-testable with no camera and no real input
(hard invariant #2). OpenCV is imported lazily so importing this module stays light.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from minecraft_cv.pipeline import Pipeline, StepResult
from minecraft_cv.tracking.tracker import HandResult, HandTracker

if TYPE_CHECKING:
    from minecraft_cv.capture.source import FrameSource
    from minecraft_cv.config import Settings
    from minecraft_cv.input.emitter import InputEmitter

# Seconds without a fresh frame before the live camera is considered stalled.
_STALL_TIMEOUT_S = 2.0
# Exponential-moving-average weight for the live FPS readout (higher = snappier).
_FPS_EMA_ALPHA = 0.1


@dataclass
class FramePacket:
    """One processed frame, ready to render.

    Attributes:
        frame: BGR ``uint8`` display frame (already mirrored if configured). An owned copy —
            safe to hand to another thread; the processor reuses its own internal buffers.
        hands: Raw tracker results (0-2 :class:`HandResult`), landmarks normalized to ``[0, 1]``
            in the display frame's coordinates. Used to draw the hand skeleton.
        step: The :class:`StepResult` from :meth:`Pipeline.step` for this frame (gestures held,
            WASD, joystick vectors, per-hand status).
        fps: Smoothed processing rate in frames per second.
        live: Whether the pipeline is currently emitting real OS input.
    """

    frame: np.ndarray
    hands: list[HandResult] = field(default_factory=list)
    step: StepResult = field(default_factory=StepResult)
    fps: float = 0.0
    live: bool = False


class FrameProcessor:
    """Owns the capture buffer, tracker, and pipeline; advances one frame at a time."""

    def __init__(
        self,
        pipeline: Pipeline,
        source: FrameSource,
        tracker: HandTracker,
        settings: Settings,
    ) -> None:
        """Assemble a processor from already-constructed components.

        Args:
            pipeline: The per-frame controller (gestures + joysticks -> emitter).
            source: Frame source (camera or clip). Read on a background buffer thread.
            tracker: Hand tracker backend.
            settings: Loaded configuration (camera mirror, tracking resolution, input mode).
        """
        self.pipeline = pipeline
        self.source = source
        self.tracker = tracker
        self.settings = settings

        res_w, res_h = settings.tracking.input_resolution
        # Pre-allocated reuse buffers for the hot loop (no per-frame allocation).
        self._small_bgr = np.empty((res_h, res_w, 3), dtype=np.uint8)
        self._small_rgb = np.empty((res_h, res_w, 3), dtype=np.uint8)
        self._res = (res_w, res_h)
        self._mirror = settings.camera.mirror
        self._live = settings.input.enabled

        self._buffer: object | None = None  # FrameBuffer, built in start()
        self._last_seq = -1
        self._last_new_frame_t = time.monotonic()
        self._fps = 0.0
        self._last_processed_t: float | None = None
        self.processed = 0
        self.dropped = 0
        self._t_start = time.monotonic()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        source: FrameSource | None = None,
        allow_uncalibrated_palm_normal: bool = False,
    ) -> FrameProcessor:
        """Build a processor (pipeline + camera + tracker) from a :class:`Settings` model.

        Args:
            settings: Loaded configuration.
            source: Optional injected frame source. If ``None``, a live AVFoundation camera is
                opened from ``settings.camera`` (this blocks / may prompt for permission, so
                call this off the GUI thread).
            allow_uncalibrated_palm_normal: Safe-preview escape hatch forwarded to the pipeline.

        Returns:
            A ready-to-:meth:`start` :class:`FrameProcessor`.
        """
        pipeline = Pipeline.from_settings(
            settings, allow_uncalibrated_palm_normal=allow_uncalibrated_palm_normal
        )
        if source is None:
            from minecraft_cv.capture.source import AVFoundationSource

            source = AVFoundationSource(
                index=settings.camera.index,
                width=settings.camera.width,
                height=settings.camera.height,
                fps=settings.camera.fps,
            )
        tracker = HandTracker.create(settings.tracking.backend, settings.tracking.device)
        return cls(pipeline, source, tracker, settings)

    def start(self) -> FrameProcessor:
        """Start the background capture buffer thread (idempotent)."""
        if self._buffer is None:
            from minecraft_cv.capture.buffer import FrameBuffer

            self._buffer = FrameBuffer(self.source).start()
            self._t_start = time.monotonic()
            self._last_new_frame_t = self._t_start
        return self

    @property
    def error(self) -> Exception | None:
        """The capture thread's stored exception, if any (camera failure)."""
        buf = self._buffer
        return getattr(buf, "error", None) if buf is not None else None

    @property
    def exhausted(self) -> bool:
        """True once the source has run out of frames and the latest has been processed."""
        buf = self._buffer
        if buf is None:
            return False
        return bool(getattr(buf, "exhausted", False))

    @property
    def fps(self) -> float:
        """Smoothed processing rate in frames per second."""
        return self._fps

    def process_once(self) -> FramePacket | None:
        """Process the newest available frame, or return ``None`` if there is nothing new.

        Returns:
            A :class:`FramePacket` when a fresh frame was processed; ``None`` when the buffer has
            no new frame yet (the caller should briefly sleep and retry) or the source is
            exhausted (check :attr:`exhausted`).

        Raises:
            Exception: Re-raises the capture thread's stored error.
            RuntimeError: If the live camera stalls (no new frame for ``_STALL_TIMEOUT_S``).
        """
        import cv2  # lazy: keeps the module importable without OpenCV

        buf = self._buffer
        if buf is None:
            self.start()
            buf = self._buffer
        assert buf is not None  # for type-checkers; start() always sets it

        if self.error is not None:
            raise self.error
        if not self.exhausted and time.monotonic() - self._last_new_frame_t > _STALL_TIMEOUT_S:
            raise RuntimeError("Camera stalled")

        seq, frame = buf.latest()  # type: ignore[attr-defined]
        if frame is None or seq == self._last_seq:
            return None
        if self._last_seq != -1 and seq > self._last_seq + 1:
            self.dropped += seq - self._last_seq - 1
        self._last_seq = seq
        self._last_new_frame_t = time.monotonic()

        # Mirror first so tracking, joystick vectors, WASD directions, and the rendered frame
        # all share one consistent (mirrored) frame of reference.
        if self._mirror:
            frame = cv2.flip(frame, 1)

        res_w, res_h = self._res
        cv2.resize(frame, (res_w, res_h), dst=self._small_bgr)
        cv2.cvtColor(self._small_bgr, cv2.COLOR_BGR2RGB, dst=self._small_rgb)

        results = self.tracker.detect(self._small_rgb)
        step = self.pipeline.step(results)
        self.processed += 1
        self._update_fps()

        return FramePacket(
            frame=frame.copy(),
            hands=list(results),
            step=step,
            fps=self._fps,
            live=self._live,
        )

    def _update_fps(self) -> None:
        now = time.monotonic()
        if self._last_processed_t is not None:
            dt = now - self._last_processed_t
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0.0 else (
                    (1.0 - _FPS_EMA_ALPHA) * self._fps + _FPS_EMA_ALPHA * inst
                )
        self._last_processed_t = now

    def recenter(self) -> None:
        """Recenter both joysticks at the current hand position (the app's 'Calibrate')."""
        self.pipeline.recenter()

    @property
    def live(self) -> bool:
        """Whether real OS input is currently being emitted."""
        return self._live

    def set_live(self, enabled: bool) -> None:
        """Swap the pipeline emitter between Dry-Run (``NullEmitter``) and Live (macOS).

        Args:
            enabled: ``True`` to emit real keyboard/mouse input; ``False`` for a safe no-op.

        Raises:
            Exception: If the macOS input backend cannot be created (e.g. missing Accessibility
                permission). The caller should surface this and stay in Dry-Run.
        """
        emitter: InputEmitter
        if enabled:
            from minecraft_cv.input.mac_emitter import MacInputEmitter

            emitter = MacInputEmitter(
                mouse_delta_scale=self.settings.input.mouse_delta_scale,
                key_repeat_guard_ms=self.settings.input.key_repeat_guard_ms,
            )
        else:
            from minecraft_cv.input.emitter import NullEmitter

            emitter = NullEmitter()
        self.pipeline.set_emitter(emitter)
        self._live = enabled

    def shutdown(self) -> None:
        """Release all held input, stop the buffer thread, and close the tracker."""
        self.pipeline.shutdown()
        buf = self._buffer
        if buf is not None:
            buf.stop()  # type: ignore[attr-defined]
            self._buffer = None
        self.tracker.close()

    @property
    def elapsed(self) -> float:
        """Seconds since the buffer thread started."""
        return time.monotonic() - self._t_start


__all__ = ["FramePacket", "FrameProcessor"]
