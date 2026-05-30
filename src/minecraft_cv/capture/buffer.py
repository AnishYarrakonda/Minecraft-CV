"""Threaded single-slot frame buffer (newest-frame-wins).

Decouples capture from processing: a background thread reads from a :class:`FrameSource` as
fast as the device allows and stores only the **latest** frame in a one-slot buffer. The
processing loop pulls the newest frame each tick; stale frames are dropped rather than
queued. For a real-time controller a *dropped old frame* beats a *late fresh action*
(see ``.claude/rules/opencv-pytorch.md`` §3).
"""

from __future__ import annotations

import threading

import numpy as np

from minecraft_cv.capture.source import FrameSource


class FrameBuffer:
    """Background-thread reader exposing only the most recent frame."""

    def __init__(self, source: FrameSource) -> None:
        """Wrap a frame source. Call :meth:`start` to begin reading."""
        self._source = source
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._seq = 0  # increments per stored frame (lets a consumer detect new frames)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._exhausted = False

    def start(self) -> FrameBuffer:
        """Start the background capture thread (idempotent)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="frame-buffer", daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._source.read()
            if frame is None:
                # Source exhausted (clip ended) or a transient miss: stop reading.
                self._exhausted = True
                break
            with self._lock:
                self._latest = frame
                self._seq += 1

    def latest(self) -> tuple[int, np.ndarray | None]:
        """Return ``(seq, frame)`` for the newest frame (``seq`` 0 / frame None if none yet).

        The sequence number lets a consumer skip reprocessing a frame it already handled.
        """
        with self._lock:
            return self._seq, self._latest

    @property
    def exhausted(self) -> bool:
        """True once the underlying source has run out of frames (e.g. clip ended)."""
        return self._exhausted

    def stop(self) -> None:
        """Signal the thread to stop, join it, and release the source. Idempotent."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
        self._source.release()

    def __enter__(self) -> FrameBuffer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
