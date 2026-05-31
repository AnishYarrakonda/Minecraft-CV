"""Background worker that drives the :class:`FrameProcessor` off the GUI thread.

Qt's UI thread (like OpenCV HighGUI) must not be blocked by the capture/inference loop, so the
whole capture -> track -> step core runs here and pushes finished frames to the GUI via the
``frame_ready`` queued signal. Start/Stop, Go-Live, and Calibrate are requested from the GUI
thread via simple flags and applied at the top of the loop on this thread.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal

from minecraft_cv.runtime import FrameProcessor

if TYPE_CHECKING:
    from minecraft_cv.capture.source import FrameSource
    from minecraft_cv.config import Settings


class PipelineWorker(QObject):
    """Runs the real-time pipeline and emits one signal per processed frame."""

    frame_ready = Signal(object)  # FramePacket
    error = Signal(str)
    started_ok = Signal()
    live_changed = Signal(bool)
    stopped = Signal()

    def __init__(self, settings: Settings, source: FrameSource | None = None) -> None:
        """Create the worker (does not open the camera until :meth:`run`).

        Args:
            settings: Loaded configuration.
            source: Optional injected frame source (defaults to the live camera).
        """
        super().__init__()
        self._settings = settings
        self._source = source
        self._stop = threading.Event()
        self._recenter_pending = False
        self._live_pending: bool | None = None

    def request_stop(self) -> None:
        """Ask the loop to finish and shut down (thread-safe)."""
        self._stop.set()

    def request_recenter(self) -> None:
        """Ask the pipeline to recenter both joysticks (thread-safe)."""
        self._recenter_pending = True

    def request_live(self, enabled: bool) -> None:
        """Ask to switch Live/Dry-Run input mode (thread-safe)."""
        self._live_pending = enabled

    def run(self) -> None:
        """Open the camera/tracker and process frames until stopped or exhausted."""
        try:
            processor = FrameProcessor.from_settings(self._settings, source=self._source)
            processor.start()
        except Exception as exc:  # noqa: BLE001 - surface any startup failure to the UI
            self.error.emit(f"Could not start camera or tracker:\n{exc}")
            self.stopped.emit()
            return

        self.started_ok.emit()
        try:
            while not self._stop.is_set():
                if self._recenter_pending:
                    self._recenter_pending = False
                    processor.recenter()
                if self._live_pending is not None:
                    want, self._live_pending = self._live_pending, None
                    if want != processor.live:
                        try:
                            processor.set_live(want)
                        except Exception as exc:  # noqa: BLE001 - permission/import failure
                            self.error.emit(
                                "Could not enable Live input. Grant Accessibility / Input "
                                "Monitoring to your terminal in System Settings → Privacy & "
                                f"Security, then try again.\n\n{exc}"
                            )
                    self.live_changed.emit(processor.live)

                if processor.error is not None:
                    self.error.emit(f"Camera error:\n{processor.error}")
                    break
                try:
                    packet = processor.process_once()
                except Exception as exc:  # noqa: BLE001 - stall / capture failure
                    self.error.emit(str(exc))
                    break
                if packet is None:
                    if processor.exhausted:
                        break
                    QThread.msleep(2)
                    continue
                self.frame_ready.emit(packet)
        finally:
            with contextlib.suppress(Exception):  # shutdown is best-effort
                processor.shutdown()
            self.stopped.emit()


__all__ = ["PipelineWorker"]
