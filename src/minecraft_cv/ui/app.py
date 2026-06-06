"""Application bootstrap and the main window assembly.

``run_app`` is the entry point used by both ``mcv ui`` and ``python main.py``. It stacks the
camera "painting" (:class:`CameraView`) on top of the compact key grid (:class:`KeymapPanel`),
with the header above, and wires them to a :class:`PipelineWorker` running on its own thread,
defaulting to safe Dry-Run.

Importing this module requires PySide6; callers that want to degrade gracefully when it is
absent (the CLI) should guard the import and surface an install hint.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from minecraft_cv.runtime import FrameProcessor
from minecraft_cv.ui.camera_view import CameraView
from minecraft_cv.ui.macos_window import keep_window_in_front, reset_window_level
from minecraft_cv.ui.panels import HeaderBar, KeymapPanel
from minecraft_cv.ui.theme import apply_theme
from minecraft_cv.ui.worker import PipelineWorker

if TYPE_CHECKING:
    from minecraft_cv.capture.source import FrameSource
    from minecraft_cv.config import Settings
    from minecraft_cv.runtime import FramePacket

# Below this window height (px), the key grid + sensitivity slider hide, leaving the header
# bar and camera only — the window collapses into a compact in-game HUD.
_COLLAPSE_HEIGHT = 460


def run_app(settings: Settings, source: FrameSource | None = None) -> int:
    """Launch the desktop app and run the Qt event loop until the window closes.

    Args:
        settings: Loaded configuration.
        source: Optional injected frame source (defaults to the live camera).

    Returns:
        Process exit code (0 on clean exit).
    """
    app = QApplication.instance() or QApplication(sys.argv[:1])
    apply_theme(app)
    window = MainWindow(settings, source=source)
    window.show()
    return int(app.exec())


class MainWindow(QMainWindow):
    """The framed application: header controls above the camera painting and the live key grid."""

    def __init__(self, settings: Settings, source: FrameSource | None = None) -> None:
        """Assemble the window and auto-start the capture session in Dry-Run."""
        super().__init__()
        self._settings = settings
        self._source = source
        self._thread: threading.Thread | None = None
        self._worker: PipelineWorker | None = None

        # Narrow + tall by default to save horizontal space alongside Minecraft; the minimum
        # is small enough to shrink into a camera-only HUD (the key grid hides below
        # ``_COLLAPSE_HEIGHT``).
        self.setWindowTitle("minecraft_cv")
        self.resize(480, 710)
        self.setMinimumSize(320, 240)
        self._pinned = False

        root = QWidget()
        root.setObjectName("Root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(10)

        # Aspect-locked camera on top: its height follows its width, so the feed fills the
        # widget (no black letterbox bands) and a vertical resize never shrinks it.
        self._camera = CameraView(settings.tracking.swap_handedness)
        outer.addWidget(self._camera)

        # Compact key grid sits directly under the camera and fills the space down to the
        # control bar. Extra height shows as empty panel below it (drag the window shorter to
        # compact); when squeezed it scrolls — the aspect-locked camera never shrinks for it.
        self._keymap = KeymapPanel(settings)
        outer.addWidget(self._keymap)

        # Status + Start/Live/Calibrate/Pin moved to the bottom; wraps to fit narrow widths.
        self._header = HeaderBar()
        outer.addWidget(self._header)

        self.setCentralWidget(root)

        self._header.startStopClicked.connect(self._toggle_session)
        self._header.liveToggled.connect(self._on_live_toggled)
        self._header.calibrateClicked.connect(self._on_calibrate)
        self._header.pinToggled.connect(self._on_pin)
        self._keymap.sensitivityChanged.connect(self._on_sensitivity_changed)

        self._start_session()

    # --- session lifecycle --------------------------------------------------
    def _start_session(self) -> None:
        if self._thread is not None:
            return
        self._worker = PipelineWorker(self._settings, source=self._source)
        # Use a Python thread, not QThread: QThread's GL context setup causes mediapipe
        # to take a GPU landmark-projection path that crashes on macOS (SIGTRAP). Qt
        # signals emitted from Python threads are queued safely through the event loop.
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error.connect(self._on_error)
        self._worker.started_ok.connect(lambda: self._header.set_running(True))
        self._worker.live_changed.connect(self._header.set_live)
        self._worker.stopped.connect(self._on_worker_stopped)
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()
        self._header.set_running(True)

    def _stop_session(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    def _toggle_session(self) -> None:
        if self._thread is None:
            self._start_session()
        else:
            self._stop_session()

    def _on_worker_stopped(self) -> None:
        if self._thread is not None:
            # Join unbounded: request_stop() has been set, so the loop exits after at most one
            # process_once, and the worker's `finally: processor.shutdown()` must run to release
            # any held input. A timeout here could abandon the daemon worker mid-gesture and
            # leave a key stuck down in-game.
            self._thread.join()
            self._thread = None
        self._worker = None
        self._header.set_running(False)
        self._header.set_live(False)

    # --- signals from the header --------------------------------------------
    def _on_live_toggled(self, enabled: bool) -> None:
        if self._worker is None:
            return
        emitter = None
        if enabled:
            # Build the macOS input emitter HERE, on the GUI main thread. pynput's keyboard
            # backend touches main-thread-only TIS APIs at construction; building it on the
            # worker thread under the running Qt event loop SIGTRAPs the whole process. The
            # worker then only installs the ready emitter (emission off-thread is safe).
            try:
                emitter = FrameProcessor.build_live_emitter(self._settings)
            except Exception as exc:  # noqa: BLE001 - surface permission/import failure
                QMessageBox.warning(
                    self,
                    "minecraft_cv",
                    "Could not enable Live input. Grant Accessibility / Input Monitoring to "
                    "your terminal in System Settings → Privacy & Security, then try again."
                    f"\n\n{exc}",
                )
                self._header.set_live(False)
                return
        self._worker.request_live(enabled, emitter)

    def _on_calibrate(self) -> None:
        if self._worker is not None:
            self._worker.request_recenter()

    def _on_pin(self, pinned: bool) -> None:
        """Pin/unpin the window above other apps, including fullscreen Minecraft.

        Replaces the standalone overlay: a cross-platform ``WindowStaysOnTopHint`` plus a native
        macOS window level (``keep_window_in_front``) so it floats over fullscreen Spaces.
        """
        self._pinned = pinned
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, pinned)
        self.show()  # re-show is required after changing window flags
        if pinned:
            keep_window_in_front(self)
        else:
            reset_window_level(self)

    def _on_sensitivity_changed(self, val: float) -> None:
        if self._worker is not None:
            self._worker.request_sensitivity(val)

    # --- signals from the worker --------------------------------------------
    def _on_frame(self, packet: FramePacket) -> None:
        self._camera.set_packet(packet)
        step = packet.step
        self._header.set_status(step.left_status, step.right_status, step.face_status)
        self._keymap.update_state(step)

    def _on_error(self, message: str) -> None:
        QMessageBox.warning(self, "minecraft_cv", message)

    # --- responsive collapse ------------------------------------------------
    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        """Hide the key grid below ``_COLLAPSE_HEIGHT`` so the window becomes a camera-only HUD."""
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._keymap.setVisible(self.height() >= _COLLAPSE_HEIGHT)

    # --- teardown -----------------------------------------------------------
    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        """Stop the session and release any held input before the window closes."""
        if self._worker is not None:
            self._worker.request_stop()
        if self._thread is not None:
            # Unbounded join so the worker's shutdown (which releases all held input) always
            # runs before the process exits — never abandon a Live worker with keys held down.
            self._thread.join()
            self._thread = None
        self._worker = None
        super().closeEvent(event)  # type: ignore[arg-type]


__all__ = ["MainWindow", "run_app"]
