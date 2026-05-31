"""Application bootstrap and the main window assembly.

``run_app`` is the entry point used by both ``mcv ui`` and ``python main.py``. It wires the
camera "painting" (:class:`CameraView`) and the sidebar "frame" (header + keymap + movement) to
a :class:`PipelineWorker` running on its own thread, defaulting to safe Dry-Run.

Importing this module requires PySide6; callers that want to degrade gracefully when it is
absent (the CLI) should guard the import and surface an install hint.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from minecraft_cv.ui import theme
from minecraft_cv.ui.camera_view import CameraView
from minecraft_cv.ui.panels import HeaderBar, KeymapPanel, MovementPanel
from minecraft_cv.ui.theme import apply_theme
from minecraft_cv.ui.worker import PipelineWorker

if TYPE_CHECKING:
    from minecraft_cv.capture.source import FrameSource
    from minecraft_cv.config import Settings
    from minecraft_cv.runtime import FramePacket


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
    """The framed application: header controls, camera painting, and live HUD sidebar."""

    def __init__(self, settings: Settings, source: FrameSource | None = None) -> None:
        """Assemble the window and auto-start the capture session in Dry-Run."""
        super().__init__()
        self._settings = settings
        self._source = source
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None

        self.setWindowTitle("minecraft_cv")
        self.resize(1180, 720)
        self.setMinimumSize(940, 600)

        root = QWidget()
        root.setObjectName("Root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = HeaderBar()
        outer.addWidget(self._header)

        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(16, 16, 16, 16)
        body_lay.setSpacing(16)

        self._camera = CameraView(settings.tracking.swap_handedness)
        body_lay.addWidget(self._camera, stretch=1)

        sidebar = QWidget()
        sidebar.setFixedWidth(theme.SIDEBAR_WIDTH)
        side_lay = QVBoxLayout(sidebar)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(14)
        self._keymap = KeymapPanel(settings)
        self._movement = MovementPanel()
        side_lay.addWidget(self._keymap)
        side_lay.addWidget(self._movement)
        side_lay.addStretch(1)
        body_lay.addWidget(sidebar)

        outer.addWidget(body, stretch=1)
        self.setCentralWidget(root)

        self._header.startStopClicked.connect(self._toggle_session)
        self._header.liveToggled.connect(self._on_live_toggled)
        self._header.calibrateClicked.connect(self._on_calibrate)
        self._header.pinToggled.connect(self._on_pin)

        self._start_session()

    # --- session lifecycle --------------------------------------------------
    def _start_session(self) -> None:
        if self._thread is not None:
            return
        self._worker = PipelineWorker(self._settings, source=self._source)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error.connect(self._on_error)
        self._worker.started_ok.connect(lambda: self._header.set_running(True))
        self._worker.live_changed.connect(self._header.set_live)
        self._worker.stopped.connect(self._on_worker_stopped)
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
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._header.set_running(False)
        self._header.set_live(False)

    # --- signals from the header --------------------------------------------
    def _on_live_toggled(self, enabled: bool) -> None:
        if self._worker is not None:
            self._worker.request_live(enabled)

    def _on_calibrate(self) -> None:
        if self._worker is not None:
            self._worker.request_recenter()

    def _on_pin(self, pinned: bool) -> None:
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, pinned)
        self.show()  # re-show is required after changing window flags

    # --- signals from the worker --------------------------------------------
    def _on_frame(self, packet: FramePacket) -> None:
        self._camera.set_packet(packet)
        step = packet.step
        self._header.set_status(step.left_status, step.right_status)
        self._keymap.update_state(step)
        self._movement.update_state(step)

    def _on_error(self, message: str) -> None:
        QMessageBox.warning(self, "minecraft_cv", message)

    # --- teardown -----------------------------------------------------------
    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override name
        """Stop the session and release any held input before the window closes."""
        if self._worker is not None:
            self._worker.request_stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        super().closeEvent(event)  # type: ignore[arg-type]


__all__ = ["MainWindow", "run_app"]
