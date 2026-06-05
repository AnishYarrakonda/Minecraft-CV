"""Compact always-on-top overlay window: camera feed + gesture toast notifications.

Launched via ``mcv overlay``.  The full desktop app (``mcv ui``) is untouched; this is an
additive second launch mode that reuses :class:`CameraView` and :class:`PipelineWorker` verbatim.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QPoint,
    QPauseAnimation,
    QPropertyAnimation,
    QSequentialAnimationGroup,
    Qt,
)
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)

from minecraft_cv.runtime import FrameProcessor
from minecraft_cv.ui import theme
from minecraft_cv.ui.camera_view import CameraView
from minecraft_cv.ui.macos_window import keep_window_in_front
from minecraft_cv.ui.worker import PipelineWorker

if TYPE_CHECKING:
    from minecraft_cv.config import Settings
    from minecraft_cv.runtime import FramePacket

_GESTURE_LABELS: dict[str, str] = {
    "attack": "ATTACK",
    "use": "USE",
    "jump": "JUMP",
    "sneak": "SNEAK",
    "sprint": "SPRINT",
    "inventory": "INV",
    "throw_item": "THROW",
    "switch_offhand": "OFFHAND",
    "hotbar_next": "HOT+",
    "hotbar_prev": "HOT-",
}

_TOAST_H = 36  # fixed toast height (px); width adapts to text
_TOAST_FONT_SIZE = 11


def _gesture_label(g: str) -> str:
    return _GESTURE_LABELS.get(g, g.upper())


class _Toast(QWidget):
    """Fade-in → hold → fade-out pill badge drawn over the bottom-left of the camera feed."""

    _PADDING_H = 16
    _CORNER_R = 10

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedHeight(_TOAST_H)
        self._text = ""

        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)

        fade_in = QPropertyAnimation(self._effect, b"opacity", self)
        fade_in.setDuration(100)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)

        hold = QPauseAnimation(600, self)

        fade_out = QPropertyAnimation(self._effect, b"opacity", self)
        fade_out.setDuration(250)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)

        self._seq = QSequentialAnimationGroup(self)
        self._seq.addAnimation(fade_in)
        self._seq.addAnimation(hold)
        self._seq.addAnimation(fade_out)
        self._seq.finished.connect(self.hide)

        self.hide()

    def show_text(self, text: str) -> None:
        """Show the badge with *text*, restarting any in-progress animation."""
        self._text = text
        font = QFont()
        font.setBold(True)
        font.setPointSize(_TOAST_FONT_SIZE)
        w = QFontMetrics(font).horizontalAdvance(text) + self._PADDING_H * 2
        self.setFixedWidth(max(w, 60))
        self._seq.stop()
        self._effect.setOpacity(0.0)
        self.show()
        self.raise_()
        self.update()
        self._seq.start()

    def paintEvent(self, event: object) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor(theme.PANEL)
        bg.setAlpha(210)
        p.setBrush(bg)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), self._CORNER_R, self._CORNER_R)
        font = QFont()
        font.setBold(True)
        font.setPointSize(_TOAST_FONT_SIZE)
        p.setFont(font)
        p.setPen(QColor(theme.TEXT))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)


class _CloseButton(QPushButton):
    """Small × dismiss button: semi-transparent at rest, opaque on hover."""

    _SZ = 26

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFixedSize(self._SZ, self._SZ)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._eff = QGraphicsOpacityEffect(self)
        self._eff.setOpacity(0.45)
        self.setGraphicsEffect(self._eff)

    def enterEvent(self, event: object) -> None:  # noqa: ARG002
        self._eff.setOpacity(1.0)

    def leaveEvent(self, event: object) -> None:  # noqa: ARG002
        self._eff.setOpacity(0.45)

    def paintEvent(self, event: object) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor(theme.PANEL)
        bg.setAlpha(200)
        p.setBrush(bg)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(theme.MUTED))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "×")


class OverlayWindow(QWidget):
    """Frameless, always-on-top overlay: camera feed with gesture toast notifications.

    Controls:
    - Drag anywhere on the camera to move the window.
    - Drag the bottom-right grip to resize (minimum 320×240).
    - Click × (top-right) to close.
    - Right-click for context menu (toggle live input / open full app / close).
    """

    def __init__(self, settings: Settings) -> None:
        """Create the overlay and auto-start the pipeline in the current input mode.

        Args:
            settings: Loaded configuration.  ``settings.input.enabled`` sets the initial mode.
        """
        super().__init__(None)
        self._settings = settings
        self._thread: threading.Thread | None = None
        self._worker: PipelineWorker | None = None
        self._prev_gestures: frozenset[str] = frozenset()
        self._drag_offset: QPoint | None = None
        self._live = settings.input.enabled
        self._native_pinned = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        # Keep this Tool window visible when the app is deactivated (e.g. Minecraft focused);
        # complements the native hidesOnDeactivate=False applied in showEvent.
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
        self.setMinimumSize(320, 240)
        self.resize(400, 300)

        # CameraView fills the window via a layout.
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._camera = CameraView(settings.tracking.swap_handedness)
        # Transparent for mouse so all clicks fall through to OverlayWindow for drag/context-menu.
        self._camera.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        lay.addWidget(self._camera)

        # Overlaid widgets positioned manually in resizeEvent (not in the layout).
        self._close_btn = _CloseButton(self)
        self._close_btn.clicked.connect(self.close)

        self._toast = _Toast(self)

        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(20, 20)

        # Ensure overlays render above CameraView.
        self._close_btn.raise_()
        self._grip.raise_()
        self._toast.raise_()

        self._start_session()

    # --- session lifecycle ---------------------------------------------------

    def _start_session(self) -> None:
        self._worker = PipelineWorker(self._settings)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error.connect(self._on_error)
        self._worker.live_changed.connect(self._on_live_changed)
        # Use a Python thread, not QThread — same reason as MainWindow (SIGTRAP on macOS).
        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _on_frame(self, packet: FramePacket) -> None:
        self._camera.set_packet(packet)
        step = packet.step
        cur: frozenset[str] = step.left_gestures | step.right_gestures | step.face_gestures
        new_fires = cur - self._prev_gestures
        self._prev_gestures = cur
        if new_fires:
            label = " + ".join(_gesture_label(g) for g in sorted(new_fires))
            self._toast.show_text(label)
            # Re-anchor toast now that its width may have changed.
            self._reposition_overlays()

    def _on_error(self, message: str) -> None:
        QMessageBox.warning(self, "minecraft_cv overlay", message)

    def _on_live_changed(self, enabled: bool) -> None:
        self._live = enabled

    # --- window drag (no title bar) ------------------------------------------

    def mousePressEvent(self, event: object) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
            self._drag_offset = (
                event.globalPosition().toPoint()  # type: ignore[attr-defined]
                - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event: object) -> None:  # type: ignore[override]
        if (
            event.buttons() & Qt.MouseButton.LeftButton  # type: ignore[attr-defined]
            and self._drag_offset is not None
        ):
            self.move(
                event.globalPosition().toPoint()  # type: ignore[attr-defined]
                - self._drag_offset
            )

    def mouseReleaseEvent(self, event: object) -> None:  # type: ignore[override]  # noqa: ARG002
        self._drag_offset = None

    # --- native window pinning -----------------------------------------------

    def showEvent(self, event: object) -> None:  # type: ignore[override]
        """Pin the native window in front once it is realized (macOS); harmless elsewhere."""
        super().showEvent(event)  # type: ignore[arg-type]
        if not self._native_pinned:
            keep_window_in_front(self)
            self._native_pinned = True

    # --- context menu --------------------------------------------------------

    def contextMenuEvent(self, event: object) -> None:  # type: ignore[override]
        menu = QMenu(self)
        live_label = "Disable Live Input" if self._live else "Enable Live Input"
        toggle_action = menu.addAction(live_label)
        menu.addSeparator()
        full_app_action = menu.addAction("Open Full App")
        menu.addSeparator()
        close_action = menu.addAction("Close Overlay")

        chosen = menu.exec(event.globalPos())  # type: ignore[attr-defined]
        if chosen is close_action:
            self.close()
        elif chosen is full_app_action:
            subprocess.Popen(
                [sys.executable, "-m", "minecraft_cv.cli", "ui"],
                start_new_session=True,
            )
        elif chosen is toggle_action:
            self._toggle_live()

    def _toggle_live(self) -> None:
        if self._worker is None:
            return
        want = not self._live
        emitter = None
        if want:
            try:
                emitter = FrameProcessor.build_live_emitter(self._settings)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "minecraft_cv",
                    "Could not enable Live input. Grant Accessibility / Input Monitoring "
                    "to your terminal in System Settings → Privacy & Security, then retry."
                    f"\n\n{exc}",
                )
                return
        self._worker.request_live(want, emitter)

    # --- layout of overlaid children -----------------------------------------

    def resizeEvent(self, event: object) -> None:  # type: ignore[override]
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._reposition_overlays()

    def _reposition_overlays(self) -> None:
        margin = 8
        btn_sz = self._close_btn.size()
        self._close_btn.move(self.width() - btn_sz.width() - margin, margin)
        grip_sz = self._grip.size()
        self._grip.move(self.width() - grip_sz.width(), self.height() - grip_sz.height())
        toast_margin = 14
        self._toast.move(
            toast_margin,
            self.height() - self._toast.height() - toast_margin,
        )

    # --- teardown ------------------------------------------------------------

    def closeEvent(self, event: object) -> None:  # type: ignore[override]
        """Stop the session and release all held input before closing."""
        if self._worker is not None:
            self._worker.request_stop()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._worker = None
        super().closeEvent(event)  # type: ignore[arg-type]


def run_overlay(settings: Settings) -> int:
    """Launch the overlay window and run the Qt event loop until it closes.

    Args:
        settings: Loaded configuration.

    Returns:
        Process exit code (0 on clean close).
    """
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from minecraft_cv.ui.theme import apply_theme
    apply_theme(app)
    win = OverlayWindow(settings)
    win.show()
    return int(app.exec())


__all__ = ["OverlayWindow", "run_overlay"]
