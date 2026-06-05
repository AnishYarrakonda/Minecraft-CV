"""Sidebar panels: header controls and the live keymap.

These read a :class:`~minecraft_cv.pipeline.StepResult` each frame and update lightweight
custom widgets; nothing here touches the OS or the pipeline directly (the window wires the
header's signals to the worker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QSlider,
)

from minecraft_cv.ui import theme
from minecraft_cv.ui.keymap import build_keymap
from minecraft_cv.ui.widgets import (
    Card,
    HealthChip,
    IndicatorDot,
    KeyCap,
    StatusPill,
)

if TYPE_CHECKING:
    from minecraft_cv.config import Settings
    from minecraft_cv.pipeline import StepResult


def _restyle(widget: QWidget) -> None:
    """Re-apply QSS after a dynamic property change."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


class HeaderBar(QWidget):
    """Top bar: title, status pill, per-hand health chips, and the control buttons."""

    startStopClicked = Signal()
    liveToggled = Signal(bool)
    calibrateClicked = Signal()
    pinToggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the header bar in stopped, Dry-Run, unpinned state."""
        super().__init__(parent)
        self.setObjectName("HeaderBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(60)
        self._running = False
        self._live = False
        self._pinned = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 0, 14, 0)
        lay.setSpacing(12)

        title = QLabel(f'<span style="color:{theme.ACCENT}">●</span> minecraft_cv')
        title.setObjectName("HeaderTitle")
        title.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(title)
        lay.addSpacing(8)

        self._pill = StatusPill()
        lay.addWidget(self._pill)
        self._chip_l = HealthChip("L")
        self._chip_r = HealthChip("R")
        self._chip_f = HealthChip("F")
        lay.addWidget(self._chip_l)
        lay.addWidget(self._chip_r)
        lay.addWidget(self._chip_f)

        lay.addStretch(1)

        self._start_btn = QPushButton("Start")
        self._start_btn.setObjectName("PrimaryButton")
        self._start_btn.clicked.connect(self.startStopClicked.emit)
        self._live_btn = QPushButton("Go Live")
        self._live_btn.setObjectName("LiveButton")
        self._live_btn.clicked.connect(self._on_live_clicked)
        self._cal_btn = QPushButton("Calibrate")
        self._cal_btn.clicked.connect(self.calibrateClicked.emit)
        self._pin_btn = QPushButton("Pin")
        self._pin_btn.setObjectName("PinButton")
        self._pin_btn.setToolTip("Keep the window on top while you play")
        self._pin_btn.clicked.connect(self._on_pin_clicked)
        for b in (self._start_btn, self._live_btn, self._cal_btn, self._pin_btn):
            lay.addWidget(b)
        self._sync_buttons()

    def _on_live_clicked(self) -> None:
        self.liveToggled.emit(not self._live)

    def _on_pin_clicked(self) -> None:
        self._pinned = not self._pinned
        self._pin_btn.setProperty("pinned", "true" if self._pinned else "false")
        _restyle(self._pin_btn)
        self.pinToggled.emit(self._pinned)

    def set_running(self, running: bool) -> None:
        """Reflect whether the capture session is active."""
        self._running = running
        self._sync_buttons()

    def set_live(self, live: bool) -> None:
        """Reflect the Live/Dry-Run input mode."""
        self._live = live
        self._pill.setLive(live)
        self._sync_buttons()

    def set_status(self, left: str, right: str, face: str = "absent") -> None:
        """Update tracking-health chips."""
        self._chip_l.setStatus(left)
        self._chip_r.setStatus(right)
        self._chip_f.setStatus(face)

    def _sync_buttons(self) -> None:
        self._start_btn.setText("Stop" if self._running else "Start")
        self._live_btn.setText("Dry-Run" if self._live else "Go Live")
        self._live_btn.setProperty("live", "true" if self._live else "false")
        self._live_btn.setEnabled(self._running)
        self._cal_btn.setEnabled(self._running)
        _restyle(self._live_btn)


class _KeymapRow(QWidget):
    """One keymap row: indicator dot, name + finger, and the key-cap."""

    def __init__(self, name: str, finger: str, key: str, accent: str) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.dot = IndicatorDot(accent)
        lay.addWidget(self.dot)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(0)
        name_lbl = QLabel(name)
        name_lbl.setObjectName("RowName")
        text.addWidget(name_lbl)
        if finger:
            finger_lbl = QLabel(f"{finger} pinch")
            finger_lbl.setObjectName("RowFinger")
            text.addWidget(finger_lbl)
        lay.addLayout(text)
        lay.addStretch(1)
        self.cap = KeyCap(key, accent)
        lay.addWidget(self.cap)

    def set_active(self, active: bool) -> None:
        self.dot.setActive(active)
        self.cap.setActive(active)


class KeymapPanel(QWidget):
    """Two cards listing every bound gesture, lit live as gestures fire."""

    sensitivityChanged = Signal(float)
    sneakSensitivityChanged = Signal(int)

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the keymap from ``settings`` (bindings + gesture config)."""
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        # Transparent background and a custom scrollbar so it doesn't overlap content on macOS
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; }
            QWidget#KeymapScrollContent { background: transparent; }
            QScrollBar:vertical {
                border: none;
                background: rgba(255, 255, 255, 0.05);
                width: 6px;
                border-radius: 3px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                min-height: 16px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                border: none;
                background: none;
            }
        """)

        content = QWidget()
        content.setObjectName("KeymapScrollContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(10)

        self._rows: dict[tuple[str, str], _KeymapRow] = {}

        cards = {
            "left": Card("LEFT HAND  ·  ACTIONS"),
            "right": Card("RIGHT HAND  ·  COMBAT"),
            "face": Card("FACE GESTURES"),
        }
        accents = {"left": theme.MOVE, "right": theme.LOOK, "face": theme.ACCENT}
        for row in build_keymap(settings):
            widget = _KeymapRow(row.name, row.finger, row.key, accents[row.hand])
            self._rows[(row.hand, row.gesture)] = widget
            cards[row.hand].add(widget)
        lay.addWidget(cards["left"])
        lay.addWidget(cards["right"])
        lay.addWidget(cards["face"])
        
        # Sensitivity slider
        self._base_sensitivity = settings.joystick.right_sensitivity
        sens_card = Card("MOUSE SENSITIVITY")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider_label = QLabel("1.00x")
        self.slider_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sens_card.add(self.slider)
        sens_card.add(self.slider_label)
        
        # Sneak Sensitivity slider
        self.sneak_slider = QSlider(Qt.Orientation.Horizontal)
        self.sneak_slider.setMinimum(1)
        self.sneak_slider.setMaximum(100)
        
        # Calculate initial value from config: engage = 0.50 + (val/100) * 0.49
        # => val = (engage - 0.50) / 0.49 * 100
        initial_engage = settings.gestures.head_pitch.engage_ratio if settings.gestures.head_pitch else 0.85
        initial_val = int(round((initial_engage - 0.50) / 0.49 * 100.0))
        initial_val = max(1, min(100, initial_val))
        
        self.sneak_slider.setValue(initial_val)
        self.sneak_slider.valueChanged.connect(self._on_sneak_slider)
        self.sneak_slider_label = QLabel(f"{initial_val}%")
        self.sneak_slider_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sens_card.add(QLabel("SNEAK SENSITIVITY"))
        sens_card.add(self.sneak_slider)
        sens_card.add(self.sneak_slider_label)
        
        lay.addWidget(sens_card)

        lay.addStretch(1)

        scroll.setWidget(content)
        main_lay.addWidget(scroll)

    def _on_slider(self, val: int) -> None:
        mult = val / 10.0
        self.slider_label.setText(f"{mult:.2f}x")
        self.sensitivityChanged.emit(self._base_sensitivity * mult)

    def _on_sneak_slider(self, val: int) -> None:
        self.sneak_slider_label.setText(f"{val}%")
        self.sneakSensitivityChanged.emit(val)

    def update_state(self, step: StepResult) -> None:
        """Light each row whose gesture is currently held."""
        for (hand, gesture), widget in self._rows.items():
            if hand == "left":
                held = step.left_gestures
            elif hand == "right":
                held = step.right_gestures
            else:
                held = step.face_gestures
            widget.set_active(gesture in held)


__all__ = ["HeaderBar", "KeymapPanel"]
