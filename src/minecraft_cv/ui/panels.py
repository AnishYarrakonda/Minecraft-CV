"""Sidebar panels: header controls, the live keymap, and movement/look status.

These read a :class:`~minecraft_cv.pipeline.StepResult` each frame and update lightweight
custom widgets; nothing here touches the OS or the pipeline directly (the window wires the
header's signals to the worker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from minecraft_cv.ui import theme
from minecraft_cv.ui.keymap import build_keymap
from minecraft_cv.ui.widgets import (
    Card,
    HealthChip,
    IndicatorDot,
    JoystickGizmo,
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
        lay.addWidget(self._chip_l)
        lay.addWidget(self._chip_r)

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

    def set_status(self, left: str, right: str) -> None:
        """Update both per-hand tracking-health chips."""
        self._chip_l.setStatus(left)
        self._chip_r.setStatus(right)

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
        lay.setSpacing(10)
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

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the keymap from ``settings`` (bindings + gesture config)."""
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        self._rows: dict[tuple[str, str], _KeymapRow] = {}

        cards = {
            "left": Card("LEFT HAND  ·  ACTIONS"),
            "right": Card("RIGHT HAND  ·  COMBAT"),
        }
        accents = {"left": theme.MOVE, "right": theme.LOOK}
        for row in build_keymap(settings):
            widget = _KeymapRow(row.name, row.finger, row.key, accents[row.hand])
            self._rows[(row.hand, row.gesture)] = widget
            cards[row.hand].add(widget)
        lay.addWidget(cards["left"])
        lay.addWidget(cards["right"])

    def update_state(self, step: StepResult) -> None:
        """Light each row whose gesture is currently held."""
        for (hand, gesture), widget in self._rows.items():
            held = step.left_gestures if hand == "left" else step.right_gestures
            widget.set_active(gesture in held)


class MovementPanel(QWidget):
    """WASD key cluster + move/look gizmos."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the movement & look status card."""
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        card = Card("MOVEMENT  ·  LOOK")
        lay.addWidget(card)

        # Move row: WASD cross + movement gizmo.
        move_row = QHBoxLayout()
        move_row.setSpacing(12)
        self._caps = {k: KeyCap(k.upper(), theme.MOVE) for k in ("w", "a", "s", "d")}
        cross = QGridLayout()
        cross.setSpacing(4)
        cross.addWidget(self._caps["w"], 0, 1)
        cross.addWidget(self._caps["a"], 1, 0)
        cross.addWidget(self._caps["s"], 1, 1)
        cross.addWidget(self._caps["d"], 1, 2)
        cross_w = QWidget()
        cross_w.setLayout(cross)
        move_row.addWidget(cross_w)
        move_row.addStretch(1)
        self._move_gizmo = JoystickGizmo(theme.MOVE, has_deadzone=True)
        move_row.addWidget(self._move_gizmo)
        card.body.addLayout(move_row)

        # Look row.
        look_row = QHBoxLayout()
        look_lbl = QLabel("Mouse look")
        look_lbl.setObjectName("RowName")
        look_row.addWidget(look_lbl)
        look_row.addStretch(1)
        self._look_gizmo = JoystickGizmo(theme.LOOK, has_deadzone=False)
        look_row.addWidget(self._look_gizmo)
        card.body.addLayout(look_row)

    def update_state(self, step: StepResult) -> None:
        """Light the WASD caps and update both gizmos from the latest step."""
        for letter, cap in self._caps.items():
            cap.setActive(letter in step.wasd_held)
        lo = step.left_output
        self._move_gizmo.setState(float(lo[0]), float(lo[1]), step.deadzone)
        ro = step.right_output
        self._look_gizmo.setState(float(ro[0]) * 0.25, float(ro[1]) * 0.25)


__all__ = ["HeaderBar", "KeymapPanel", "MovementPanel"]
