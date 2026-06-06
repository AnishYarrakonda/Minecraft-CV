"""Panels: header controls and the live key grid (shown under the camera).

These read a :class:`~minecraft_cv.pipeline.StepResult` each frame and update lightweight
custom widgets; nothing here touches the OS or the pipeline directly (the window wires the
header's signals to the worker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from minecraft_cv.ui import theme
from minecraft_cv.ui.keymap import KeyRow, build_keymap
from minecraft_cv.ui.widgets import (
    FlowLayout,
    HealthChip,
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
    """Compact control bar (status pill, per-hand chips, action buttons) shown at the bottom.

    Lays out with a :class:`FlowLayout` so the chips + buttons wrap to extra rows instead of
    squishing/truncating in a narrow window. (Named ``HeaderBar`` for history; it now lives at
    the bottom of the window, below the camera and key grid.)
    """

    startStopClicked = Signal()
    liveToggled = Signal(bool)
    calibrateClicked = Signal()
    pinToggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the control bar in stopped, Dry-Run, unpinned state."""
        super().__init__(parent)
        self.setObjectName("HeaderBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Let the wrapped (multi-row) height propagate to the parent layout.
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        self.setSizePolicy(policy)
        self._running = False
        self._live = False
        self._pinned = False

        lay = FlowLayout(self, margin=0, h_spacing=8, v_spacing=8)

        self._pill = StatusPill()
        lay.addWidget(self._pill)
        self._chip_l = HealthChip("L")
        self._chip_r = HealthChip("R")
        self._chip_f = HealthChip("F")
        lay.addWidget(self._chip_l)
        lay.addWidget(self._chip_r)
        lay.addWidget(self._chip_f)

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


class KeymapPanel(QWidget):
    """A compact key-cap grid (grouped MOVE / COMBAT / FACE) that lights live as gestures fire.

    Lives directly under the camera in the vertical layout. Caps wrap via :class:`FlowLayout`
    so they stay readable in a narrow window; the mouse-sensitivity slider sits below the grid.
    """

    sensitivityChanged = Signal(float)

    _SECTION_TITLES = {"left": "MOVE", "right": "COMBAT", "face": "FACE"}
    _SECTION_ORDER = ("left", "right", "face")

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the keymap from ``settings`` (bindings + gesture config)."""
        super().__init__(parent)
        # Preferred (not Expanding) vertically with height-for-width: take the real wrapped
        # content height when the window is tall, and scroll (via the inner QScrollArea) when
        # squeezed shorter. heightForWidth avoids FlowLayout's sizeHint over-estimating the
        # height (it otherwise assumes every cap stacks in its own row).
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
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
        self._content = content
        lay = QVBoxLayout(content)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        self._caps: dict[tuple[str, str], KeyCap] = {}
        accents = {"left": theme.MOVE, "right": theme.LOOK, "face": theme.ACCENT}

        grouped: dict[str, list[KeyRow]] = {hand: [] for hand in self._SECTION_ORDER}
        for row in build_keymap(settings):
            grouped.setdefault(row.hand, []).append(row)

        for hand in self._SECTION_ORDER:
            rows = grouped.get(hand) or []
            if not rows:
                continue
            label = QLabel(self._SECTION_TITLES.get(hand, hand.upper()))
            label.setObjectName("CardTitle")
            lay.addWidget(label)
            lay.addWidget(self._build_flow(rows, accents[hand]))

        # Mouse sensitivity, directly under the controls.
        self._base_sensitivity = settings.joystick.right_sensitivity
        sens_label = QLabel("MOUSE SENSITIVITY")
        sens_label.setObjectName("CardTitle")
        lay.addSpacing(4)
        lay.addWidget(sens_label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self.slider)
        self.slider_label = QLabel("1.00x")
        self.slider_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.slider_label)

        lay.addStretch(1)

        scroll.setWidget(content)
        main_lay.addWidget(scroll)

    def _build_flow(self, rows: list[KeyRow], accent: str) -> QWidget:
        """Return a host widget holding the section's key-caps in a wrapping flow layout.

        Each cap shows its key label and lights with ``accent`` when its gesture is held; the
        gesture name + finger is exposed as a tooltip so the compact grid stays uncluttered.
        """
        host = QWidget()
        # Let the wrapping height propagate to the parent QVBoxLayout.
        policy = host.sizePolicy()
        policy.setHeightForWidth(True)
        policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        host.setSizePolicy(policy)
        flow = FlowLayout(host, margin=0, h_spacing=6, v_spacing=6)
        for row in rows:
            cap = KeyCap(row.key, accent)
            tip = row.name + (f"  ·  {row.finger}" if row.finger else "")
            cap.setToolTip(tip)
            self._caps[(row.hand, row.gesture)] = cap
            flow.addWidget(cap)
        return host

    def _on_slider(self, val: int) -> None:
        mult = val / 10.0
        self.slider_label.setText(f"{mult:.2f}x")
        self.sensitivityChanged.emit(self._base_sensitivity * mult)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt override name
        """Height depends on width: caps wrap, so the parent must use the wrapped height."""
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt override name
        """Return the grid's height at ``width`` px (caps laid out, typically 1 row per section)."""
        h = self._content.heightForWidth(width)
        return h if h >= 0 else self._content.sizeHint().height()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override name
        """Preferred size; height is the wrapped content height at the current width."""
        w = self.width() if self.width() > 0 else 320
        return QSize(self._content.sizeHint().width(), self.heightForWidth(w))

    def update_state(self, step: StepResult) -> None:
        """Light each key-cap whose gesture is currently held."""
        for (hand, gesture), cap in self._caps.items():
            if hand == "left":
                held = step.left_gestures
            elif hand == "right":
                held = step.right_gestures
            else:
                held = step.face_gestures
            cap.setActive(gesture in held)


__all__ = ["HeaderBar", "KeymapPanel"]
