"""Reusable custom-painted widgets for the dark-glass HUD.

Each stateful widget caches its value and only repaints when it changes, so feeding them at
camera frame rate stays cheap (no per-frame stylesheet re-polishing).
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from minecraft_cv.ui import theme

_DARK_ON_ACCENT = "#FFFFFF"


class KeyCap(QWidget):
    """A keyboard-style cap that lights up when its gesture is active."""

    def __init__(
        self, label: str, accent: str = theme.ACCENT, parent: QWidget | None = None
    ) -> None:
        """Create a key-cap.

        Args:
            label: The key text to show (e.g. ``"Space"``, ``"W"``, ``"LMB"``).
            accent: Hex color used when active.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._label = label
        self._accent = accent
        self._active = False
        self._font = QFont()
        self._font.setFamilies(["SF Mono", "JetBrains Mono", "Menlo", "Consolas"])
        self._font.setPointSize(11)
        self._font.setBold(True)
        fm = QFontMetrics(self._font)
        self._w = max(30, fm.horizontalAdvance(label) + 22)
        self.setFixedSize(self._w, 30)

    def setActive(self, active: bool) -> None:
        """Light or dim the cap; repaints only on a state change."""
        if active != self._active:
            self._active = active
            self.update()

    def sizeHint(self) -> QSize:  # noqa: D102
        return QSize(self._w, 30)

    def paintEvent(self, event: object) -> None:  # noqa: D102, ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        radius = theme.RADIUS_CAP
        if self._active:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(self._accent))
            p.drawRoundedRect(r, radius, radius)
            # subtle top highlight
            hi = QRectF(r.left() + 2, r.top() + 2, r.width() - 4, r.height() * 0.45)
            p.setBrush(QColor(255, 255, 255, 38))
            p.drawRoundedRect(hi, radius - 2, radius - 2)
            p.setPen(QColor(_DARK_ON_ACCENT))
        else:
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.setBrush(QColor(theme.BG_ELEV))
            p.drawRoundedRect(r, radius, radius)
            p.setPen(QColor(theme.MUTED))
        p.setFont(self._font)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._label)
        p.end()


class IndicatorDot(QWidget):
    """A small status dot with a soft glow when active."""

    def __init__(self, accent: str = theme.ACCENT, parent: QWidget | None = None) -> None:
        """Create an indicator dot tinted with ``accent`` when active."""
        super().__init__(parent)
        self._accent = accent
        self._active = False
        self.setFixedSize(18, 18)

    def setActive(self, active: bool) -> None:
        """Toggle the dot; repaints only on a state change."""
        if active != self._active:
            self._active = active
            self.update()

    def paintEvent(self, event: object) -> None:  # noqa: D102, ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        if self._active:
            glow = QColor(self._accent)
            glow.setAlpha(60)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(glow)
            p.drawEllipse(QPointF(cx, cy), 9, 9)
            p.setBrush(QColor(self._accent))
            p.drawEllipse(QPointF(cx, cy), 5, 5)
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.IDLE))
            p.drawEllipse(QPointF(cx, cy), 5, 5)
        p.end()


class StatusPill(QWidget):
    """A pill showing the input mode: emerald ``DRY RUN`` or red ``LIVE``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the pill in safe Dry-Run state."""
        super().__init__(parent)
        self._live = False
        self._font = QFont()
        self._font.setFamilies(["SF Mono", "Menlo", "Consolas"])
        self._font.setPointSize(11)
        self._font.setBold(True)
        self.setFixedHeight(26)
        self.setMinimumWidth(96)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def setLive(self, live: bool) -> None:
        """Switch between Live and Dry-Run; repaints on change."""
        if live != self._live:
            self._live = live
            self.update()

    def paintEvent(self, event: object) -> None:  # noqa: D102, ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        accent = QColor(theme.LIVE if self._live else theme.ACCENT)
        text = "LIVE" if self._live else "DRY RUN"
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        radius = r.height() / 2
        bg = QColor(accent)
        bg.setAlpha(36)
        p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 170), 1))
        p.setBrush(bg)
        p.drawRoundedRect(r, radius, radius)
        p.setBrush(accent)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(r.left() + 13, r.center().y()), 4, 4)
        p.setPen(accent)
        p.setFont(self._font)
        p.drawText(
            r.adjusted(24, 0, -8, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            text,
        )
        p.end()


class HealthChip(QWidget):
    """Per-hand tracking-health chip: ``L OK`` / ``R SETTLING`` / ``L NO HAND``."""

    _LABELS = {"normal": "OK", "stabilizing": "SETTLING", "absent": "ABSENT", "tracking": "OK"}
    _COLORS = {"normal": theme.ACCENT, "stabilizing": theme.WARN, "absent": theme.FAINT, "tracking": theme.ACCENT}

    def __init__(self, hand_letter: str, parent: QWidget | None = None) -> None:
        """Create a chip labeled for one hand (``"L"`` or ``"R"``)."""
        super().__init__(parent)
        self._letter = hand_letter
        self._status = "absent"
        self._font = QFont()
        self._font.setFamilies(["SF Mono", "Menlo", "Consolas"])
        self._font.setPointSize(10)
        self._font.setBold(True)
        self.setFixedHeight(24)
        self.setMinimumWidth(96)

    def setStatus(self, status: str) -> None:
        """Update the tracking status (``normal``/``stabilizing``/``absent``)."""
        if status != self._status:
            self._status = status
            self.update()

    def paintEvent(self, event: object) -> None:  # noqa: D102, ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        accent = QColor(self._COLORS.get(self._status, theme.FAINT))
        label = self._LABELS.get(self._status, "—")
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        radius = 7
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(QColor(theme.BG_ELEV))
        p.drawRoundedRect(r, radius, radius)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(accent)
        p.drawEllipse(QPointF(r.left() + 12, r.center().y()), 4, 4)
        p.setFont(self._font)
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(r.left() + 22, r.top(), 16, r.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._letter)
        p.setPen(accent)
        p.drawText(QRectF(r.left() + 38, r.top(), r.width() - 44, r.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
        p.end()


class JoystickGizmo(QWidget):
    """A compact circular gizmo: deadzone ring, neutral, and the live output vector."""

    def __init__(self, accent: str, has_deadzone: bool, parent: QWidget | None = None) -> None:
        """Create a gizmo.

        Args:
            accent: Hex color for the vector + active highlight.
            has_deadzone: Whether to draw the deadzone ring (movement stick only).
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._accent = accent
        self._has_deadzone = has_deadzone
        self._vec = (0.0, 0.0)
        self._deadzone = 0.0
        self.setFixedSize(92, 92)

    def setState(self, vx: float, vy: float, deadzone: float = 0.0) -> None:
        """Set the output vector (clamped for display) and deadzone fraction; repaints."""
        vx = max(-1.0, min(1.0, vx))
        vy = max(-1.0, min(1.0, vy))
        if (vx, vy) != self._vec or deadzone != self._deadzone:
            self._vec = (vx, vy)
            self._deadzone = deadzone
            self.update()

    def paintEvent(self, event: object) -> None:  # noqa: D102, ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        radius = min(cx, cy) - 6
        # track
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(QColor(theme.BG_ELEV))
        p.drawEllipse(QPointF(cx, cy), radius, radius)
        # crosshair
        p.setPen(QPen(QColor(theme.IDLE), 1))
        p.drawLine(QPointF(cx - radius, cy), QPointF(cx + radius, cy))
        p.drawLine(QPointF(cx, cy - radius), QPointF(cx, cy + radius))
        # deadzone ring
        if self._has_deadzone and self._deadzone > 0:
            dz = min(radius, self._deadzone * radius / 0.25)  # scale 0.25 fraction -> full
            pen = QPen(QColor(theme.FAINT), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), dz, dz)
        # vector
        vx, vy = self._vec
        mag = (vx * vx + vy * vy) ** 0.5
        accent = QColor(self._accent)
        ex, ey = cx + vx * radius, cy + vy * radius
        if mag > 1e-3:
            p.setPen(QPen(accent, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(cx, cy), QPointF(ex, ey))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(accent)
            p.drawEllipse(QPointF(ex, ey), 5, 5)
        # neutral hub
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.TEXT))
        p.drawEllipse(QPointF(cx, cy), 3.5, 3.5)
        p.end()


class Card(QFrame):
    """A frosted rounded container with an uppercase title and a vertical content area."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Create a titled card; add child widgets via :meth:`add` or :attr:`body`."""
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(10)
        self._title = QLabel(title)
        self._title.setObjectName("CardTitle")
        outer.addWidget(self._title)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(8)
        outer.addLayout(self.body)

    def add(self, widget: QWidget) -> None:
        """Append a widget to the card body."""
        self.body.addWidget(widget)


__all__ = [
    "Card", "HealthChip", "IndicatorDot", "JoystickGizmo", "KeyCap", "StatusPill",
]
