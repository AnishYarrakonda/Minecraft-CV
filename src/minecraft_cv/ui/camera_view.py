"""The camera "painting": a clean live feed with a subtle glowing hand skeleton.

No HUD panels are drawn over the feed (those live in the sidebar); only the hand skeleton,
rounded framing, and a small unobtrusive mode/FPS badge appear here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from minecraft_cv.ui import theme
from minecraft_cv.ui.skeleton import (
    FACE_COLOR,
    FACE_CONNECTIONS,
    FACE_KEY_LANDMARKS,
    FINGERTIPS,
    HAND_CONNECTIONS,
    fit_rect,
    to_widget,
)

if TYPE_CHECKING:
    from minecraft_cv.runtime import FramePacket

# Glow strokes: (pen width, alpha) drawn back-to-front for a soft neon edge.
_GLOW_PASSES = ((9.0, 38), (5.0, 110), (2.4, 255))
# Subtler glow for the face overlay — thinner, lower alpha to stay secondary.
_FACE_GLOW_PASSES = ((6.0, 22), (3.0, 70), (1.4, 180))
_IMG_RADIUS = 18


class CameraView(QWidget):
    """Renders the latest frame letterboxed, with a glowing skeleton over detected hands."""

    def __init__(self, swap_handedness: bool, parent: QWidget | None = None) -> None:
        """Create the view.

        Args:
            swap_handedness: Mirror MediaPipe's L/R labels (matches the pipeline) so the user's
                physical left hand is tinted as the movement hand.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._swap = swap_handedness
        self._pixmap: QPixmap | None = None
        self._frame_ref: object | None = None  # keep numpy buffer alive behind the QImage
        self._hands: list[object] = []
        self._face: object | None = None
        self._live = False
        self._fps = 0.0
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_packet(self, packet: FramePacket) -> None:
        """Store the newest processed frame + hands and schedule a repaint."""
        frame = packet.frame
        h, w = frame.shape[:2]
        img = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888)
        self._pixmap = QPixmap.fromImage(img)
        self._frame_ref = frame
        self._hands = list(packet.hands)
        self._face = packet.face
        self._live = packet.live
        self._fps = packet.fps
        self.update()

    def _hand_color(self, handedness: str) -> str:
        label = handedness
        if self._swap:
            label = "Right" if label == "Left" else "Left"
        return theme.MOVE if label == "Left" else theme.LOOK

    def paintEvent(self, event: object) -> None:  # noqa: D102, ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.BG))

        if self._pixmap is None:
            self._paint_placeholder(p)
            p.end()
            return

        pm = self._pixmap
        rect = fit_rect(pm.width(), pm.height(), self.width(), self.height())
        x, y, w, h = rect
        img_rect = QRectF(x, y, w, h)

        # Rounded framing for the feed.
        clip = QPainterPath()
        clip.addRoundedRect(img_rect, _IMG_RADIUS, _IMG_RADIUS)
        p.save()
        p.setClipPath(clip)
        p.drawPixmap(img_rect, pm, QRectF(pm.rect()))
        self._paint_face(p, rect)
        self._paint_skeleton(p, rect)
        p.restore()

        # Hairline frame edge.
        p.setPen(QPen(QColor(theme.BORDER_HI), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(img_rect.adjusted(0.5, 0.5, -0.5, -0.5), _IMG_RADIUS, _IMG_RADIUS)

        self._paint_badges(p, img_rect)
        p.end()

    def _paint_face(self, p: QPainter, rect: tuple[float, float, float, float]) -> None:
        face = self._face
        if face is None or face.landmarks is None:  # type: ignore[attr-defined]
            return
        lms = face.landmarks  # type: ignore[attr-defined]
        if len(lms) < max(FACE_KEY_LANDMARKS) + 1:
            return

        pts = {i: to_widget(float(lms[i][0]), float(lms[i][1]), rect) for i in FACE_KEY_LANDMARKS}
        base = QColor(FACE_COLOR)

        # Glow connections.
        for width, alpha in _FACE_GLOW_PASSES:
            col = QColor(base)
            col.setAlpha(alpha)
            pen = QPen(col, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            for a, b in FACE_CONNECTIONS:
                ax, ay = pts[a]
                bx, by = pts[b]
                p.drawLine(QPointF(ax, ay), QPointF(bx, by))

        # Key-point dots.
        p.setPen(Qt.PenStyle.NoPen)
        for px, py in pts.values():
            glow = QColor(base)
            glow.setAlpha(50)
            p.setBrush(glow)
            p.drawEllipse(QPointF(px, py), 4.0, 4.0)
            p.setBrush(base)
            p.drawEllipse(QPointF(px, py), 1.8, 1.8)

    def _paint_skeleton(self, p: QPainter, rect: tuple[float, float, float, float]) -> None:
        for hand in self._hands:
            lms = hand.landmarks  # type: ignore[attr-defined]
            pts = [to_widget(float(lm[0]), float(lm[1]), rect) for lm in lms]
            base = QColor(self._hand_color(hand.handedness))  # type: ignore[attr-defined]

            for width, alpha in _GLOW_PASSES:
                col = QColor(base)
                col.setAlpha(alpha)
                pen = QPen(col, width)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                for a, b in HAND_CONNECTIONS:
                    ax, ay = pts[a]
                    bx, by = pts[b]
                    p.drawLine(QPointF(ax, ay), QPointF(bx, by))

            p.setPen(Qt.PenStyle.NoPen)
            for i, (px, py) in enumerate(pts):
                tip = i in FINGERTIPS
                glow = QColor(base)
                glow.setAlpha(70)
                p.setBrush(glow)
                p.drawEllipse(QPointF(px, py), 8 if tip else 6, 8 if tip else 6)
                p.setBrush(QColor("#FFFFFF") if tip else base)
                p.drawEllipse(QPointF(px, py), 3.4 if tip else 2.6, 3.4 if tip else 2.6)

    def _paint_badges(self, p: QPainter, img_rect: QRectF) -> None:
        accent = QColor(theme.LIVE if self._live else theme.ACCENT)
        text = "LIVE" if self._live else "DRY RUN"
        font = QFont()
        font.setFamilies(["SF Mono", "Menlo", "Consolas"])
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)

        # Mode badge, bottom-left.
        pill = QRectF(img_rect.left() + 12, img_rect.bottom() - 32, 84, 20)
        bg = QColor(0, 0, 0, 130)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(pill, 10, 10)
        p.setBrush(accent)
        p.drawEllipse(QPointF(pill.left() + 11, pill.center().y()), 3, 3)
        p.setPen(accent)
        p.drawText(pill.adjusted(20, 0, -5, 0),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

        # FPS, bottom-right.
        fps_rect = QRectF(img_rect.right() - 88, img_rect.bottom() - 32, 74, 20)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 130))
        p.drawRoundedRect(fps_rect, 10, 10)
        p.setPen(QColor(theme.TEXT))
        p.drawText(fps_rect.adjusted(10, 0, -10, 0),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   f"{self._fps:4.0f} FPS")

    def _paint_placeholder(self, p: QPainter) -> None:
        p.setPen(QColor(theme.MUTED))
        font = QFont()
        font.setPointSize(15)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Camera starting…")


__all__ = ["CameraView"]
