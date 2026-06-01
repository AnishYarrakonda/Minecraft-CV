"""Dark-glass design system: color tokens, fonts, and the global stylesheet.

A single source of truth for the app's look. Colors are plain hex strings (usable in QSS and
as :class:`QColor` via :func:`color`); :func:`apply_theme` installs the palette, default font,
and stylesheet onto the :class:`QApplication`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# --- Color tokens (hex) ------------------------------------------------------
BG = "#09090B"          # window background (zinc-950)
BG_ELEV = "#111113"     # slightly raised surface
PANEL = "#18181B"       # card body (zinc-900)
PANEL_HI = "#1F1F23"    # hover / elevated card
BORDER = "#27272A"      # hairline border (zinc-800)
BORDER_HI = "#3F3F46"   # brighter border (hover/focus, zinc-700)
TEXT = "#FAFAFA"        # primary text (zinc-50)
MUTED = "#A1A1AA"       # secondary text / inactive labels (zinc-400)
FAINT = "#52525B"       # tertiary text (zinc-600)
ACCENT = "#3B82F6"      # primary accent (blue-500)
MOVE = "#60A5FA"        # left hand / movement (blue-400)
LOOK = "#818CF8"        # right hand / look (indigo-400)
LIVE = "#EF4444"        # Live input (red-500)
WARN = "#F59E0B"        # stabilizing / warning (amber-500)
IDLE = "#27272A"        # inactive indicator fill (zinc-800)

# --- Fonts -------------------------------------------------------------------
UI_FONTS = '"SF Pro Text", "Inter", "Helvetica Neue", "Segoe UI", sans-serif'
MONO_FONTS = '"SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace'

# --- Geometry ----------------------------------------------------------------
RADIUS_CARD = 8
RADIUS_CAP = 6
SIDEBAR_WIDTH = 320


def color(hex_str: str, alpha: int = 255) -> QColor:
    """Return a :class:`QColor` for a token hex string with an optional alpha (0-255)."""
    c = QColor(hex_str)
    c.setAlpha(alpha)
    return c


def _stylesheet() -> str:
    return f"""
    QWidget {{
        background: transparent;
        color: {TEXT};
        font-family: {UI_FONTS};
        font-size: 12px;
    }}
    QMainWindow, #Root {{
        background: {BG};
    }}
    #Card {{
        background: {PANEL};
        border: none;
        border-radius: {RADIUS_CARD}px;
    }}
    #CardTitle {{
        color: {MUTED};
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    #HeaderTitle {{
        color: {TEXT};
        font-size: 14px;
        font-weight: 600;
    }}
    #HeaderBar {{
        background: {BG};
        border-bottom: 1px solid {BORDER};
    }}
    #RowName {{ color: {TEXT}; font-size: 12px; font-weight: 500; }}
    #RowFinger {{ color: {FAINT}; font-size: 11px; }}
    #FpsLabel {{ color: {FAINT}; font-family: {MONO_FONTS}; font-size: 11px; }}
    QPushButton {{
        background: transparent;
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 500;
    }}
    QPushButton:hover {{ background: {PANEL_HI}; border-color: {BORDER_HI}; }}
    QPushButton:pressed {{ background: {BG_ELEV}; }}
    QPushButton:disabled {{ color: {FAINT}; border-color: {BORDER}; }}
    QPushButton#PrimaryButton {{
        background: {ACCENT}; color: #FFFFFF; border: none; font-weight: 600;
    }}
    QPushButton#PrimaryButton:hover {{ background: #2563EB; }}
    QPushButton#PrimaryButton:pressed {{ background: #1D4ED8; }}
    QPushButton#LiveButton[live="true"] {{
        background: {LIVE}; color: #FFFFFF; border: none;
    }}
    QPushButton#LiveButton[live="true"]:hover {{ background: #DC2626; }}
    QPushButton#PinButton[pinned="true"] {{
        background: transparent; color: {ACCENT}; border-color: {ACCENT};
    }}
    QToolTip {{
        background: {PANEL}; color: {TEXT};
        border: 1px solid {BORDER_HI}; border-radius: 6px; padding: 4px 8px;
        font-size: 12px;
    }}
    """


def apply_theme(app: QApplication) -> None:
    """Install the dark palette, default font, and global stylesheet onto ``app``."""
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, color(BG))
    pal.setColor(QPalette.ColorRole.WindowText, color(TEXT))
    pal.setColor(QPalette.ColorRole.Base, color(PANEL))
    pal.setColor(QPalette.ColorRole.AlternateBase, color(BG_ELEV))
    pal.setColor(QPalette.ColorRole.Text, color(TEXT))
    pal.setColor(QPalette.ColorRole.Button, color(PANEL))
    pal.setColor(QPalette.ColorRole.ButtonText, color(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, color(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, color("#0A0E0C"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, color(BG_ELEV))
    pal.setColor(QPalette.ColorRole.ToolTipText, color(TEXT))
    app.setPalette(pal)

    font = QFont()
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setPointSize(12)
    app.setFont(font)

    app.setStyleSheet(_stylesheet())
    # Keep widget backgrounds crisp on top of the dark window.
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, True)


__all__ = [
    "ACCENT", "BG", "BG_ELEV", "BORDER", "BORDER_HI", "FAINT", "IDLE", "LIVE", "LOOK",
    "MONO_FONTS", "MOVE", "MUTED", "PANEL", "PANEL_HI", "RADIUS_CAP", "RADIUS_CARD",
    "SIDEBAR_WIDTH", "TEXT", "UI_FONTS", "WARN", "apply_theme", "color",
]
