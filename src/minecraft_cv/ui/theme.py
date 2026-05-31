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
BG = "#0E1116"          # window background (near-black charcoal)
BG_ELEV = "#141921"     # slightly raised surface
PANEL = "#171C24"       # frosted-glass card body
PANEL_HI = "#1C222C"    # hover / elevated card
BORDER = "#262E39"      # hairline border
BORDER_HI = "#333D4A"   # brighter border (hover/focus)
TEXT = "#E6EAF0"        # primary text
MUTED = "#8A94A2"       # secondary text / inactive labels
FAINT = "#5A6470"       # tertiary text
ACCENT = "#2BD576"      # primary accent (emerald) — Dry-Run / OK / engaged
MOVE = "#56DAFF"        # left hand / movement (cyan)
LOOK = "#A78BFA"        # right hand / look (violet)
LIVE = "#FF5C5C"        # Live input (red)
WARN = "#FFB454"        # stabilizing / warning (amber)
IDLE = "#2A313B"        # inactive indicator fill

# --- Fonts -------------------------------------------------------------------
UI_FONTS = '"SF Pro Text", "Inter", "Helvetica Neue", "Segoe UI", sans-serif'
MONO_FONTS = '"SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace'

# --- Geometry ----------------------------------------------------------------
RADIUS_CARD = 14
RADIUS_CAP = 9
SIDEBAR_WIDTH = 340


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
        font-size: 13px;
    }}
    QMainWindow, #Root {{
        background: {BG};
    }}
    #Card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_CARD}px;
    }}
    #CardTitle {{
        color: {MUTED};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
    }}
    #HeaderTitle {{
        color: {TEXT};
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    #HeaderBar {{
        background: {BG_ELEV};
        border-bottom: 1px solid {BORDER};
    }}
    #RowName {{ color: {TEXT}; font-size: 13px; font-weight: 600; }}
    #RowFinger {{ color: {FAINT}; font-size: 11px; }}
    #FpsLabel {{ color: {MUTED}; font-family: {MONO_FONTS}; font-size: 12px; }}
    QPushButton {{
        background: {PANEL};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 9px;
        padding: 7px 14px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background: {PANEL_HI}; border-color: {BORDER_HI}; }}
    QPushButton:pressed {{ background: {BG_ELEV}; }}
    QPushButton:disabled {{ color: {FAINT}; border-color: {BORDER}; }}
    QPushButton#PrimaryButton {{
        background: {ACCENT}; color: #0A0E0C; border: none;
    }}
    QPushButton#PrimaryButton:hover {{ background: #34E483; }}
    QPushButton#LiveButton[live="true"] {{
        background: {LIVE}; color: #1A0606; border: none;
    }}
    QPushButton#LiveButton[live="true"]:hover {{ background: #FF7070; }}
    QPushButton#PinButton[pinned="true"] {{
        background: {MOVE}; color: #04141A; border: none;
    }}
    QToolTip {{
        background: {BG_ELEV}; color: {TEXT};
        border: 1px solid {BORDER_HI}; border-radius: 6px; padding: 4px 7px;
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
    font.setPointSize(13)
    app.setFont(font)

    app.setStyleSheet(_stylesheet())
    # Keep widget backgrounds crisp on top of the dark window.
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, True)


__all__ = [
    "ACCENT", "BG", "BG_ELEV", "BORDER", "BORDER_HI", "FAINT", "IDLE", "LIVE", "LOOK",
    "MONO_FONTS", "MOVE", "MUTED", "PANEL", "PANEL_HI", "RADIUS_CAP", "RADIUS_CARD",
    "SIDEBAR_WIDTH", "TEXT", "UI_FONTS", "WARN", "apply_theme", "color",
]
