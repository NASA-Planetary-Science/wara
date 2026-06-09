"""Palette, fonts, stylesheet, and small paint helpers for the beta GUI."""
import matplotlib
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPixmap, QPainter, QIcon, QBrush

# ── Palette ─────────────────────────────────────────────────────────────────
BG_DARK      = "#0a0a0f"
BG_PANEL     = "#12121d"
BG_PANEL2    = "#0f0f18"
BG_INPUT     = "#1b1b2b"
BG_BTN       = "#272740"
BORDER_BTN   = "#6a6ab0"
BG_PLOT      = "#07070f"
LOGO_GREEN   = "#5cf04a"
ACCENT_CYAN  = "#00e5ff"
ACCENT_AMBER = "#ffb300"
ACCENT_GREEN = "#39ff14"
ACCENT_RED   = "#ff5577"
BORDER       = "#2a2a45"
GRID         = "#33335a"
TEXT_PRIMARY = "#f4f6ff"
TEXT_DIM     = "#aab2e0"
SNR_PURPLE   = "#b388ff"   # SNR overlay curve
REF_LINE     = "#ff3df0"   # Nuclear Database reference lines

# Colors for kept/overlaid spectra (active spectrum stays cyan).
OVERLAY_COLORS = ["#ff8a3d", "#5cf04a", "#ffb300", "#b388ff", "#4d9bff", "#ff5577"]

FONT_FAMILY  = "'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif"
MONO_FAMILY  = "'Consolas', 'Cascadia Mono', monospace"

# Former tabs → (label, distinct dot color)
NAV_SECTIONS = [
    ("Spectrum",    "#00e5ff"),
    ("Calibration", "#ffb300"),
    ("Efficiency",  "#5cf04a"),
    ("Resolution",  "#ff5577"),
    ("API",         "#b388ff"),
    ("Diagnostics", "#ff8a3d"),
    ("PNG",         "#4d9bff"),
]

# Tabs with a real/mocked options column. Others collapse the column.
TABS_WITH_OPTIONS = {"Spectrum"}


def apply_mpl_theme():
    matplotlib.rcParams.update({
        "font.size":         12,   # accessibility floor (tick labels, etc.)
        "figure.facecolor":  BG_PLOT,
        "axes.facecolor":    BG_PLOT,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_DIM,
        "axes.grid":         True,
        "grid.color":        GRID,
        "grid.linewidth":    0.5,
        "grid.alpha":        0.7,
        "xtick.color":       TEXT_DIM,
        "ytick.color":       TEXT_DIM,
        "text.color":        TEXT_PRIMARY,
        "savefig.facecolor": BG_PLOT,
    })


def dot_icon(color, size=14):
    pm = QPixmap(size, size); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(color)))
    p.drawEllipse(1, 1, size - 2, size - 2); p.end()
    return QIcon(pm)


def recolor_toolbar_icons(toolbar, color):
    """matplotlib ships dark icons; tint them light for contrast on dark bg."""
    dpr = toolbar.devicePixelRatioF() or 1.0
    phys = int(22 * dpr)
    for action in toolbar.actions():
        ic = action.icon()
        if ic.isNull():
            continue
        pm = ic.pixmap(QSize(phys, phys))
        if pm.isNull():
            continue
        tinted = QPixmap(pm.size()); tinted.fill(Qt.transparent)
        p = QPainter(tinted)
        p.drawPixmap(0, 0, pm)
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(tinted.rect(), QColor(color))
        p.end()
        tinted.setDevicePixelRatio(dpr)
        action.setIcon(QIcon(tinted))


STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: 15px;
}}

QWidget#nav_panel {{ background-color: {BG_PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QWidget#opt_panel {{ background-color: {BG_PANEL2}; border: 1px solid {BORDER}; border-radius: 8px; }}
QWidget#content {{ background-color: {BG_PLOT}; border: 1px solid {BORDER}; border-radius: 8px; }}

QLabel#title {{ font-size: 31px; font-weight: 800; letter-spacing: 3px; color: {ACCENT_GREEN}; }}
QLabel#section_header {{
    font-size: 13px; color: {TEXT_DIM}; letter-spacing: 2px; font-weight: 700;
    padding: 8px 0px 2px 2px;
}}
QLabel#opt_title {{ font-size: 18px; font-weight: 800; letter-spacing: 2px; color: {TEXT_PRIMARY}; }}
QFrame#separator {{ background-color: {BORDER}; min-height: 1px; max-height: 1px; }}

QPushButton#nav_btn {{
    background-color: {BG_BTN}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER_BTN};
    border-radius: 6px; padding: 10px 12px; text-align: left; font-size: 16px; font-weight: 600;
}}
QPushButton#nav_btn:hover {{ background-color: #313152; border-color: #5a5a8c; }}
QPushButton#nav_btn:checked {{
    background-color: #16384f; border: 1px solid {ACCENT_CYAN}; color: #ffffff; font-weight: 700;
}}

QPushButton#action_btn {{
    background-color: {BG_BTN}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER_BTN};
    border-radius: 5px; padding: 9px 14px; font-size: 14px; font-weight: 600;
}}
QPushButton#action_btn:hover {{ background-color: #16384f; border: 1px solid {ACCENT_CYAN}; color: {ACCENT_CYAN}; }}
QPushButton:disabled {{ color: #5a5a7a; border-color: #2e2e48; background-color: #1a1a28; }}

QPushButton#primary_btn {{
    background-color: #11405e; color: #aef3ff; border: 1px solid {ACCENT_CYAN};
    border-radius: 5px; padding: 9px 14px; font-size: 14px; font-weight: 700;
}}
QPushButton#primary_btn:hover {{ background-color: #1a5778; }}
QPushButton#danger_btn {{
    background-color: #3a1420; color: #ff8aa0; border: 1px solid {ACCENT_RED};
    border-radius: 5px; padding: 9px 14px; font-size: 14px; font-weight: 700;
}}
QPushButton#danger_btn:hover {{ background-color: #4d1a2a; }}
QPushButton#yellow_btn {{
    background-color: #463808; color: #ffd766; border: 1px solid {ACCENT_AMBER};
    border-radius: 5px; padding: 9px 14px; font-size: 14px; font-weight: 700;
}}
QPushButton#yellow_btn:hover {{ background-color: #5c4a0a; }}
QPushButton#fit_btn {{
    background-color: #1c4a1c; color: #c6ffb8; border: 1px solid {LOGO_GREEN};
    border-radius: 5px; padding: 10px 14px; font-size: 15px; font-weight: 800; letter-spacing: 1px;
}}
QPushButton#fit_btn:hover {{ background-color: #266026; }}
QPushButton#fit_btn:checked {{
    background-color: #266026; color: {ACCENT_GREEN}; border: 2px solid {ACCENT_GREEN};
}}
QPushButton#mini_btn {{
    background-color: {BG_BTN}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER_BTN};
    border-radius: 4px; padding: 4px 7px; font-size: 13px; font-weight: 600;
}}
QPushButton#mini_btn:hover {{ background-color: #16384f; border: 1px solid {ACCENT_CYAN}; color: {ACCENT_CYAN}; }}

QDialogButtonBox QPushButton {{
    background-color: {BG_BTN}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER_BTN};
    border-radius: 5px; padding: 6px 18px; font-size: 14px; font-weight: 600; min-width: 76px;
}}
QDialogButtonBox QPushButton:hover {{ border: 1px solid {ACCENT_CYAN}; color: {ACCENT_CYAN}; }}
QDialogButtonBox QPushButton:default {{ border: 1px solid {ACCENT_CYAN}; color: {ACCENT_CYAN}; }}

QCheckBox {{ color: {TEXT_PRIMARY}; spacing: 9px; font-size: 15px; padding: 3px 0; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 2px solid #5a5a8c; border-radius: 4px; background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{ background: {ACCENT_CYAN}; border-color: {ACCENT_CYAN}; }}
/* Spectra-list visibility toggles use a neutral color (cyan means "active"). */
QCheckBox#vis_check::indicator:checked {{ background: #9aa0b8; border-color: #9aa0b8; }}

QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background: {BG_INPUT}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 4px 6px; color: {TEXT_PRIMARY}; font-family: {MONO_FAMILY}; font-size: 14px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {BG_PANEL}; border: 1px solid {BORDER}; color: {TEXT_PRIMARY};
    selection-background-color: #16263a; selection-color: {ACCENT_CYAN};
}}

QPushButton#collapse_btn {{
    background: {BG_BTN}; color: {TEXT_DIM}; border: 1px solid #8888c0;
    border-radius: 5px; font-size: 13px; font-weight: 800; padding: 0;
}}
QPushButton#collapse_btn:hover {{ background: #8888c0; color: {BG_DARK}; }}

QPushButton#remove_btn {{
    background: transparent; color: {ACCENT_RED}; border: none;
    font-size: 17px; font-weight: 800; padding: 0;
}}
QPushButton#remove_btn:hover {{ color: #ff8aa0; }}
QLabel#swatch {{ border-radius: 3px; }}

QPushButton#spectrum_name {{
    background: transparent; border: none; text-align: left;
    color: {TEXT_PRIMARY}; font-size: 14px; padding: 0;
}}
QPushButton#spectrum_name:hover {{ color: {ACCENT_CYAN}; }}
QPushButton#spectrum_name_active {{
    background: transparent; border: none; text-align: left;
    color: {ACCENT_CYAN}; font-size: 14px; font-weight: 700; padding: 0;
}}

QFrame#subpanel {{ background-color: {BG_INPUT}; border: 1px solid {ACCENT_CYAN}; border-radius: 6px; }}
QFrame#subpanel QLabel {{ font-size: 14px; color: {TEXT_DIM}; }}

QToolBar#plot_toolbar {{
    background: {BG_INPUT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px; spacing: 2px;
}}
QToolBar#plot_toolbar QToolButton {{
    background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 6px 8px;
}}
QToolBar#plot_toolbar QToolButton:hover {{ background: #16263a; border: 1px solid {ACCENT_CYAN}; }}
QToolBar#plot_toolbar QToolButton:checked {{ background: #16263a; border: 1px solid {ACCENT_CYAN}; }}
QToolBar#plot_toolbar QLabel {{ color: {TEXT_DIM}; font-family: {MONO_FAMILY}; }}

QSlider {{ min-height: 22px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BG_INPUT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACCENT_CYAN}; width: 14px; margin: -6px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #66f0ff; }}
QSlider::sub-page:horizontal {{ background: #16384f; border-radius: 2px; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{
    background: {BG_BTN}; color: {TEXT_DIM}; border: 1px solid {BORDER};
    padding: 7px 18px; margin-right: 3px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    font-size: 14px; font-weight: 700;
}}
QTabBar::tab:selected {{ background: #16384f; color: {ACCENT_CYAN}; border-color: {ACCENT_CYAN}; }}
QTabBar::tab:hover {{ color: {TEXT_PRIMARY}; }}

QTableView {{
    background: {BG_INPUT}; alternate-background-color: #15151f;
    color: {TEXT_PRIMARY}; gridline-color: {BORDER}; border: 1px solid {BORDER};
    selection-background-color: #16384f; selection-color: {ACCENT_CYAN};
    font-size: 15px;
}}
QHeaderView::section {{
    background-color: {BG_BTN}; color: {TEXT_PRIMARY}; border: 1px solid {BORDER};
    padding: 4px 6px; font-weight: 600;
}}
QTableView QTableCornerButton::section {{ background: {BG_BTN}; border: 1px solid {BORDER}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QLabel#stat_key {{ color: {TEXT_DIM}; font-size: 14px; }}
QLabel#version {{ color: {TEXT_DIM}; font-size: 12px; letter-spacing: 1px; }}
QLabel#placeholder {{ color: {TEXT_DIM}; font-size: 19px; font-weight: 700; letter-spacing: 2px; }}

QStatusBar {{
    background: {BG_PANEL}; color: {TEXT_DIM}; border-top: 1px solid {BORDER};
    font-size: 13px; letter-spacing: 1px;
}}
"""
