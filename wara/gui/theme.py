"""
Dark theme for the WARA GUI.

Call apply_dark_theme(app) in main() after QApplication() and before
creating any widgets. Sets matplotlib rcParams and the app-level QSS.
"""
import matplotlib

# ── Palette ────────────────────────────────────────────────────────────────────
BG_DARK      = "#0a0a0f"
BG_PANEL     = "#0f0f1a"
BG_WIDGET    = "#13131f"
BG_HOVER     = "#1c1c2e"
ACCENT_CYAN  = "#00e5ff"
ACCENT_GREEN = "#39ff14"
ACCENT_AMBER = "#ffb300"
ACCENT_RED   = "#ff3b5c"
TEXT_PRIMARY = "#e8eaf6"
TEXT_DIM     = "#5c6bc0"
BORDER       = "#1e1e3a"
GRID         = "#1a1a2e"

# ── Runtime styling constants (imported by mixin files) ────────────────────────
TABLE_HEADER_STYLE = (
    "::section { background-color: #1a2a3a; color: #e8eaf6;"
    " border: 1px solid #1e1e3a; }"
)
TABLE_INDEX_STYLE = (
    "::section { background-color: #0f1a22; color: #b0b8d0;"
    " border: 1px solid #1e1e3a; }"
)
WARNING_LABEL_STYLE = "color: #ff6b6b; font-size: 10px;"

# ── Matplotlib dark rcParams ───────────────────────────────────────────────────
_MPL_DARK = {
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    "#06060e",
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT_DIM,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.7,
    "xtick.color":       TEXT_DIM,
    "ytick.color":       TEXT_DIM,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "text.color":        TEXT_PRIMARY,
    "figure.edgecolor":  BG_DARK,
    "savefig.facecolor": BG_DARK,
    "legend.facecolor":  BG_PANEL,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": TEXT_PRIMARY,
    "legend.fontsize":   9,
}

# ── Global QSS ─────────────────────────────────────────────────────────────────
DARK_QSS = f"""
/* ── Base ── */
QMainWindow, QDialog, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 11px;
}}
QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}

/* ── Buttons (base) ── */
QPushButton {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {ACCENT_CYAN};
    color: {ACCENT_CYAN};
}}
QPushButton:pressed {{
    background-color: #0d1b2a;
    border-color: {ACCENT_CYAN};
}}
QPushButton:checked {{
    background-color: #0d2030;
    border-color: {ACCENT_CYAN};
    color: {ACCENT_CYAN};
    font-weight: bold;
}}
QPushButton:disabled {{
    background-color: {BG_DARK};
    color: #3a3a5a;
    border-color: #1a1a2a;
}}

/* ── Danger / Reset ── */
#button_remove_cal, #button_reset_cal, #button_reset_eff,
#button_reset_fwhm, #isotID_button_clear, #button_remove_vlines {{
    border-color: #9b2335;
    color: #e05c6c;
}}
#button_remove_cal:hover, #button_reset_cal:hover, #button_reset_eff:hover,
#button_reset_fwhm:hover, #isotID_button_clear:hover,
#button_remove_vlines:hover {{
    background-color: #2a0d12;
    border-color: #e05c6c;
    color: #ff8099;
}}

/* ── Option / Selector (background model, cal type, fit) ── */
#pushButton_poly1, #pushButton_poly2, #pushButton_poly3,
#pushButton_poly4, #pushButton_poly5, #pushButton_exp,
#button_cal1, #button_cal2, #button_cal3,
#button_fit1_eff, #button_fit2_eff, #button_extrapolate {{
    border-color: #2d7a4f;
    color: #4caf7d;
}}
#pushButton_poly1:checked, #pushButton_poly2:checked, #pushButton_poly3:checked,
#pushButton_poly4:checked, #pushButton_poly5:checked, #pushButton_exp:checked,
#button_cal1:checked, #button_cal2:checked, #button_cal3:checked,
#button_extrapolate:checked {{
    background-color: #0d2a1a;
    border-color: #4caf7d;
    color: {ACCENT_GREEN};
    font-weight: bold;
}}

/* ── Toggle (add peak, Gauss selector, cal equations) ── */
#button_add_peak, #pushButton_gauss, #pushButton_gauss2, #button_cal_eqns {{
    border-color: #4a4a5a;
    color: #9090a8;
}}
#button_add_peak:checked, #pushButton_gauss:checked,
#pushButton_gauss2:checked {{
    background-color: #1a1a2e;
    border-color: {ACCENT_CYAN};
    color: {ACCENT_CYAN};
    font-weight: bold;
}}

/* ── Input / Navigation ── */
#button_add_cal, #button_fwhm1, #button_fwhm2, #isotID_button_apply {{
    border-color: #1a4f7a;
    color: #4a90d9;
}}
#button_fwhm1:checked, #button_fwhm2:checked {{
    background-color: #0d1f35;
    border-color: #4a90d9;
    color: #80b8f0;
    font-weight: bold;
}}

/* ── Scale Toggle ── */
#pushButton_scale, #button_origin, #button_origin_fwhm, #button_yscale_eff {{
    border-color: #8a6d00;
    color: #f0c040;
}}

/* ── Apply / Confirm ── */
#button_apply_cal, #button_add_eff {{
    border-color: #7a4a20;
    color: #d4845a;
}}

/* ── Configuration ── */
#button_customize {{
    border-color: #3a5070;
    color: #7a9fc0;
}}

/* ── Data Entry ── */
#button_add_fwhm {{
    border-color: #6a6020;
    color: #c0b060;
}}

/* ── Peak Finding ── */
#button_find_peaks {{
    border-color: #1a6a5a;
    color: #50c0a0;
}}

/* ── Isotope ID ── */
#button_identify_peaks {{
    border-color: #7a5a30;
    color: #c09060;
}}

/* ── Tab Widget ── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background-color: {BG_PANEL};
}}
QTabBar::tab {{
    background-color: {BG_WIDGET};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    border-bottom: none;
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {ACCENT_CYAN};
}}
QTabBar::tab:hover:!selected {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}

/* ── Input Widgets ── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 6px;
    selection-background-color: #1a3a5a;
    selection-color: {TEXT_PRIMARY};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT_CYAN};
}}
QComboBox {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 6px;
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    selection-background-color: {BG_HOVER};
    selection-color: {ACCENT_CYAN};
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 3px 6px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {BG_WIDGET};
    border: none;
    width: 16px;
}}

/* ── Checkboxes & Radio Buttons ── */
QCheckBox, QRadioButton {{
    color: {TEXT_PRIMARY};
    spacing: 6px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {BG_WIDGET};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT_CYAN};
    border-color: {ACCENT_CYAN};
}}
QRadioButton::indicator {{
    border-radius: 7px;
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT_CYAN};
    border-color: {ACCENT_CYAN};
}}

/* ── GroupBox ── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 6px;
    color: {TEXT_DIM};
    font-size: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    top: -1px;
    color: {TEXT_DIM};
}}

/* ── Tables ── */
QTableWidget, QTableView {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    selection-background-color: #1a2a3a;
    selection-color: {ACCENT_CYAN};
    alternate-background-color: {BG_WIDGET};
}}
QHeaderView::section {{
    background-color: #1a2a3a;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    padding: 4px;
    font-size: 10px;
}}
QTableCornerButton::section {{
    background-color: #1a2a3a;
    border: 1px solid {BORDER};
}}

/* ── Scroll Bars ── */
QScrollBar:vertical {{
    background-color: {BG_DARK};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar:horizontal {{
    background-color: {BG_DARK};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background-color: {BORDER};
    border-radius: 4px;
    min-height: 20px;
    min-width: 20px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background-color: {TEXT_DIM};
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

/* ── Menu Bar ── */
QMenuBar {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item:selected {{
    background-color: {BG_HOVER};
    color: {ACCENT_CYAN};
}}
QMenu {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
}}
QMenu::item:selected {{
    background-color: {BG_HOVER};
    color: {ACCENT_CYAN};
}}
QMenu::separator {{
    height: 1px;
    background-color: {BORDER};
    margin: 2px 4px;
}}

/* ── Toolbar (matplotlib NavigationToolbar) ── */
QToolBar {{
    background-color: {BG_PANEL};
    border-bottom: 1px solid {BORDER};
    spacing: 3px;
    padding: 2px;
    font-size: 14px;
}}
QToolBar QToolButton {{
    background-color: transparent;
    color: {TEXT_DIM};
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 3px 5px;
    font-size: 14px;
}}
QToolBar QToolButton:hover {{
    background-color: {BG_HOVER};
    border-color: {BORDER};
    color: {TEXT_PRIMARY};
}}

/* ── Status Bar ── */
QStatusBar {{
    background-color: {BG_PANEL};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
    font-size: 10px;
}}

/* ── Splitter ── */
QSplitter::handle {{
    background-color: {BORDER};
}}

/* ── Slider ── */
QSlider::groove:horizontal {{
    height: 3px;
    background-color: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background-color: {ACCENT_CYAN};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::sub-page:horizontal {{
    background-color: {ACCENT_CYAN};
    border-radius: 2px;
}}
"""


def apply_dark_theme(app):
    """Set matplotlib rcParams and global QSS. Call after QApplication(), before any widgets."""
    for k, v in _MPL_DARK.items():
        matplotlib.rcParams[k] = v
    app.setStyleSheet(DARK_QSS)
