"""
wara_shell_v1.py — Design iteration for the modernized WARA GUI.

An EMPTY SHELL (no real wara functionality wired up) used to agree on look &
layout before touching production code.

Layout — three columns:
  1. NAV panel     : logo + WARA wordmark, the 7 former tabs as color-dot rows,
                     global File actions, version.
  2. OPTIONS panel : changes with the selected tab (QStackedWidget). Holds the
                     controls/stats specific to that tab. Spectrum is fully
                     mocked; the rest are placeholders for now.
  3. PLOT area     : large content area (QStackedWidget). Spectrum shows a real
                     synthetic plot + matplotlib nav toolbar.

Run:  python scratch/wara_shell_v1.py
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QButtonGroup, QFrame, QSizePolicy, QSpacerItem,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QScrollArea, QLineEdit,
)
from PyQt5.QtCore import Qt, QSize, QPoint
from PyQt5.QtGui import QColor, QPalette, QPixmap, QPainter, QIcon, QBrush

# ── Palette ─────────────────────────────────────────────────────────────────
BG_DARK      = "#0a0a0f"
BG_PANEL     = "#12121d"
BG_PANEL2    = "#0f0f18"
BG_INPUT     = "#1b1b2b"
BG_BTN       = "#272740"
BORDER_BTN   = "#454571"
BG_PLOT      = "#07070f"
LOGO_GREEN   = "#5cf04a"
ACCENT_CYAN  = "#00e5ff"
ACCENT_AMBER = "#ffb300"
ACCENT_GREEN = "#39ff14"
ACCENT_RED   = "#ff5577"
BORDER       = "#2a2a45"
GRID         = "#1a1a2e"
TEXT_PRIMARY = "#f4f6ff"
TEXT_DIM     = "#aab2e0"

FONT_FAMILY  = "'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif"
MONO_FAMILY  = "'Consolas', 'Cascadia Mono', monospace"

LOGO_PATH = os.path.join(os.path.dirname(__file__), "..", "wara", "ui", "wara-logo.png")

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

# Tabs that currently have a mocked options column. Tabs not listed here
# collapse the options column and give the plot the full width.
TABS_WITH_OPTIONS = {"Spectrum"}

# ── Dark matplotlib style ───────────────────────────────────────────────────
matplotlib.rcParams.update({
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

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: 15px;
}}

QWidget#nav_panel {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QWidget#opt_panel {{
    background-color: {BG_PANEL2};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

QLabel#title {{
    font-size: 31px;
    font-weight: 800;
    letter-spacing: 3px;
    color: {LOGO_GREEN};
}}
QLabel#section_header {{
    font-size: 13px;
    color: {TEXT_DIM};
    letter-spacing: 2px;
    font-weight: 700;
    padding: 8px 0px 2px 2px;
}}
QLabel#opt_title {{
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 2px;
    color: {TEXT_PRIMARY};
}}
QFrame#separator {{
    background-color: {BORDER};
    min-height: 1px;
    max-height: 1px;
}}

QPushButton#nav_btn {{
    background-color: {BG_BTN};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_BTN};
    border-radius: 6px;
    padding: 10px 12px;
    text-align: left;
    font-size: 16px;
    font-weight: 600;
}}
QPushButton#nav_btn:hover {{ background-color: #313152; border-color: #5a5a8c; }}
QPushButton#nav_btn:checked {{
    background-color: #16384f;
    border: 1px solid {ACCENT_CYAN};
    color: #ffffff;
    font-weight: 700;
}}

QPushButton#action_btn {{
    background-color: {BG_BTN};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_BTN};
    border-radius: 5px;
    padding: 9px 14px;
    font-size: 14px;
    font-weight: 600;
}}
QPushButton#action_btn:hover {{ background-color: #16384f; border: 1px solid {ACCENT_CYAN}; color: {ACCENT_CYAN}; }}

QPushButton#primary_btn {{
    background-color: #11405e;
    color: #aef3ff;
    border: 1px solid {ACCENT_CYAN};
    border-radius: 5px;
    padding: 9px 14px;
    font-size: 14px;
    font-weight: 700;
}}
QPushButton#primary_btn:hover {{ background-color: #1a5778; }}
QPushButton#danger_btn {{
    background-color: #3a1420;
    color: #ff8aa0;
    border: 1px solid {ACCENT_RED};
    border-radius: 5px;
    padding: 9px 14px;
    font-size: 14px;
    font-weight: 700;
}}
QPushButton#danger_btn:hover {{ background-color: #4d1a2a; }}

QPushButton#yellow_btn {{
    background-color: #463808;
    color: #ffd766;
    border: 1px solid {ACCENT_AMBER};
    border-radius: 5px;
    padding: 9px 14px;
    font-size: 14px;
    font-weight: 700;
}}
QPushButton#yellow_btn:hover {{ background-color: #5c4a0a; }}

QCheckBox {{ color: {TEXT_PRIMARY}; spacing: 9px; font-size: 15px; padding: 3px 0; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER}; border-radius: 4px; background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{ background: {ACCENT_CYAN}; border-color: {ACCENT_CYAN}; }}

QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background: {BG_INPUT}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 4px 6px; color: {TEXT_PRIMARY}; font-family: {MONO_FAMILY}; font-size: 14px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {BG_PANEL}; border: 1px solid {BORDER}; color: {TEXT_PRIMARY};
    selection-background-color: #16263a; selection-color: {ACCENT_CYAN};
}}

QPushButton#fit_btn {{
    background-color: #1c4a1c; color: #c6ffb8;
    border: 1px solid {LOGO_GREEN}; border-radius: 5px;
    padding: 10px 14px; font-size: 15px; font-weight: 800; letter-spacing: 1px;
}}
QPushButton#fit_btn:hover {{ background-color: #266026; }}
QPushButton#mini_btn {{
    background-color: {BG_BTN}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_BTN}; border-radius: 4px;
    padding: 4px 7px; font-size: 13px; font-weight: 600;
}}
QPushButton#mini_btn:hover {{ background-color: #16384f; border: 1px solid {ACCENT_CYAN}; color: {ACCENT_CYAN}; }}

/* High-contrast matplotlib toolbar (placed on top of the plot) */
QToolBar#plot_toolbar {{
    background: {BG_INPUT}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 4px; spacing: 2px;
}}
QToolBar#plot_toolbar QToolButton {{
    background: transparent; border: 1px solid transparent;
    border-radius: 4px; padding: 6px 8px;
}}
QToolBar#plot_toolbar QToolButton:hover {{
    background: #16263a; border: 1px solid {ACCENT_CYAN};
}}
QToolBar#plot_toolbar QToolButton:checked {{
    background: #16263a; border: 1px solid {ACCENT_CYAN};
}}
QToolBar#plot_toolbar QLabel {{ color: {TEXT_DIM}; font-family: {MONO_FAMILY}; }}

QPushButton#collapse_btn {{
    background: {BG_INPUT}; color: {TEXT_DIM};
    border: 1px solid {BORDER}; border-radius: 4px;
    font-size: 17px; font-weight: 800; padding: 0;
}}
QPushButton#collapse_btn:hover {{ color: {ACCENT_CYAN}; border-color: {ACCENT_CYAN}; }}

QFrame#subpanel {{
    background-color: {BG_INPUT};
    border: 1px solid {ACCENT_CYAN};
    border-radius: 6px;
}}
QFrame#subpanel QLabel {{ font-size: 14px; color: {TEXT_DIM}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QLabel#stat_key {{ color: {TEXT_DIM}; font-size: 14px; }}
QLabel#version {{ color: {TEXT_DIM}; font-size: 11px; letter-spacing: 1px; }}

QWidget#content {{
    background-color: {BG_PLOT};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QLabel#placeholder {{
    color: {TEXT_DIM}; font-size: 19px; font-weight: 700; letter-spacing: 2px;
}}

QStatusBar {{
    background: {BG_PANEL}; color: {TEXT_DIM};
    border-top: 1px solid {BORDER}; font-size: 13px; letter-spacing: 1px;
}}
"""


# ── Small helpers ───────────────────────────────────────────────────────────
def hsep():
    f = QFrame(); f.setObjectName("separator"); f.setFrameShape(QFrame.NoFrame)
    return f


def header(text):
    h = QLabel(text); h.setObjectName("section_header")
    return h


def dot_icon(color, size=14):
    pm = QPixmap(size, size); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(color)))
    p.drawEllipse(1, 1, size - 2, size - 2); p.end()
    return QIcon(pm)


def recolor_toolbar_icons(toolbar, color):
    """matplotlib ships dark icons; tint them light for contrast on dark bg."""
    for action in toolbar.actions():
        ic = action.icon()
        if ic.isNull():
            continue
        pm = ic.pixmap(QSize(22, 22))
        if pm.isNull():
            continue
        tinted = QPixmap(pm.size())
        tinted.fill(Qt.transparent)
        p = QPainter(tinted)
        p.drawPixmap(0, 0, pm)
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(tinted.rect(), QColor(color))
        p.end()
        action.setIcon(QIcon(tinted))


def stat_row(key, accent=TEXT_PRIMARY):
    row = QWidget()
    lay = QHBoxLayout(row); lay.setContentsMargins(2, 2, 2, 2)
    k = QLabel(key); k.setObjectName("stat_key")
    v = QLabel("—")
    v.setStyleSheet(f"color: {accent}; font-size: 15px; font-weight: 700; "
                    f"font-family: {MONO_FAMILY};")
    v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lay.addWidget(k); lay.addStretch(1); lay.addWidget(v)
    return row


# ── Matplotlib canvas with a synthetic demo spectrum ────────────────────────
class SpectrumCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10, 5), tight_layout=True, facecolor=BG_PLOT)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._draw_demo()

    def _draw_demo(self):
        np.random.seed(42)
        n = 4096; ch = np.arange(n); cal = 0.345
        counts = (50 * np.exp(-ch / 2000) + 10 + np.random.poisson(8, n)).astype(float)
        peaks = [(244, 3000, 20), (344, 9000, 18), (779, 2500, 20),
                 (964, 1500, 22), (1112, 3500, 24), (1408, 4500, 26)]
        for kev, amp, sig in peaks:
            counts += amp * np.exp(-0.5 * ((ch - kev / cal) / sig) ** 2)
        counts = np.random.poisson(counts).astype(float)
        x = cal * ch
        ax = self.ax
        ax.fill_between(x, counts, alpha=0.12, color=ACCENT_CYAN, linewidth=0)
        ax.step(x, counts, where="mid", color=ACCENT_CYAN, linewidth=0.9, alpha=0.95)
        for kev, _amp, _sig in peaks:
            ax.axvline(kev, color=ACCENT_AMBER, linewidth=0.7, linestyle="--", alpha=0.6)
            ax.text(kev, counts.max() * 0.9, f"{kev}", color=ACCENT_AMBER,
                    fontsize=8, ha="center", rotation=90, alpha=0.85)
        ax.set_xlabel("Energy (keV)", color=TEXT_DIM, fontsize=11, fontweight="bold")
        ax.set_ylabel("Counts", color=TEXT_DIM, fontsize=11, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.tick_params(colors=TEXT_DIM, which="both", length=3)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", color=GRID, linewidth=0.5, alpha=0.7)
        self.draw_idle()


# ── Pages (the plot column) ─────────────────────────────────────────────────
class SpectrumPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        self.canvas = SpectrumCanvas()
        # Toolbar on TOP of the plot, high contrast.
        nav = NavToolbar(self.canvas, self)
        nav.setObjectName("plot_toolbar")
        nav.setIconSize(QSize(22, 22))
        recolor_toolbar_icons(nav, TEXT_PRIMARY)
        lay.addWidget(nav)
        lay.addWidget(self.canvas, stretch=1)


class PlaceholderPage(QWidget):
    def __init__(self, name):
        super().__init__()
        self.setObjectName("content")
        lay = QVBoxLayout(self)
        lab = QLabel(f"{name.upper()} PLOT AREA")
        lab.setObjectName("placeholder")
        lab.setAlignment(Qt.AlignCenter)
        lay.addWidget(lab)


# ── Options panels (the middle column) ──────────────────────────────────────
class PeakFindPanel(QFrame):
    """Collapsible sub-panel revealed when Auto-Find Peaks is clicked —
    mirrors wara's Find-Peaks options, plus an 'Add peaks manually' toggle
    (= Add peak). Expands inline, pushing the buttons below it down."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("subpanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)

        def field(label, default=""):
            e = QLineEdit(default); e.setMaximumWidth(86)
            lay.addWidget(_labeled_row(label, e))
            return e

        self.snr = field("SNR >", "3")
        self.ref_ch = field("Ref. channel")
        self.ref_fwhm = field("Ref. FWHM")

        # X-range (optional)
        xr = QWidget(); xl = QHBoxLayout(xr); xl.setContentsMargins(0, 0, 0, 0); xl.setSpacing(5)
        xl.addWidget(QLabel("Xrange")); xl.addStretch(1)
        self.x0 = QLineEdit(); self.x0.setPlaceholderText("x0"); self.x0.setMaximumWidth(62)
        self.x1 = QLineEdit(); self.x1.setPlaceholderText("x1"); self.x1.setMaximumWidth(62)
        xl.addWidget(self.x0); xl.addWidget(self.x1)
        lay.addWidget(xr)

        det = QComboBox(); det.addItems(["HPGe", "LaBr/CeBr", "NaI", "Plastic Scint."])
        lay.addWidget(_labeled_row("Detector", det))
        self.detector = det

        lay.addWidget(hsep())
        self.cb_quick = QCheckBox("Quick Find"); self.cb_quick.setChecked(True)
        self.cb_kernel = QCheckBox("Kernel Method")
        self.cb_snr = QCheckBox("Plot SNR")
        self.cb_manual = QCheckBox("Add peaks manually")
        self.cb_manual.setToolTip("Click on the plot to place peaks by hand")
        for cb in (self.cb_quick, self.cb_kernel, self.cb_snr, self.cb_manual):
            lay.addWidget(cb)

        b = QPushButton("Find Peaks"); b.setObjectName("primary_btn")
        b.setCursor(Qt.PointingHandCursor)
        lay.addWidget(b)


def _labeled_row(label, widget, apply_btn=False):
    """A 'Label  [input]  (Apply)' row used by the customize controls."""
    row = QWidget()
    rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(6)
    rl.addWidget(QLabel(label))
    rl.addStretch(1)
    rl.addWidget(widget)
    if apply_btn:
        b = QPushButton("Apply"); b.setObjectName("mini_btn")
        b.setCursor(Qt.PointingHandCursor)
        rl.addWidget(b)
    return row


def spectrum_options():
    """Fully mocked options for the Spectrum tab (scrollable)."""
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(6)

    title = QLabel("SPECTRUM"); title.setObjectName("opt_title")
    lay.addWidget(title)
    lay.addWidget(hsep())

    # ── Live readouts ──────────────────────────────────────────────
    lay.addWidget(header("CURSOR"))
    lay.addWidget(stat_row("X (ch/keV)", ACCENT_CYAN))
    lay.addWidget(stat_row("Counts", ACCENT_GREEN))
    lay.addWidget(header("STATS"))
    lay.addWidget(stat_row("Channels"))
    lay.addWidget(stat_row("Max Counts", ACCENT_AMBER))
    lay.addWidget(stat_row("Total Counts"))

    # ── Display ────────────────────────────────────────────────────
    lay.addWidget(hsep())
    lay.addWidget(header("DISPLAY"))
    lay.addWidget(QCheckBox("Log Y-axis"))
    pk = QCheckBox("Show Peaks"); pk.setChecked(True)
    lay.addWidget(pk)
    cb_cr = QCheckBox("Set as count rate")
    lay.addWidget(cb_cr)
    cb_keep = QCheckBox("Keep spectrum visible")
    cb_keep.setToolTip("Overlay newly loaded spectra on top of the current one")
    lay.addWidget(cb_keep)

    # ── Customize (from wara's Customize window) ───────────────────
    lay.addWidget(hsep())
    lay.addWidget(header("CUSTOMIZE"))
    spin = QSpinBox(); spin.setRange(3, 99); spin.setSingleStep(2); spin.setValue(7)
    spin.setMaximumWidth(62)
    lay.addWidget(_labeled_row("Smooth", spin, apply_btn=True))
    shift_box = QDoubleSpinBox(); shift_box.setRange(-1e6, 1e6); shift_box.setDecimals(2)
    shift_box.setMaximumWidth(62)
    lay.addWidget(_labeled_row("Shift", shift_box, apply_btn=True))
    rebin = QSpinBox(); rebin.setRange(1, 64); rebin.setValue(1); rebin.setMaximumWidth(62)
    lay.addWidget(_labeled_row("Rebin", rebin, apply_btn=True))
    yconst = QDoubleSpinBox(); yconst.setRange(-1e9, 1e9); yconst.setDecimals(3)
    yconst.setValue(1.0); yconst.setMaximumWidth(62)
    lay.addWidget(_labeled_row("y × const", yconst, apply_btn=True))
    xconst = QDoubleSpinBox(); xconst.setRange(-1e9, 1e9); xconst.setDecimals(3)
    xconst.setValue(1.0); xconst.setMaximumWidth(62)
    lay.addWidget(_labeled_row("x × const", xconst, apply_btn=True))
    b_labels = QPushButton("Axis & Legend…"); b_labels.setObjectName("action_btn")
    b_labels.setCursor(Qt.PointingHandCursor); lay.addWidget(b_labels)

    # ── Spectrum actions ───────────────────────────────────────────
    lay.addWidget(hsep())
    lay.addWidget(header("SPECTRUM"))
    b_reset = QPushButton("Reset View"); b_reset.setObjectName("action_btn")
    b_remcal = QPushButton("Remove Calibration"); b_remcal.setObjectName("action_btn")
    b_addsub = QPushButton("Add / Subtract"); b_addsub.setObjectName("action_btn")
    for b in (b_reset, b_remcal, b_addsub):
        b.setCursor(Qt.PointingHandCursor); lay.addWidget(b)

    # ── Peaks & fitting ────────────────────────────────────────────
    lay.addWidget(hsep())
    lay.addWidget(header("PEAKS & FITTING"))
    b_fit = QPushButton("Drag and Fit"); b_fit.setObjectName("fit_btn")
    b_fit.setToolTip("Open the combined fitting window (peakfit + advanced fit)")
    b_fit.setCursor(Qt.PointingHandCursor)
    lay.addWidget(b_fit)

    # Auto-Find Peaks expands an inline options sub-panel (pushes Isotope ID down)
    b_find = QPushButton("Auto-Find Peaks  ▾"); b_find.setObjectName("primary_btn")
    b_find.setCursor(Qt.PointingHandCursor)
    lay.addWidget(b_find)
    pf_panel = PeakFindPanel()
    pf_panel.setVisible(False)
    lay.addWidget(pf_panel)

    def _toggle_pf():
        show = not pf_panel.isVisible()
        pf_panel.setVisible(show)
        b_find.setText("Auto-Find Peaks  ▴" if show else "Auto-Find Peaks  ▾")
    b_find.clicked.connect(_toggle_pf)

    b_iso = QPushButton("Isotope ID"); b_iso.setObjectName("yellow_btn")
    b_iso.setCursor(Qt.PointingHandCursor)
    lay.addWidget(b_iso)

    lay.addStretch(1)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(inner)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return scroll


def placeholder_options(name):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(14, 14, 14, 14)
    lay.setSpacing(6)
    title = QLabel(name.upper()); title.setObjectName("opt_title")
    lay.addWidget(title)
    lay.addWidget(hsep())
    hint = QLabel(f"{name} options\nwill live here")
    hint.setObjectName("stat_key"); hint.setWordWrap(True)
    lay.addWidget(hint)
    lay.addStretch(1)
    return w


# ── Main window ─────────────────────────────────────────────────────────────
class WaraShell(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WARA  ·  Spectrum Analysis")
        self.resize(1920, 980)
        self.setMinimumSize(1480, 760)
        self.setStyleSheet(STYLESHEET)

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(BG_DARK))
        self.setPalette(pal)

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_nav())          # column 1
        root.addWidget(self._build_options())       # column 2
        root.addWidget(self._build_stack(), stretch=1)  # column 3

        self.statusBar().showMessage("  Ready  ·  design shell — no functionality wired yet")

    # column 1 — global navigation
    def _build_nav(self):
        panel = QWidget(); panel.setObjectName("nav_panel")
        panel.setAttribute(Qt.WA_StyledBackground, True)
        panel.setFixedWidth(230)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(14, 16, 14, 16); lay.setSpacing(8)

        brand = QHBoxLayout(); brand.setSpacing(10)
        logo = QLabel(); pm = QPixmap(LOGO_PATH)
        if not pm.isNull():
            logo.setPixmap(pm.scaledToHeight(44, Qt.SmoothTransformation))
        brand.addWidget(logo)
        t = QLabel("WARA"); t.setObjectName("title")
        brand.addWidget(t); brand.addStretch(1)
        lay.addLayout(brand)
        lay.addSpacing(6); lay.addWidget(hsep())

        lay.addWidget(header("ANALYSIS"))
        self.nav_group = QButtonGroup(self); self.nav_group.setExclusive(True)
        for idx, (name, color) in enumerate(NAV_SECTIONS):
            btn = QPushButton("   " + name); btn.setObjectName("nav_btn")
            btn.setCheckable(True); btn.setCursor(Qt.PointingHandCursor)
            btn.setIcon(dot_icon(color)); btn.setIconSize(QSize(14, 14))
            self.nav_group.addButton(btn, idx); lay.addWidget(btn)
        self.nav_group.idClicked.connect(self._on_nav)

        lay.addSpacing(4); lay.addWidget(hsep())
        lay.addWidget(header("FILE"))
        for text in ("Open Spectrum", "Save", "Clear"):
            b = QPushButton(text); b.setObjectName("action_btn")
            b.setCursor(Qt.PointingHandCursor); lay.addWidget(b)

        lay.addItem(QSpacerItem(20, 12, QSizePolicy.Minimum, QSizePolicy.Expanding))
        ver = QLabel("v1.0"); ver.setObjectName("version")
        ver.setAlignment(Qt.AlignHCenter); lay.addWidget(ver)
        return panel

    # column 2 — tab-specific options (collapsible; auto-hides when empty)
    OPT_W = 376

    def _build_options(self):
        panel = QWidget(); panel.setObjectName("opt_panel")
        panel.setAttribute(Qt.WA_StyledBackground, True)
        panel.setFixedWidth(self.OPT_W)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        # Top bar with the collapse toggle
        bar = QWidget()
        bl = QHBoxLayout(bar); bl.setContentsMargins(6, 6, 6, 0); bl.setSpacing(0)
        bl.addStretch(1)
        self.collapse_btn = QPushButton("‹"); self.collapse_btn.setObjectName("collapse_btn")
        self.collapse_btn.setCursor(Qt.PointingHandCursor)
        self.collapse_btn.setFixedSize(24, 24)
        self.collapse_btn.clicked.connect(self._toggle_options)
        bl.addWidget(self.collapse_btn)
        outer.addWidget(bar)

        self.opt_stack = QStackedWidget()
        for name, _c in NAV_SECTIONS:
            self.opt_stack.addWidget(
                spectrum_options() if name in TABS_WITH_OPTIONS
                else placeholder_options(name))
        outer.addWidget(self.opt_stack)
        self.opt_panel = panel
        self._opt_collapsed = False
        return panel

    def _toggle_options(self):
        self._opt_collapsed = not self._opt_collapsed
        self._apply_opt_state()

    def _apply_opt_state(self):
        if self._opt_collapsed:
            self.opt_stack.hide()
            self.opt_panel.setFixedWidth(34)
            self.collapse_btn.setText("›")
            self.collapse_btn.setToolTip("Expand options")
        else:
            self.opt_stack.show()
            self.opt_panel.setFixedWidth(self.OPT_W)
            self.collapse_btn.setText("‹")
            self.collapse_btn.setToolTip("Collapse options")

    # column 3 — plots
    def _build_stack(self):
        self.stack = QStackedWidget()
        for name, _c in NAV_SECTIONS:
            self.stack.addWidget(SpectrumPage() if name == "Spectrum"
                                 else PlaceholderPage(name))
        return self.stack

    def _on_nav(self, idx):
        name = NAV_SECTIONS[idx][0]
        self.stack.setCurrentIndex(idx)
        self.opt_stack.setCurrentIndex(idx)
        # Auto-hide the whole column for tabs with no options; otherwise show
        # it honoring the user's manual collapse state.
        has_opts = name in TABS_WITH_OPTIONS
        self.opt_panel.setVisible(has_opts)
        if has_opts:
            self._apply_opt_state()
        self.statusBar().showMessage(f"  {name}")

    def show(self):
        super().show()
        first = self.nav_group.button(0)
        if first:
            first.setChecked(True); self._on_nav(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("WARA")
    win = WaraShell()
    win.show()
    sys.exit(app.exec_())
