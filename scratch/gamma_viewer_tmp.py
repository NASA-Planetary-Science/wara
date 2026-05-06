"""
gamma_viewer.py — Modern Gamma-Ray Spectrum Viewer
Requires: PyQt5, matplotlib, numpy, scipy
    pip install PyQt5 matplotlib numpy scipy
"""

import sys
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QStatusBar, QFrame,
    QSlider, QCheckBox, QSplitter, QSizePolicy, QSpacerItem,
    QGroupBox, QScrollArea, QToolBar, QAction, QSpinBox,
    QDoubleSpinBox, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QFontDatabase

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch

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

# ── Global stylesheet ──────────────────────────────────────────────────────────
STYLE = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}
QLabel#title {{
    font-size: 20px;
    font-weight: bold;
    letter-spacing: 4px;
    color: {ACCENT_CYAN};
}}
QLabel#subtitle {{
    font-size: 10px;
    color: {TEXT_DIM};
    letter-spacing: 2px;
}}
QLabel#value_label {{
    font-size: 13px;
    color: {ACCENT_GREEN};
    font-weight: bold;
}}
QLabel#section_header {{
    font-size: 10px;
    color: {TEXT_DIM};
    letter-spacing: 3px;
    font-weight: bold;
    padding: 4px 0px 2px 0px;
}}
QPushButton {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 7px 14px;
    font-family: 'Consolas', monospace;
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border: 1px solid {ACCENT_CYAN};
    color: {ACCENT_CYAN};
}}
QPushButton:pressed {{
    background-color: #0d1b2a;
    border: 1px solid {ACCENT_CYAN};
}}
QPushButton#primary_btn {{
    background-color: #0d1f2d;
    border: 1px solid {ACCENT_CYAN};
    color: {ACCENT_CYAN};
    font-weight: bold;
}}
QPushButton#primary_btn:hover {{
    background-color: #112a3d;
    border: 1px solid #80f0ff;
    color: #80f0ff;
}}
QPushButton#danger_btn {{
    border: 1px solid {ACCENT_RED};
    color: {ACCENT_RED};
}}
QPushButton#danger_btn:hover {{
    background-color: #1a0a10;
    border: 1px solid #ff6b84;
    color: #ff6b84;
}}
QFrame#separator {{
    background-color: {BORDER};
    min-height: 1px;
    max-height: 1px;
}}
QFrame#panel {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 8px;
    font-size: 10px;
    color: {TEXT_DIM};
    letter-spacing: 2px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    top: -1px;
    color: {TEXT_DIM};
}}
QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 8px;
    font-size: 11px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {BG_WIDGET};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_CYAN};
    border-color: {ACCENT_CYAN};
}}
QSlider::groove:horizontal {{
    height: 3px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_CYAN};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT_CYAN};
    border-radius: 2px;
}}
QComboBox {{
    background: {BG_WIDGET};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
    font-family: 'Consolas', monospace;
    font-size: 11px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    color: {TEXT_PRIMARY};
    selection-background-color: {BG_HOVER};
    selection-color: {ACCENT_CYAN};
}}
QSpinBox, QDoubleSpinBox {{
    background: {BG_WIDGET};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 6px;
    color: {TEXT_PRIMARY};
    font-family: 'Consolas', monospace;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {BG_WIDGET};
    border: none;
    width: 16px;
}}
QStatusBar {{
    background: {BG_PANEL};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
    font-size: 10px;
    letter-spacing: 1px;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {BG_DARK};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolBar {{
    background: {BG_PANEL};
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 3px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 3px;
    color: {TEXT_DIM};
    padding: 3px 6px;
    font-size: 11px;
}}
QToolBar QToolButton:hover {{
    border-color: {BORDER};
    color: {TEXT_PRIMARY};
    background: {BG_HOVER};
}}
"""

# ── Matplotlib dark style ──────────────────────────────────────────────────────
MPL_STYLE = {
    "figure.facecolor":       BG_DARK,
    "axes.facecolor":         "#06060e",
    "axes.edgecolor":         BORDER,
    "axes.labelcolor":        TEXT_DIM,
    "axes.grid":              True,
    "grid.color":             GRID,
    "grid.linewidth":         0.5,
    "grid.alpha":             0.7,
    "xtick.color":            TEXT_DIM,
    "ytick.color":            TEXT_DIM,
    "xtick.labelsize":        9,
    "ytick.labelsize":        9,
    "text.color":             TEXT_PRIMARY,
    "figure.edgecolor":       BG_DARK,
    "savefig.facecolor":      BG_DARK,
    "legend.facecolor":       BG_PANEL,
    "legend.edgecolor":       BORDER,
    "legend.labelcolor":      TEXT_PRIMARY,
    "legend.fontsize":        9,
}

for k, v in MPL_STYLE.items():
    matplotlib.rcParams[k] = v


# ── File parsing ───────────────────────────────────────────────────────────────
def load_spectrum(path: str):
    """
    Returns (channels, counts, energy_cal) where energy_cal may be None.
    Supports:
      - .spe / .SPE  (ORTEC / IAEA format)
      - .csv / .txt  (two-column: channel,counts  or  energy,counts)
      - single-column plain text (counts only)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".spe",):
        return _parse_spe(path)
    elif ext in (".csv", ".txt", ".dat"):
        return _parse_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _parse_spe(path):
    channels, counts, energy_cal = [], [], None
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()

    in_data = False
    ecal_coeffs = None
    for i, line in enumerate(lines):
        if "$DATA:" in line:
            in_data = True
            continue
        if in_data and line.startswith("$"):
            in_data = False
        if in_data:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    counts.append(float(parts[0]))
                except ValueError:
                    if not counts:  # skip first line (range)
                        continue
        if "$ENER_FIT:" in line:
            try:
                ecal_coeffs = list(map(float, lines[i + 1].strip().split()))
            except Exception:
                pass

    counts = np.array(counts)
    channels = np.arange(len(counts))
    if ecal_coeffs and len(ecal_coeffs) >= 2:
        energy_cal = ecal_coeffs[0] + ecal_coeffs[1] * channels
    return channels, counts, energy_cal


def _parse_csv(path):
    try:
        data = np.loadtxt(path, delimiter=",", comments="#")
    except Exception:
        try:
            data = np.loadtxt(path, comments="#")
        except Exception as e:
            raise ValueError(f"Cannot parse file: {e}")

    if data.ndim == 1:
        counts = data
        channels = np.arange(len(counts))
        return channels, counts, None
    elif data.shape[1] >= 2:
        channels = data[:, 0]
        counts = data[:, 1]
        return channels, counts, None
    raise ValueError("Cannot interpret file columns.")


# ── Canvas ─────────────────────────────────────────────────────────────────────
class SpectrumCanvas(FigureCanvas):
    crosshair_moved = pyqtSignal(float, float)  # x, y at cursor

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 5), tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {BG_DARK}; border-radius: 6px;")

        self._channels   = None
        self._counts     = None
        self._x_axis     = None   # energy or channel
        self._x_label    = "Channel"
        self._log_y      = False
        self._smooth     = False
        self._smooth_win = 5
        self._show_peaks = True

        self._hline = None
        self._vline = None
        self._peaks = []

        # crosshair
        self._ch_h = None
        self._ch_v = None
        self.mpl_connect("motion_notify_event", self._on_mouse_move)

        self._draw_empty()

    # ── Drawing ────────────────────────────────────────────────────────────────
    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_facecolor("#06060e")
        self.ax.text(
            0.5, 0.5, "NO SPECTRUM LOADED",
            transform=self.ax.transAxes,
            ha="center", va="center",
            fontsize=16, color=BORDER,
            fontfamily="monospace", fontweight="bold",
            alpha=0.5,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.draw()

    def plot_spectrum(self, channels, counts, energy=None,
                      log_y=False, smooth=False, smooth_win=5,
                      show_peaks=True):
        self._channels   = channels
        self._counts     = counts
        self._log_y      = log_y
        self._smooth     = smooth
        self._smooth_win = smooth_win
        self._show_peaks = show_peaks

        if energy is not None:
            self._x_axis  = energy
            self._x_label = "Energy (keV)"
        else:
            self._x_axis  = channels
            self._x_label = "Channel"

        self._redraw()

    def _redraw(self):
        if self._counts is None:
            return

        self.ax.clear()
        self.ax.set_facecolor("#06060e")

        x = self._x_axis
        y = self._counts.copy()

        # optional smoothing (moving average)
        if self._smooth and self._smooth_win > 1:
            w = max(2, self._smooth_win | 1)  # ensure odd
            kernel = np.ones(w) / w
            y = np.convolve(y, kernel, mode="same")

        # gradient fill under curve
        self.ax.fill_between(x, y, alpha=0.12, color=ACCENT_CYAN, linewidth=0)

        # main spectrum line
        self.ax.step(x, y, where="mid", color=ACCENT_CYAN,
                     linewidth=0.9, alpha=0.95, label="Spectrum")

        # peak annotations
        if self._show_peaks and len(self._peaks):
            for px, py, label in self._peaks:
                self.ax.axvline(px, color=ACCENT_AMBER, linewidth=0.8,
                                linestyle="--", alpha=0.7)
                self.ax.text(px, py * 1.05, label,
                             color=ACCENT_AMBER, fontsize=7,
                             ha="center", rotation=45, alpha=0.9)

        if self._log_y:
            self.ax.set_yscale("log")
        else:
            self.ax.set_yscale("linear")

        # axes styling
        self.ax.set_xlabel(self._x_label, color=TEXT_DIM, fontsize=10,
                           fontfamily="monospace")
        self.ax.set_ylabel("Counts", color=TEXT_DIM, fontsize=10,
                           fontfamily="monospace")
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.ax.tick_params(colors=TEXT_DIM, which="both", length=3)
        self.ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.ax.grid(True, which="major", color=GRID, linewidth=0.5, alpha=0.7)
        self.ax.grid(True, which="minor", color=GRID, linewidth=0.2, alpha=0.3)

        # ROI lines
        if self._hline is not None:
            self.ax.axhline(self._hline, color=ACCENT_RED,
                            linewidth=0.8, linestyle=":", alpha=0.7)
        if self._vline is not None:
            self.ax.axvline(self._vline, color=ACCENT_GREEN,
                            linewidth=0.8, linestyle=":", alpha=0.7)

        self.draw_idle()

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return
        self.crosshair_moved.emit(event.xdata or 0, event.ydata or 0)

    def set_log_y(self, enabled):
        self._log_y = enabled
        self._redraw()

    def set_smooth(self, enabled, window=None):
        self._smooth = enabled
        if window is not None:
            self._smooth_win = window
        self._redraw()

    def set_peaks(self, peaks):
        """peaks: list of (x, y, label)"""
        self._peaks = peaks
        self._redraw()

    def add_vline(self, x):
        self._vline = x
        self._redraw()

    def clear_vline(self):
        self._vline = None
        self._redraw()


# ── Stats panel ────────────────────────────────────────────────────────────────
class StatRow(QWidget):
    def __init__(self, label, value="—", color=TEXT_PRIMARY):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        self.val = QLabel(value)
        self.val.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
        self.val.setAlignment(Qt.AlignRight)
        lay.addWidget(lbl)
        lay.addStretch()
        lay.addWidget(self.val)

    def set_value(self, v):
        self.val.setText(str(v))


# ── Sidebar ────────────────────────────────────────────────────────────────────
class SidePanel(QWidget):
    log_toggled     = pyqtSignal(bool)
    smooth_toggled  = pyqtSignal(bool)
    smooth_changed  = pyqtSignal(int)
    peaks_toggled   = pyqtSignal(bool)
    auto_peaks_req  = pyqtSignal()
    clear_peaks_req = pyqtSignal()
    export_req      = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(210)
        self.setObjectName("panel")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 14, 12, 14)
        root.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────────
        title = QLabel("γ-RAY")
        title.setObjectName("title")
        subtitle = QLabel("SPECTRUM ANALYZER")
        subtitle.setObjectName("subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)
        root.addWidget(self._sep())

        # ── Cursor info ────────────────────────────────────────────────────────
        root.addWidget(self._section("CURSOR"))
        self.cur_x = StatRow("X (ch/keV)", "—", ACCENT_CYAN)
        self.cur_y = StatRow("Counts", "—", ACCENT_GREEN)
        root.addWidget(self.cur_x)
        root.addWidget(self.cur_y)
        root.addWidget(self._sep())

        # ── Spectrum stats ─────────────────────────────────────────────────────
        root.addWidget(self._section("SPECTRUM"))
        self.stat_ch  = StatRow("Channels", "—")
        self.stat_max = StatRow("Max Counts", "—", ACCENT_AMBER)
        self.stat_sum = StatRow("Total Counts", "—")
        self.stat_pk  = StatRow("Peak Ch / keV", "—", ACCENT_CYAN)
        for w in (self.stat_ch, self.stat_max, self.stat_sum, self.stat_pk):
            root.addWidget(w)
        root.addWidget(self._sep())

        # ── Display options ────────────────────────────────────────────────────
        root.addWidget(self._section("DISPLAY"))

        self.cb_log = QCheckBox("Log Y-axis")
        self.cb_log.toggled.connect(self.log_toggled)
        root.addWidget(self.cb_log)

        self.cb_smooth = QCheckBox("Smooth")
        self.cb_smooth.toggled.connect(self.smooth_toggled)
        root.addWidget(self.cb_smooth)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("  Window:"))
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(3, 99)
        self.smooth_spin.setSingleStep(2)
        self.smooth_spin.setValue(7)
        self.smooth_spin.setEnabled(False)
        self.smooth_spin.valueChanged.connect(self.smooth_changed)
        smooth_row.addWidget(self.smooth_spin)
        root.addLayout(smooth_row)
        self.cb_smooth.toggled.connect(self.smooth_spin.setEnabled)

        self.cb_peaks = QCheckBox("Show Peaks")
        self.cb_peaks.setChecked(True)
        self.cb_peaks.toggled.connect(self.peaks_toggled)
        root.addWidget(self.cb_peaks)
        root.addWidget(self._sep())

        # ── Peak finding ───────────────────────────────────────────────────────
        root.addWidget(self._section("PEAK FINDER"))
        btn_find = QPushButton("Auto-Find Peaks")
        btn_find.setObjectName("primary_btn")
        btn_find.clicked.connect(self.auto_peaks_req)
        btn_clear = QPushButton("Clear Peaks")
        btn_clear.setObjectName("danger_btn")
        btn_clear.clicked.connect(self.clear_peaks_req)
        root.addWidget(btn_find)
        root.addWidget(btn_clear)
        root.addWidget(self._sep())

        # ── Export ─────────────────────────────────────────────────────────────
        root.addWidget(self._section("EXPORT"))
        btn_export = QPushButton("Save Figure")
        btn_export.clicked.connect(self.export_req)
        root.addWidget(btn_export)

        root.addStretch()

        # version tag
        ver = QLabel("v1.0  ·  LBNL")
        ver.setStyleSheet(f"color: {BORDER}; font-size: 9px; letter-spacing: 1px;")
        ver.setAlignment(Qt.AlignCenter)
        root.addWidget(ver)

    def _sep(self):
        sep = QFrame()
        sep.setObjectName("separator")
        return sep

    def _section(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("section_header")
        return lbl

    def update_cursor(self, x, y):
        self.cur_x.set_value(f"{x:.2f}")
        self.cur_y.set_value(f"{int(y):,}")

    def update_stats(self, channels, counts, x_axis, x_label):
        n = len(counts)
        mx = int(counts.max())
        total = int(counts.sum())
        pk_ch = x_axis[int(np.argmax(counts))]
        unit = "keV" if "keV" in x_label else "ch"
        self.stat_ch.set_value(f"{n:,}")
        self.stat_max.set_value(f"{mx:,}")
        self.stat_sum.set_value(f"{total:,}")
        self.stat_pk.set_value(f"{pk_ch:.1f} {unit}")


# ── Main window ────────────────────────────────────────────────────────────────
class GammaViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("γ-Ray Spectrum Viewer")
        self.resize(1200, 720)
        self.setMinimumSize(900, 580)

        self._channels = None
        self._counts   = None
        self._energy   = None
        self._peaks    = []

        self._apply_style()
        self._build_ui()

    def _apply_style(self):
        self.setStyleSheet(STYLE)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(BG_DARK))
        self.setPalette(pal)

    def _build_ui(self):
        # Toolbar
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        act_open  = QAction("⊕  LOAD SPECTRUM", self)
        act_open.triggered.connect(self._open_file)
        act_demo  = QAction("◈  DEMO DATA", self)
        act_demo.triggered.connect(self._load_demo)
        act_reset = QAction("⟳  RESET ZOOM", self)
        act_reset.triggered.connect(self._reset_zoom)
        for a in (act_open, act_demo, act_reset):
            tb.addAction(a)

        # Central
        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(8, 8, 8, 8)
        main_lay.setSpacing(8)

        # Sidebar
        self.panel = SidePanel()
        self.panel.log_toggled.connect(self._on_log)
        self.panel.smooth_toggled.connect(self._on_smooth)
        self.panel.smooth_changed.connect(self._on_smooth_win)
        self.panel.peaks_toggled.connect(self._on_peaks_toggle)
        self.panel.auto_peaks_req.connect(self._auto_peaks)
        self.panel.clear_peaks_req.connect(self._clear_peaks)
        self.panel.export_req.connect(self._export)

        # Canvas + nav
        canvas_wrap = QWidget()
        canvas_lay  = QVBoxLayout(canvas_wrap)
        canvas_lay.setContentsMargins(0, 0, 0, 0)
        canvas_lay.setSpacing(4)

        self.canvas = SpectrumCanvas()
        self.canvas.crosshair_moved.connect(self._on_cursor)

        nav = NavToolbar(self.canvas, self)
        nav.setStyleSheet(f"""
            QToolBar {{ background: {BG_PANEL}; border: 1px solid {BORDER};
                        border-radius: 4px; padding: 2px; }}
            QToolButton {{ color: {TEXT_DIM}; background: transparent;
                           border: none; padding: 3px 6px; }}
            QToolButton:hover {{ color: {ACCENT_CYAN}; }}
        """)

        canvas_lay.addWidget(self.canvas)
        canvas_lay.addWidget(nav)

        main_lay.addWidget(self.panel)
        main_lay.addWidget(canvas_wrap, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._set_status("Ready  ·  Load a spectrum file to begin")

        self._file_label = QLabel("")
        self._file_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; padding-right: 8px;")
        self.status.addPermanentWidget(self._file_label)

    # ── File I/O ───────────────────────────────────────────────────────────────
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Spectrum",
            filter="Spectrum files (*.spe *.SPE *.csv *.txt *.dat);;All (*)"
        )
        if not path:
            return
        try:
            ch, counts, energy = load_spectrum(path)
            self._load_data(ch, counts, energy, label=os.path.basename(path))
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _load_demo(self):
        """Generate a synthetic Eu-152 like spectrum."""
        np.random.seed(42)
        n = 4096
        ch = np.arange(n)
        bg = 50 * np.exp(-ch / 2000) + 10 + np.random.poisson(8, n)
        peaks_def = [
            (244,  3000,  20),
            (344,  9000,  18),
            (411,  1000,  22),
            (444,   800,  25),
            (779,  2500,  20),
            (964,  1500,  22),
            (1112, 3500,  24),
            (1408, 4500,  26),
        ]
        # map channel → rough keV:  E ≈ 0.345 * ch
        cal = 0.345
        counts = bg.astype(float)
        for pk_kev, amp, sig in peaks_def:
            pk_ch = int(pk_kev / cal)
            counts += amp * np.exp(-0.5 * ((ch - pk_ch) / sig) ** 2)

        counts = np.random.poisson(counts).astype(float)
        energy = cal * ch
        self._load_data(ch, counts, energy, label="DEMO  Eu-152 (synthetic)")

    def _load_data(self, channels, counts, energy, label=""):
        self._channels = channels
        self._counts   = counts
        self._energy   = energy
        self._peaks    = []

        self.canvas.plot_spectrum(
            channels, counts, energy,
            log_y=self.panel.cb_log.isChecked(),
            smooth=self.panel.cb_smooth.isChecked(),
            smooth_win=self.panel.smooth_spin.value(),
            show_peaks=self.panel.cb_peaks.isChecked(),
        )
        x_axis  = energy if energy is not None else channels
        x_label = "Energy (keV)" if energy is not None else "Channel"
        self.panel.update_stats(channels, counts, x_axis, x_label)
        self._file_label.setText(label)
        self._set_status(f"Loaded  ·  {len(counts):,} channels  ·  {int(counts.sum()):,} total counts")

    # ── Controls ───────────────────────────────────────────────────────────────
    def _on_log(self, state):
        self.canvas.set_log_y(state)

    def _on_smooth(self, state):
        self.canvas.set_smooth(state, self.panel.smooth_spin.value())

    def _on_smooth_win(self, val):
        if self.panel.cb_smooth.isChecked():
            self.canvas.set_smooth(True, val)

    def _on_peaks_toggle(self, state):
        self.canvas.plot_spectrum(
            self._channels, self._counts, self._energy,
            log_y=self.panel.cb_log.isChecked(),
            smooth=self.panel.cb_smooth.isChecked(),
            smooth_win=self.panel.smooth_spin.value(),
            show_peaks=state,
        )
        if state:
            self.canvas.set_peaks(self._peaks)

    def _auto_peaks(self):
        if self._counts is None:
            return
        try:
            from scipy.signal import find_peaks
        except ImportError:
            QMessageBox.warning(self, "Missing Dependency",
                                "scipy is required for peak finding.\n  pip install scipy")
            return

        y = self._counts.copy()
        height = y.max() * 0.03
        distance = max(5, len(y) // 100)
        idxs, props = find_peaks(y, height=height, distance=distance,
                                 prominence=height * 0.5)

        x_axis = self._energy if self._energy is not None else self._channels
        self._peaks = [(x_axis[i], y[i], f"{x_axis[i]:.1f}") for i in idxs]
        self.canvas.set_peaks(self._peaks)
        self.panel.cb_peaks.setChecked(True)
        self._set_status(f"Found {len(idxs)} peaks")

    def _clear_peaks(self):
        self._peaks = []
        self.canvas.set_peaks([])
        self._set_status("Peaks cleared")

    def _on_cursor(self, x, y):
        self.panel.update_cursor(x, y)

    def _reset_zoom(self):
        if self._counts is None:
            return
        self.canvas.ax.autoscale()
        self.canvas.draw_idle()

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "spectrum.png",
            filter="PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self.canvas.fig.savefig(path, dpi=200, bbox_inches="tight",
                                    facecolor=BG_DARK)
            self._set_status(f"Saved → {os.path.basename(path)}")

    def _set_status(self, msg):
        self.status.showMessage(f"  {msg}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Gamma Spectrum Viewer")
    win = GammaViewer()
    win.show()
    sys.exit(app.exec_())