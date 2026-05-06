"""
gamma_viewer_from_ui.py — Gamma-Ray Spectrum Viewer loaded from gamma_viewer.ui
"""

import sys
import os
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QFileDialog, QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5 import uic

# ── Palette (must match the .ui stylesheet) ────────────────────────────────────
BG_DARK      = "#0a0a0f"
BG_PANEL     = "#0f0f1a"
ACCENT_CYAN  = "#00e5ff"
ACCENT_AMBER = "#ffb300"
ACCENT_GREEN = "#39ff14"
ACCENT_RED   = "#ff3b5c"
TEXT_DIM     = "#5c6bc0"
BORDER       = "#1e1e3a"
GRID         = "#1a1a2e"
TEXT_PRIMARY = "#e8eaf6"

UI_FILE = os.path.join(os.path.dirname(__file__), "gamma_viewer.ui")

# ── Apply dark Matplotlib style at import time ─────────────────────────────────
matplotlib.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    "#06060e",
    "axes.edgecolor":    "#1e1e3a",
    "axes.labelcolor":   "#5c6bc0",
    "axes.grid":         True,
    "grid.color":        "#1a1a2e",
    "grid.linewidth":    0.5,
    "grid.alpha":        0.7,
    "xtick.color":       "#5c6bc0",
    "ytick.color":       "#5c6bc0",
    "text.color":        "#e8eaf6",
    "figure.edgecolor":  BG_DARK,
    "savefig.facecolor": BG_DARK,
    "legend.facecolor":  "#0f0f1a",
    "legend.edgecolor":  "#1e1e3a",
    "legend.labelcolor": "#e8eaf6",
})


# ── File parsing ───────────────────────────────────────────────────────────────
def load_spectrum(path: str):
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
                    if not counts:
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
        return np.arange(len(data)), data, None
    elif data.shape[1] >= 2:
        return data[:, 0], data[:, 1], None
    raise ValueError("Cannot interpret file columns.")


# ── Canvas ─────────────────────────────────────────────────────────────────────
class SpectrumCanvas(FigureCanvas):
    from PyQt5.QtCore import pyqtSignal
    crosshair_moved = pyqtSignal(float, float)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 5), tight_layout=True, facecolor=BG_DARK)
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {BG_DARK}; border-radius: 6px;")

        self._channels = self._counts = self._x_axis = None
        self._x_label    = "Channel"
        self._log_y      = False
        self._smooth     = False
        self._smooth_win = 5
        self._show_peaks = True
        self._hline = self._vline = None
        self._peaks = []

        self.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._draw_empty()

    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_facecolor("#06060e")
        self.ax.text(0.5, 0.5, "NO SPECTRUM LOADED",
                     transform=self.ax.transAxes, ha="center", va="center",
                     fontsize=16, color=BORDER, fontfamily="monospace",
                     fontweight="bold", alpha=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.draw()

    def plot_spectrum(self, channels, counts, energy=None,
                      log_y=False, smooth=False, smooth_win=5, show_peaks=True):
        self._channels   = channels
        self._counts     = counts
        self._log_y      = log_y
        self._smooth     = smooth
        self._smooth_win = smooth_win
        self._show_peaks = show_peaks
        if energy is not None:
            self._x_axis, self._x_label = energy, "Energy (keV)"
        else:
            self._x_axis, self._x_label = channels, "Channel"
        self._redraw()

    def _redraw(self):
        if self._counts is None:
            return
        self.ax.clear()
        self.ax.set_facecolor("#06060e")
        x = self._x_axis
        y = self._counts.copy()
        if self._smooth and self._smooth_win > 1:
            w = max(2, self._smooth_win | 1)
            y = np.convolve(y, np.ones(w) / w, mode="same")
        self.ax.fill_between(x, y, alpha=0.12, color=ACCENT_CYAN, linewidth=0)
        self.ax.step(x, y, where="mid", color=ACCENT_CYAN,
                     linewidth=0.9, alpha=0.95, label="Spectrum")
        if self._show_peaks and self._peaks:
            for px, py, label in self._peaks:
                self.ax.axvline(px, color=ACCENT_AMBER, linewidth=0.8,
                                linestyle="--", alpha=0.7)
                self.ax.text(px, py * 1.05, label, color=ACCENT_AMBER,
                             fontsize=7, ha="center", rotation=45, alpha=0.9)
        self.ax.set_yscale("log" if self._log_y else "linear")
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
        self._peaks = peaks
        self._redraw()

    def clear_vline(self):
        self._vline = None
        self._redraw()


# ── Main window ────────────────────────────────────────────────────────────────
class GammaViewer(QMainWindow):

    def __init__(self):
        super().__init__()

        uic.loadUi(UI_FILE, self)

        self._post_load_fixups()
        self._wire_signals()

        self._channels = self._counts = self._energy = None
        self._peaks    = []

        self._set_status("Ready  ·  Load a spectrum file to begin")

    # ── Post-load setup ────────────────────────────────────────────────────────
    def _post_load_fixups(self):
        # Dark window palette
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(BG_DARK))
        self.setPalette(pal)

        lay = self.canvas_wrap.layout()

        # Replace canvas placeholder with real SpectrumCanvas
        real_canvas = SpectrumCanvas()
        lay.replaceWidget(self.canvas, real_canvas)
        self.canvas.deleteLater()
        self.canvas = real_canvas

        # Replace nav placeholder with the real NavigationToolbar2QT
        real_nav = NavToolbar(self.canvas, self)
        real_nav.setStyleSheet(
            f"QToolBar {{ background: {BG_PANEL}; border: 1px solid {BORDER};"
            f"            border-radius: 4px; padding: 2px; }}"
            f"QToolButton {{ color: {TEXT_DIM}; background: transparent;"
            f"               border: none; padding: 3px 6px; }}"
            f"QToolButton:hover {{ color: {ACCENT_CYAN}; }}"
        )
        lay.replaceWidget(self.nav_toolbar, real_nav)
        self.nav_toolbar.deleteLater()
        self.nav_toolbar = real_nav

        # Pure QWidget containers don't paint their background from a stylesheet
        # unless WA_StyledBackground is set explicitly.
        for w in self.findChildren(QWidget):
            w.setAttribute(Qt.WA_StyledBackground, True)

        # Fix objectNames so the stylesheet selectors apply correctly
        for w in (self.section_cursor, self.section_spectrum,
                  self.section_display, self.section_peaks, self.section_export):
            w.setObjectName("section_header")
        for w in (self.sep0, self.sep1, self.sep2, self.sep3, self.sep4):
            w.setObjectName("separator")
        self.btn_find.setObjectName("primary_btn")
        self.btn_clear.setObjectName("danger_btn")

        # Re-polish widgets whose objectName changed
        for w in (self.btn_find, self.btn_clear):
            w.style().unpolish(w)
            w.style().polish(w)

        # Permanent file-name label in status bar
        self._file_label = QLabel("")
        self._file_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; padding-right: 8px;")
        self.statusBar.addPermanentWidget(self._file_label)

    def _wire_signals(self):
        # Toolbar actions
        self.actionOpen.triggered.connect(self._open_file)
        self.actionDemo.triggered.connect(self._load_demo)
        self.actionReset.triggered.connect(self._reset_zoom)

        # Side panel controls
        self.cb_log.toggled.connect(self._on_log)
        self.cb_smooth.toggled.connect(self._on_smooth)
        self.cb_smooth.toggled.connect(self.smooth_spin.setEnabled)
        self.smooth_spin.valueChanged.connect(self._on_smooth_win)
        self.cb_peaks.toggled.connect(self._on_peaks_toggle)
        self.btn_find.clicked.connect(self._auto_peaks)
        self.btn_clear.clicked.connect(self._clear_peaks)
        self.btn_export.clicked.connect(self._export)

        # Canvas crosshair
        self.canvas.crosshair_moved.connect(self._on_cursor)

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
        np.random.seed(42)
        n   = 4096
        ch  = np.arange(n)
        bg  = 50 * np.exp(-ch / 2000) + 10 + np.random.poisson(8, n)
        cal = 0.345
        counts = bg.astype(float)
        for pk_kev, amp, sig in [
            (244, 3000, 20), (344, 9000, 18), (411, 1000, 22), (444,  800, 25),
            (779, 2500, 20), (964, 1500, 22), (1112, 3500, 24), (1408, 4500, 26),
        ]:
            pk_ch = int(pk_kev / cal)
            counts += amp * np.exp(-0.5 * ((ch - pk_ch) / sig) ** 2)
        counts = np.random.poisson(counts).astype(float)
        self._load_data(ch, counts, cal * ch, label="DEMO  Eu-152 (synthetic)")

    def _load_data(self, channels, counts, energy, label=""):
        self._channels = channels
        self._counts   = counts
        self._energy   = energy
        self._peaks    = []
        self.canvas.plot_spectrum(
            channels, counts, energy,
            log_y=self.cb_log.isChecked(),
            smooth=self.cb_smooth.isChecked(),
            smooth_win=self.smooth_spin.value(),
            show_peaks=self.cb_peaks.isChecked(),
        )
        x_axis  = energy if energy is not None else channels
        x_label = "Energy (keV)" if energy is not None else "Channel"
        self._update_stats(channels, counts, x_axis, x_label)
        self._file_label.setText(label)
        self._set_status(
            f"Loaded  ·  {len(counts):,} channels  ·  {int(counts.sum()):,} total counts")

    # ── Stat display ───────────────────────────────────────────────────────────
    def _update_stats(self, channels, counts, x_axis, x_label):
        unit = "keV" if "keV" in x_label else "ch"
        pk   = x_axis[int(np.argmax(counts))]
        self.lbl_channels_val.setText(f"{len(counts):,}")
        self.lbl_max_val.setText(f"{int(counts.max()):,}")
        self.lbl_total_val.setText(f"{int(counts.sum()):,}")
        self.lbl_peak_val.setText(f"{pk:.1f} {unit}")

    def _on_cursor(self, x, y):
        self.lbl_cursor_x_val.setText(f"{x:.2f}")
        self.lbl_cursor_y_val.setText(f"{int(y):,}")

    # ── Controls ───────────────────────────────────────────────────────────────
    def _on_log(self, state):
        self.canvas.set_log_y(state)

    def _on_smooth(self, state):
        self.canvas.set_smooth(state, self.smooth_spin.value())

    def _on_smooth_win(self, val):
        if self.cb_smooth.isChecked():
            self.canvas.set_smooth(True, val)

    def _on_peaks_toggle(self, state):
        self.canvas.plot_spectrum(
            self._channels, self._counts, self._energy,
            log_y=self.cb_log.isChecked(),
            smooth=self.cb_smooth.isChecked(),
            smooth_win=self.smooth_spin.value(),
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
        y        = self._counts.copy()
        height   = y.max() * 0.03
        distance = max(5, len(y) // 100)
        idxs, _ = find_peaks(y, height=height, distance=distance,
                              prominence=height * 0.5)
        x_axis = self._energy if self._energy is not None else self._channels
        self._peaks = [(x_axis[i], y[i], f"{x_axis[i]:.1f}") for i in idxs]
        self.canvas.set_peaks(self._peaks)
        self.cb_peaks.setChecked(True)
        self._set_status(f"Found {len(idxs)} peaks")

    def _clear_peaks(self):
        self._peaks = []
        self.canvas.set_peaks([])
        self._set_status("Peaks cleared")

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
        self.statusBar.showMessage(f"  {msg}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Gamma Spectrum Viewer")
    win = GammaViewer()
    win.show()
    sys.exit(app.exec_())
