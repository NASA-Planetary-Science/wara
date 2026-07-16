"""Combine-runs dialog for the API tab (``CombineRunsDialog``)."""
import shutil
import traceback
from datetime import datetime

import numpy as np

from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSizePolicy, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt, QSize

from wara import read_parquet_api, combine_runs as cr

from . import theme as T
from .widgets import hsep, header, labeled_row
from .api_common import (
    API_PLOT_BG, DEFAULT_EBINS, DEFAULT_TBINS, _draw_axes_placeholder,
)


class CombineRunsDialog(QDialog):
    """Visualize and stitch several API runs (any dates) into one.

    The Energy and Time tabs overlay the per-run spectra for the selected channel
    so you can compare the runs before combining; "Combine & save" simply
    concatenates the runs (in table order) and writes the result as a new run
    (settings copied from the first run + provenance README). No drift correction
    is applied here  -- if the combined run needs gain/time alignment, do it
    afterwards with the single-run Shifts window (combine first, then shift)."""

    def __init__(self, controller):
        super().__init__(controller.app)
        self.c = controller
        self.setWindowTitle("Combine multiple runs")
        self.setStyleSheet(T.STYLESHEET)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.resize(1100, 720)

        self.runs_data = None        # list of (date, runnr, df) once loaded
        self.channels = []           # channels common to all loaded runs
        self._data_path = None       # data path of the first seeded/loaded run

        self.ax_raw = {}
        self.canvas = {}
        self.toolbar = {}
        self.fig = {}

        root = QHBoxLayout(self); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)

        # Plot side: Energy / Time tabs, each raw-over-aligned.
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_plot_tab("energy"), "Energy")
        self.tabs.addTab(self._build_plot_tab("time"), "Time")
        root.addWidget(self.tabs, 1)

        # Control side.
        root.addWidget(self._build_controls(), 0)

    # ── construction ──────────────────────────────────────────────────────────
    def _build_plot_tab(self, kind):
        tab = QWidget()
        col = QVBoxLayout(tab); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(2)
        fig = Figure(figsize=(6, 6), facecolor=API_PLOT_BG)
        self.fig[kind] = fig
        self.ax_raw[kind] = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumWidth(540)
        self.canvas[kind] = canvas
        toolbar = NavToolbar(canvas, self)
        toolbar.setObjectName("plot_toolbar")
        toolbar.setIconSize(QSize(22, 22))
        T.recolor_toolbar_icons(toolbar, T.TEXT_PRIMARY)
        self.toolbar[kind] = toolbar
        col.addWidget(toolbar)
        col.addWidget(canvas, 1)
        self._draw_placeholder(self.ax_raw[kind], "Add runs and press Load runs",
                               self._xlabel(kind), self._yscale(kind))
        return tab

    def _build_controls(self):
        side = QVBoxLayout(); side.setSpacing(6)
        side.addWidget(header("RUNS TO COMBINE"))
        note = QLabel("Runs are concatenated in table order (dates may differ). "
                      "No shifting is applied  -- align the combined run later "
                      "with the Shifts window if needed.")
        note.setObjectName("stat_key"); note.setWordWrap(True)
        side.addWidget(note)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Date", "Run"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(150)
        side.addWidget(self.table)

        rowbtns = QHBoxLayout(); rowbtns.setContentsMargins(0, 0, 0, 0)
        btn_add = QPushButton("+ Add row"); btn_add.setObjectName("mini_btn")
        btn_add.setCursor(Qt.PointingHandCursor)
        btn_add.clicked.connect(lambda: self._add_row())
        btn_del = QPushButton("Remove selected"); btn_del.setObjectName("mini_btn")
        btn_del.setCursor(Qt.PointingHandCursor)
        btn_del.clicked.connect(self._remove_selected)
        rowbtns.addWidget(btn_add); rowbtns.addWidget(btn_del)
        rw = QWidget(); rw.setLayout(rowbtns); side.addWidget(rw)

        btn_load = QPushButton("Load runs"); btn_load.setObjectName("primary_btn")
        btn_load.setCursor(Qt.PointingHandCursor)
        btn_load.clicked.connect(self._load_runs)
        side.addWidget(btn_load)

        side.addWidget(hsep()); side.addWidget(header("VISUALIZATION"))
        self.cb_channel = QComboBox()
        self.cb_channel.setToolTip("Channel shown in the overlay plots")
        self.cb_channel.currentIndexChanged.connect(lambda *_: self._preview_all())
        r, _ = labeled_row("Preview ch", self.cb_channel); side.addWidget(r)

        self.cb_ecol = QComboBox()
        self.cb_ecol.setToolTip(
            "Energy column used for the overlay and carried into the combined "
            "run. Runs missing it get it filled by copying an available energy "
            "column (see the warning on load).")
        self.cb_ecol.currentIndexChanged.connect(lambda *_: self._preview("energy"))
        r, _ = labeled_row("Energy column", self.cb_ecol); side.addWidget(r)

        self.ed_ebins = QLineEdit(str(DEFAULT_EBINS)); self.ed_ebins.setFixedWidth(70)
        self.ed_ebins.setToolTip("Number of energy bins in the overlay plot")
        r, _ = labeled_row("Energy bins", self.ed_ebins); side.addWidget(r)
        self.ed_tbins = QLineEdit(str(DEFAULT_TBINS)); self.ed_tbins.setFixedWidth(70)
        self.ed_tbins.setToolTip("Number of time (dt) bins in the overlay plot")
        r, _ = labeled_row("Time bins", self.ed_tbins); side.addWidget(r)

        btn_prev = QPushButton("Preview"); btn_prev.setObjectName("action_btn")
        btn_prev.setCursor(Qt.PointingHandCursor)
        btn_prev.clicked.connect(self._preview_all)
        side.addWidget(btn_prev)

        side.addWidget(hsep()); side.addWidget(header("SAVE COMBINED RUN"))
        self.ed_date = QLineEdit(); self.ed_date.setFixedWidth(150)
        r, _ = labeled_row("Date", self.ed_date); side.addWidget(r)
        self.ed_run = QLineEdit(); self.ed_run.setFixedWidth(150)
        self.ed_run.setPlaceholderText("new run number")
        r, _ = labeled_row("Run", self.ed_run); side.addWidget(r)

        btn_save = QPushButton("Combine && save"); btn_save.setObjectName("open_btn")
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.clicked.connect(self._combine_save)
        side.addWidget(btn_save)

        self.lbl_state = QLabel(""); self.lbl_state.setObjectName("stat_key")
        self.lbl_state.setWordWrap(True); side.addWidget(self.lbl_state)
        side.addStretch(1)

        holder = QWidget(); holder.setFixedWidth(300); holder.setLayout(side)
        return holder

    # ── runs table ──────────────────────────────────────────────────────────
    def _add_row(self, date="", runnr=""):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(date)))
        self.table.setItem(r, 1, QTableWidgetItem(str(runnr)))

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _table_runs(self):
        """Parse the table into a list of (date, runnr); skips blank rows and
        returns None on a bad run number (with a status message)."""
        runs = []
        for r in range(self.table.rowCount()):
            d_item = self.table.item(r, 0)
            n_item = self.table.item(r, 1)
            date = d_item.text().strip() if d_item else ""
            ntxt = n_item.text().strip() if n_item else ""
            if not date and not ntxt:
                continue
            try:
                runnr = int(ntxt)
            except ValueError:
                self.lbl_state.setText(f"Row {r + 1}: run number must be an integer")
                return None
            runs.append((date, runnr))
        return runs

    def seed(self, date, runnr, data_path):
        """Pre-fill the first run (the one currently open in the API tab) when the
        table is still empty, so the common 'this run plus a few more' flow starts
        ready to go."""
        self._data_path = data_path
        if self.table.rowCount() == 0:
            self._add_row(date, runnr)
            self._add_row()  # one blank row ready for the next run
        if not self.ed_date.text().strip():
            self.ed_date.setText(str(date or ""))

    # ── load / preview ────────────────────────────────────────────────────────
    def _load_runs(self):
        runs = self._table_runs()
        if runs is None:
            return
        if len(runs) < 2:
            self.lbl_state.setText("Add at least two runs to combine")
            return
        self.lbl_state.setText("Loading runs...")
        try:
            self.runs_data = cr.read_runs(runs, data_path=self._data_path)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.lbl_state.setText(f"Could not load runs: {exc}")
            self.runs_data = None
            return
        # Harmonize energy columns across runs so the overlay and the combined
        # run share one schema (older runs may lack "energy_orig"); missing
        # columns are filled by copying an available energy column.
        self.runs_data, patched = cr.harmonize_energy_columns(self.runs_data)
        self.channels = cr.channels_in_common(self.runs_data)
        prev = self.cb_channel.currentText()  # keep the last preview channel
        self.cb_channel.blockSignals(True)
        self.cb_channel.clear()
        self.cb_channel.addItems([str(ch) for ch in self.channels])
        if prev and prev in {str(ch) for ch in self.channels}:
            self.cb_channel.setCurrentText(prev)
        self.cb_channel.blockSignals(False)

        ecols = cr.energy_columns(self.runs_data[0][2]) if self.runs_data else []
        prev_e = self.cb_ecol.currentText()  # keep the last energy column
        self.cb_ecol.blockSignals(True)
        self.cb_ecol.clear()
        self.cb_ecol.addItems(ecols)
        if prev_e and prev_e in ecols:
            self.cb_ecol.setCurrentText(prev_e)
        elif "energy_orig" in ecols:
            self.cb_ecol.setCurrentText("energy_orig")
        self.cb_ecol.blockSignals(False)

        n = sum(len(df) for _, _, df in self.runs_data)
        chan_txt = (', '.join(str(c) for c in self.channels)
                    if self.channels else "none shared (preview off)")
        self.lbl_state.setText(
            f"Loaded {len(self.runs_data)} runs · channels {chan_txt} "
            f"· {n:,} events")
        if patched:
            self._warn_patched_energy(patched)
        self._preview_all()

    def _warn_patched_energy(self, patched):
        """Warn that some runs were missing an energy column, which was filled by
        copying an available one so the combine has a consistent schema."""
        lines = [f"{run}: {', '.join(f'{c} ← copied from {src}' for c, src in created.items())}"
                 for run, created in patched.items()]
        detail = "\n".join(lines)
        self.lbl_state.setText(
            "⚠ some runs were missing an energy column -- filled by copying "
            f"({'; '.join(patched)}). See the popup for details.")
        QMessageBox.warning(
            self, "Energy columns harmonized",
            "Some runs did not carry every energy column, so the missing column "
            "was created by copying an available one to keep the combined run's "
            "schema consistent (needed for post-combine shifts):\n\n"
            f"{detail}\n\n"
            "The copied column is a stand-in, not an independent measurement -- "
            "check that using it for calibration/shifts is appropriate.")

    def _preview_channel(self):
        try:
            return self.channels[self.cb_channel.currentIndex()]
        except (IndexError, ValueError):
            return None

    def _bins(self, kind):
        ed = self.ed_ebins if kind == "energy" else self.ed_tbins
        default = DEFAULT_EBINS if kind == "energy" else DEFAULT_TBINS
        try:
            return max(2, int(float(ed.text().strip())))
        except ValueError:
            return default

    def _preview_all(self):
        if self.runs_data is None:
            return
        for kind in ("energy", "time"):
            self._preview(kind)

    def _preview(self, kind):
        ch = self._preview_channel()
        if self.runs_data is None or ch is None:
            return
        ecol = self.cb_ecol.currentText() or None
        try:
            spectra = cr.run_spectra(self.runs_data, ch, kind,
                                     bins=self._bins(kind), ecol=ecol)
        except Exception as exc:  # noqa: BLE001
            self.lbl_state.setText(f"Preview failed: {exc}")
            return
        labels = [f"{d}-{r}" for d, r, _ in self.runs_data]
        self._draw_overlay(self.ax_raw[kind], spectra, labels,
                           f"Runs overlay  -- ch {ch}",
                           self._xlabel(kind), self._yscale(kind))
        self.fig[kind].tight_layout()
        self.canvas[kind].draw_idle()
        self.toolbar[kind].update()

    # ── drawing ───────────────────────────────────────────────────────────────
    @staticmethod
    def _xlabel(kind):
        return "Raw channel" if kind == "energy" else "dt (ns)"

    @staticmethod
    def _yscale(kind):
        return "log" if kind == "energy" else "linear"

    def _style_axis(self, ax, xlabel, yscale):
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, color=T.TEXT_DIM, fontsize=12)
        ax.set_facecolor(API_PLOT_BG)
        ax.tick_params(colors=T.TEXT_DIM, which="both", length=3, labelsize=11)
        for sp_ in ax.spines.values():
            sp_.set_color(T.BORDER)

    def _draw_placeholder(self, ax, message, xlabel, yscale="log"):
        _draw_axes_placeholder(ax, message, xlabel, self._style_axis, yscale)

    def _draw_overlay(self, ax, spectra, labels, title, xlabel, yscale="log"):
        ax.clear()
        # Start above viridis's dark-blue end so the first (reference) run reads
        # as a light teal that stands out against the dark plot background.
        n = max(len(spectra), 1)
        colors = cm.viridis(np.linspace(0.35, 1.0, n))
        for i, spe in enumerate(spectra):
            ax.step(spe.energies, spe.counts, where="mid", color=colors[i],
                    lw=0.9, alpha=0.85, label=labels[i] if i < len(labels) else None)
        ax.set_title(title, color=T.TEXT_PRIMARY, fontsize=12)
        self._style_axis(ax, xlabel, yscale)
        ax.legend(loc="upper right", fontsize=9, facecolor=API_PLOT_BG,
                  edgecolor=T.BORDER, labelcolor=T.TEXT_DIM, framealpha=0.7,
                  ncol=2 if len(spectra) > 4 else 1)

    # ── combine & save ──────────────────────────────────────────────────────
    def _combine_save(self):
        if self.runs_data is None:
            self.lbl_state.setText("Load the runs first")
            return
        new_date = self.ed_date.text().strip()
        if not new_date:
            self.lbl_state.setText("Enter a date for the combined run")
            return
        try:
            new_runnr = int(self.ed_run.text().strip())
        except ValueError:
            self.lbl_state.setText("New run number must be an integer")
            return

        try:
            run_dir, _, _ = read_parquet_api.run_parquet_path(
                new_date, new_runnr, self._data_path)
        except Exception as exc:  # noqa: BLE001
            self.lbl_state.setText(f"Bad destination date/run: {exc}")
            return
        if run_dir.exists():
            resp = QMessageBox.question(
                self, "Overwrite run?",
                f"Run {new_date}-{new_runnr} already exists at:\n{run_dir}\n\n"
                "Overwrite its combined parquet data?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if resp != QMessageBox.Yes:
                self.lbl_state.setText("Combine cancelled")
                return

        self.lbl_state.setText("Combining...")
        try:
            combined, info = cr.combine_runs(self.runs_data)
            out = read_parquet_api.save_combined_run(
                combined, new_date, new_runnr, self._data_path, overwrite=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.lbl_state.setText(f"Could not combine/save: {exc}")
            return

        extra = self._write_metadata(run_dir, new_date, new_runnr, info)
        msg = (f"Saved combined run {new_date}-{new_runnr} "
               f"({info['n_events']:,} events from {len(info['sources'])} runs) "
               f"→ {out}")
        if extra:
            msg += f"  ·  {extra}"
        self.lbl_state.setText(msg)
        self.c._status(msg)

        # Combining invalidates any per-run calibration; warn that the dropped
        # columns mean the combined run must be re-calibrated before analysis.
        if info["dropped_cal"]:
            QMessageBox.information(
                self, "Combined run is uncalibrated",
                f"Saved combined run {new_date}-{new_runnr}.\n\n"
                f"The source runs carried calibration columns "
                f"({', '.join(info['dropped_cal'])}) which were removed  -- "
                "combining runs invalidates them.\n\n"
                "Re-calibrate the energy (and re-align the time, if needed) on "
                "the combined run before any quantitative analysis.")

    def _write_metadata(self, dst_run_dir, new_date, new_runnr, info):
        """Merge *every* source run's settings folder into the combined run and
        write a provenance README listing all sources. Best-effort; returns a
        short note.

        Each source run's ``settings/*-stats-*`` files carry that run's live-time
        and counts, and the combined-run totals (live time, alphas, neutron
        yield) are computed downstream by summing across all stats files in the
        folder. So the combined run must hold the stats files from *all* sources,
        not just the first  -- otherwise those totals reflect one run only. The
        files are uniquely named per run (``RUN-<date>-<runnr>-...``), so merging
        them into one folder is collision-free."""
        notes = []
        dst_settings = dst_run_dir / "settings"
        copied, no_settings, failed = [], [], []
        for (d, r, _) in self.runs_data:
            try:
                src_run_dir, _, _ = read_parquet_api.run_parquet_path(
                    d, r, self._data_path)
            except Exception:  # noqa: BLE001
                failed.append(f"{d}-{r}")
                continue
            src_settings = src_run_dir / "settings"
            if not src_settings.is_dir():
                no_settings.append(f"{d}-{r}")
                continue
            try:
                shutil.copytree(src_settings, dst_settings, dirs_exist_ok=True)
                copied.append(f"{d}-{r}")
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                failed.append(f"{d}-{r}")
        if copied:
            notes.append(f"settings merged from {len(copied)}/"
                         f"{len(self.runs_data)} runs")
        if no_settings:
            notes.append(f"{len(no_settings)} run(s) had no settings")
        if failed:
            notes.append(f"{len(failed)} settings copy failed")

        try:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                "This run was combined (concatenated) from multiple source runs "
                "by the wara API tab. No drift correction was applied.",
                "",
                f"Combined on : {stamp}",
                f"New run     : {new_date}-{new_runnr}",
                f"Total events: {info['n_events']:,}",
                "",
                "Source runs (in combine order):",
            ]
            for d, r, n in info["sources"]:
                lines.append(f"  - {d}-{r}  ({n:,} events)")
            if info["dropped_cal"]:
                lines += [
                    "",
                    f"Calibration columns removed: {', '.join(info['dropped_cal'])}"
                    "  -- combining invalidates per-run calibration. "
                    "RE-CALIBRATE the energy (and re-align time, if needed) on "
                    "this combined run before analysis.",
                ]
            lines += [
                "",
                "The settings/ folder holds the *-stats-* files from ALL source "
                "runs (uniquely named per run), so live-time, total counts and "
                "neutron yield are summed across every run automatically.",
            ]
            if no_settings:
                lines.append(
                    "WARNING: no settings folder found for these runs, so their "
                    "live-time/counts are missing from the totals: "
                    + ", ".join(no_settings))
            if failed:
                lines.append(
                    "WARNING: settings copy failed for these runs (see console): "
                    + ", ".join(failed))
            (dst_run_dir / "README.txt").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
            notes.append("README written")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        return "; ".join(notes)
