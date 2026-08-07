"""Fast Filter / CFD overlays on the GUI's random-trace sampler.

The API diagnostics dialog (API tab -> "Diagnostics...", **Binary** tab,
**Random traces** view) can overlay the two filters the Pixie-16 computes
internally but never stores -- the trapezoidal fast filter and the 500 MHz CFD
trace -- reconstructed offline by :mod:`wara.pixie_trace_analysis` from each
sampled trace and the run's DSP settings. Each gets its own y-axis, with a
dashed line at the channel's ``FastThresh`` / ``CFDThresh`` register.

Normally you tick the two checkboxes by hand; this script does it in code so
the whole path -- load, sample, overlay -- runs unattended. The run below needs
its folder reachable through ``data-path.txt``; swap DATE/RUN/CHANNEL for one
of your own.
"""
import sys

from PyQt5.QtWidgets import QApplication

from wara import pixie_trace_analysis as pta
from wara.gui.app import WaraApp
from wara.gui.api_diagnostics import DiagnosticsDialog

DATE, RUN, CHANNEL = "2025-04-18", 8, 7
SOURCE = "Trace data"   # or "Binary data" for a run whose traces landed there

app = QApplication.instance() or QApplication(sys.argv)
win = WaraApp()
dlg = DiagnosticsDialog(win.api)

# ── load the run's list-mode data ────────────────────────────────────────────
dlg.ed_bin_date.setText(DATE)
dlg.ed_bin_run.setText(str(RUN))
dlg.cmb_bin_type.setCurrentText(SOURCE)
dlg._load_bin()
print(dlg.lbl_bin_state.text())

# The overlays need FastLength / FastGap / FastThresh / CFDThresh from the run's
# settings file; without it the two checkboxes stay disabled.
if not dlg._bin_fast:
    raise SystemExit("No DSP settings for this run -- overlays unavailable")
for ch, (FL, FG, fast_thresh, cfd_thresh) in sorted(dlg._bin_fast.items()):
    print(f"  ch{ch}: FL={FL} FG={FG} FastThresh={fast_thresh} "
          f"CFDThresh={cfd_thresh}")

# ── show one channel, sample a few traces, overlay both filters ──────────────
for ch in dlg._bin_chans:
    dlg._bin_ch_visible[ch] = (ch == CHANNEL)
dlg._update_bin_ch_df()

dlg.ed_bin_ntraces.setText("5")
dlg.cb_bin_ff.setChecked(True)    # "Fast Filter"
dlg.cb_bin_cfd.setChecked(True)   # "CFD"

# Stock 500 MHz firmware fixes the CFD weight at w=1; our custom firmware runs
# it at w=0.3125. The "Custom firmware (w=0.3125)" checkbox picks between them
# and is on by default -- uncheck it for a standard Pixie-16 where w=1 holds.
dlg.cb_bin_cfd_custom_w.setChecked(True)
print(f"CFD weight: w={pta.CFD_W_CUSTOM if dlg.cb_bin_cfd_custom_w.isChecked() else pta.CFD_W:g}")

dlg._sample_bin_traces()
print(dlg.lbl_bin_state.text())

# ── restrict the random sample to a highlighted energy region ────────────────
# Dragging a span on the Energy view highlights an energy window; the sampler
# then draws only from it. Here we set one in code (roughly the middle third of
# the loaded energy range) and resample.
if dlg.df_bin_ch is not None and "energy" in dlg.df_bin_ch.columns:
    e = dlg.df_bin_ch["energy"]
    lo, hi = e.quantile(0.33), e.quantile(0.66)
    dlg._on_bin_span(float(lo), float(hi))
    dlg._sample_bin_traces()
    print(dlg.lbl_bin_state.text())

# ── the colour-coded data-table view ─────────────────────────────────────────
# The Binary tab's "Data table" sub-tab shades each numeric column low-to-high
# so patterns pop out. It shows the first 500 rows by default -- raise "Rows"
# for more. Jump to it and (re)fill.
for i in range(dlg.bin_plot_tabs.count()):
    if dlg.bin_plot_tabs.tabText(i) == "Data table":
        dlg.bin_plot_tabs.setCurrentIndex(i)
        break
dlg._refresh_bin_table()
print(dlg.lbl_bin_state.text())

dlg.show()
sys.exit(app.exec_())
