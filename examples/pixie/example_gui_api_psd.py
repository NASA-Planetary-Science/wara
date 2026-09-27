"""Neutron/gamma PSD on the GUI's API tab ("Neutron run").

Loads an API run whose events carry a trace, ticks **Neutron run** (the PSD
panel appears under the X-Y map) and **Add alpha energy** (the spectra move to
the left column, the maps to the right), keeps only the neutron band with a
PSD cut, and opens the **PSD gates...** window. Normally you do all of this by
hand; the script just drives the same controls.

The run below needs its folder reachable through ``data-path.txt``.
"""
import sys

from PyQt5.QtWidgets import QApplication

from wara.gui.app import WaraApp
from wara.gui.theme import NAV_SECTIONS

DATE, RUN, CHANNEL = "2026-09-24", 1, 1      # EJ250 neutron detector

app = QApplication.instance() or QApplication(sys.argv)
win = WaraApp()
idx = [name for name, _ in NAV_SECTIONS].index("API")
win.nav_group.button(idx).setChecked(True)
win._on_nav(idx)
api = win.api

api.opts.ed_date.setText(DATE)
api.opts.ed_run.setText(str(RUN))
api.opts.ed_ch.setText(str(CHANNEL))
api._load()
api.opts.cb_psd.setChecked(True)       # "Neutron run": computes the PSD
api.opts.cb_alpha.setChecked(True)     # "Add alpha energy"
print(api.opts.lbl_psd.text())

# Keep the upper (neutron) band above 15k channels -- the same as dragging a
# box on the PSD panel with Interactive cuts on.
api.apply_psd_filter(15000, 65535, 0.405, 0.5)
print(f"{api.df_current.shape[0]:,} neutron events")

api._open_psd_gates()                  # tune the gates on aligned traces
win.show()
sys.exit(app.exec_())
