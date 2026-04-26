import random
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from wara import tlist
from wara import file_reader
from wara import spectrum as sp
from wara import decay_exponential as decay


class PngMixin:

    def png_info_activate(self):
        self.w_info_png.activateWindow()
        self.w_info_png.show()

    def initialize_plots_png(self):
        self.fig_png_dt = self.plot_dt_png.canvas.figure
        self.fig_png_e = self.plot_e_png.canvas.figure
        self.fig_png_decay = self.plot_decay_png.canvas.figure
        self.fig_png_et = self.plot_EvsT_png.canvas.figure
        self.fig_png_dt.clear()
        self.fig_png_e.clear()
        self.fig_png_decay.clear()
        self.fig_png_et.clear()
        self.fig_png_dt.set_constrained_layout(True)
        self.fig_png_e.set_constrained_layout(True)
        self.fig_png_decay.set_constrained_layout(True)
        self.fig_png_et.set_constrained_layout(True)
        self.fig_png_dt.patch.set_alpha(0)
        self.fig_png_e.patch.set_alpha(0)
        self.fig_png_decay.patch.set_alpha(0)
        self.fig_png_et.patch.set_alpha(0)
        self.ax_png_dt = self.fig_png_dt.add_subplot()
        self.ax_png_e = self.fig_png_e.add_subplot()
        self.ax_png_decay = self.fig_png_decay.add_subplot()
        self.ax_png_et = self.fig_png_et.add_subplot()
        self.fig_png_dt.canvas.draw_idle()
        self.fig_png_e.canvas.draw_idle()
        self.fig_png_decay.canvas.draw_idle()
        self.fig_png_et.canvas.draw_idle()
        plt.rcParams.update({"font.size": 18})

    def load_file_png(self):
        try:
            fname = self.filePath_png.text()
            period = eval(self.period_png.text())
            if self.multiscancheck.isChecked():
                ms = file_reader.ReadMultiScanTlist(fname)
                ms.read_file()
                df_raw = ms.df
            elif self.caencheck.isChecked():
                caen = file_reader.ReadCaenListMode(fname)
                caen.parse_data()
                df_raw = pd.DataFrame()
                df_raw["channel"] = caen.df["channel"]
                df_raw["ts"] = caen.df["ts (ns)"]
            self.png = tlist.Tlist(df_raw, period=period)
            if self.isevaluable(self.tbins_png.text()):
                tbins_png = eval(self.tbins_png.text())
            else:
                tbins_png = 200
            if self.isevaluable(self.ebins_png.text()):
                ebins_png = eval(self.ebins_png.text())
            else:
                ebins_png = 2**12
            if self.pushButton_keep_png.isChecked():
                print("Keeping all figures")
            else:
                print("Resetting all figures")
                self.reset_png_plots()
            self.plot_png_dt()
            self.plot_png_e()
        except Exception as e:
            print("An unknown error occurred:", str(e))
            print(
                """ERROR: Could not load file. Make sure you are using the full
                path and input an integer for the period"""
            )

    def reset_png(self):
        self.initialize_plots_png()
        self.load_file_png()
        self.textBrowser_png.setText("")

    def png_spect_yscale(self):
        try:
            if self.png_spect_scale == "log":
                self.ax_png_e.set_yscale("linear")
                self.png_spect_scale = "linear"
            else:
                self.ax_png_e.set_yscale("log")
                self.png_spect_scale = "log"
            self.fig_png_e.canvas.draw_idle()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def reset_png_plots(self):
        self.ax_png_dt.clear()
        self.ax_png_e.clear()
        self.ax_png_decay.clear()
        self.ax_png_et.clear()

    def plot_png_dt(self):
        if self.pushButton_keep_png.isChecked() is False:
            self.ax_png_dt.clear()
        self.png.plot_time_hist(ax=self.ax_png_dt)
        self.fig_png_dt.canvas.draw_idle()

    def plot_png_e(self):
        if self.pushButton_keep_png.isChecked() is False:
            self.ax_png_e.clear()
        self.png.hist_erg()
        self.png.plot_spect_erg_all(self.png.x, self.png.spect, ax=self.ax_png_e)
        self.ax_png_e.set_yscale(self.png_spect_scale)
        self.fig_png_e.canvas.draw_idle()

    def plot_png_decay(self):
        if self.pushButton_keep_png.isChecked() is False:
            self.ax_png_decay.clear()
        self.png.plot_die_away(ax=self.ax_png_decay)
        self.fig_png_decay.canvas.draw_idle()

    def plot_png_fit_decay(self):
        if self.pushButton_keep_png.isChecked() is False:
            self.ax_png_decay.clear()
        if self.checkBox_single_png.isChecked():
            self.exp_png.fit_single_decay()
        elif self.checkBox_double_png.isChecked():
            self.exp_png.fit_double_decay()
        show_comps = False
        if self.pushButton_fit_comps_png.isChecked():
            show_comps = True
        self.exp_png.plot(ax_fit=self.ax_png_decay, show_components=show_comps)
        self.fig_png_decay.canvas.draw_idle()

    def plot_png_et(self):
        if self.pushButton_keep_png.isChecked() is False:
            self.ax_png_et.clear()
        self.png.plot_time_hist(ax=self.ax_png_et)
        self.fig_png_et.canvas.draw_idle()

    def change_tbins_png(self):
        try:
            tbins = eval(self.tbins_png.text())
            self.png.tbins = tbins
            self.plot_png_dt()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def change_ebins_png(self):
        try:
            ebins = eval(self.ebins_png.text())
            self.png.ebins = ebins
            self.plot_png_e()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def apply_trange_png(self):
        try:
            colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
            trange = [eval(self.t0_png.text()), eval(self.t1_png.text())]
            self.png.filter_tdata(trange=trange)
            self.plot_png_e()
            self.png.plot_vlines_t(ax=self.ax_png_dt, color=random.choice(colors))
            self.fig_png_dt.canvas.draw_idle()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def apply_erange_png(self):
        try:
            colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
            erange = [eval(self.e0_png.text()), eval(self.e1_png.text())]
            self.png.filter_edata(erange=erange)
            self.plot_png_et()
            self.png.plot_vlines_e(ax=self.ax_png_e, color=random.choice(colors))
            self.fig_png_e.canvas.draw_idle()
            self.fig_png_et.canvas.draw_idle()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def apply_die_away(self):
        try:
            t_start = eval(self.t_start_png.text())
            t_end = eval(self.t_end_png.text())
            self.png.filter_tdata(trange=[t_start, t_end], restore_df=False)
            self.png.hist_time()
            self.plot_png_decay()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def apply_die_away_fit(self):
        try:
            self.exp_png = decay.Decay_exp(
                x=self.png.xd, y=self.png.yd, yerr=self.png.yerrd
            )
            self.plot_png_fit_decay()
            fit_report = self.exp_png.fit_result.fit_report()
            self.textBrowser_png.setText(fit_report)
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def send_to_spect_png(self):
        try:
            self.reset_spect_params()
            self.spect = sp.Spectrum(counts=self.png.spect)
            self.e_units = "channels"
            self.e_units_orig = deepcopy(self.e_units)
            self.spect_orig = deepcopy(self.spect)
            self.create_graph(fit=False, reset=True)
            self.search = 0
        except Exception as e:
            print("An unknown error occurred:", str(e))
