import numpy as np
import time
import tempfile
from copy import deepcopy
from scipy import ndimage
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QUrl
from matplotlib.widgets import SpanSelector, RectangleSelector
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from wara import read_parquet_api
from wara import helper_api
from wara import apicalc
from wara import spectrum as sp


class ApiMixin:

    def activate_load_api_file(self):
        try:
            self.load_api_file()
        except Exception as e:
            print("ERROR: Could not load data file")
            print("An unknown error occurred:", str(e))

    def load_api_file(self):
        self.reset_api_figure()
        self.dt_flag = 0
        self.xy_flag = 0
        self.en_flag = 0
        self.raw_flag = 0  # raw data flag
        self.calibrated_flag = 0  # time-shifted and calibrated data flag
        self.sims_flag = 0  # simulations data flag
        filepath = self.api_path_txt.text()
        if filepath == "":
            data_path = None
        else:
            data_path = filepath
        ch_txt = self.api_ch_txt.text()
        date = self.api_date_txt.text()
        runnr = int(self.api_run_txt.text())
        if self.checkBox_exp_raw.isChecked():
            self.raw_flag = 1
            print("Loading experimental data")
        elif self.checkBox_exp_cal.isChecked():
            self.calibrated_flag = 1
            print("Loading experimental data that has been calibrated and time-shifted")
        elif self.checkBox_sims.isChecked():
            self.sims_flag = 1
            print("Loading simulated data")
        if ch_txt == "6" or ch_txt == "7" or ch_txt == "10" or ch_txt == "11":
            self.ebins = 2**14
        else:
            self.ebins = 2**12
        if ch_txt == "9":
            ch = 9
            runnr = int(self.api_run_txt.text())
            self.df_api = read_parquet_api.read_parquet_file(
                date, runnr, ch, flood_field=True, data_path=data_path
            )
            FF = True
        else:
            FF = False
            ch = int(ch_txt)
            self.df_api = read_parquet_api.read_parquet_file(
                date=date, runnr=runnr, ch=ch, flood_field=False, data_path=data_path
            )
            self.df_api["dt"] *= 1e9  # to ns
        if self.df_api is None:
            QMessageBox.critical(
                None,
                "Error while opening parquet data",
                f"No parquet file available for run {date}-{runnr}.",
            )
        self.df_current = self.df_api.copy()
        self.df_previous = self.df_api.copy()
        self.initialize_plots_api(FF)
        self.api_textEdit.setText("")
        try:
            self.read_settings_file(date, runnr, ch, data_path)
        except Exception as e:
            print("ERROR: Settings file not found")
            print(e)

    def read_settings_file(self, date, runnr, ch, data_path):
        tot_time = apicalc.get_total_time(date, runnr, ch, data_path)
        tot_alphas = apicalc.get_total_counts(date, runnr, ch=9, data_path=data_path)
        neutron_yield = apicalc.calculate_neutron_yield(
            date, runnr, ch=9, data_path=data_path
        )
        alphas_txt = f"Total alphas: {tot_alphas:.3E}"
        time_txt = (
            f"Live time (HH:MM:SS): {time.strftime('%H:%M:%S', time.gmtime(tot_time))}"
        )
        nyield_txt = f"Neutron yield (n/s): {neutron_yield:.3E}"
        self.api_textEdit.setText(f"{alphas_txt}\n{time_txt}\n{nyield_txt}")

    def initialize_plots_api(self, flood_field=False):
        if self.sims_flag:
            self.api_xkey = "X"
            self.api_ykey = "Y"
            self.api_ekey = "energy"
            self.xyplane = (-0.2, 0.2, -0.2, 0.2)
            self.erange_api = [0, self.df_current["energy"].max()]  # in MeV
        elif self.calibrated_flag:
            self.df_current = apicalc.calc_own_pos(self.df_current)
            self.api_xkey = "X2"
            self.api_ykey = "Y2"
            self.api_ekey = "energy"
            self.xyplane = (-0.9, 0.9, -0.9, 0.9)  # x and y limits
            self.erange_api = [0, self.df_current["energy"].max()]
        elif self.raw_flag:
            self.df_current = apicalc.calc_own_pos(self.df_current)
            self.api_xkey = "X2"
            self.api_ykey = "Y2"
            self.api_ekey = "energy_orig"
            self.xyplane = (-0.9, 0.9, -0.9, 0.9)  # x and y limits
            self.erange_api = [0, 2**16]  # [0,8]
        else:
            print("ERROR: Cannot determine the data type (raw, calibrated, or simulated)")

        self.hexbins = 80  # x-y bins
        self.tbins = 512  # time bins
        self.colormap = "plasma"
        if flood_field:
            self.plot_xy_api(self.df_current)
        else:
            self.plot_energy_hist_api(self.df_current)
            self.plot_time_hist_api(self.df_current)
            self.plot_xy_api(self.df_current)
            self.espan_api()
            self.xyspan_api()
            self.tspan_api()
        self.fig_api.canvas.draw_idle()

    def reset_api_figure(self):
        self.fig_api = self.api_plot.canvas.figure
        self.fig_api.clear()
        self.fig_api.set_constrained_layout(True)
        gs = self.fig_api.add_gridspec(
            2, 2, width_ratios=[0.5, 0.5], height_ratios=[1, 1]
        )

        # axes
        self.ax_api_spe = self.fig_api.add_subplot(gs[0, 0])
        self.ax_api_dt = self.fig_api.add_subplot(gs[1, 0])

        self.ax_api_xy = self.fig_api.add_subplot(gs[:, 1])
        self.api_xy_scale = "linear"
        self.api_button_logxy.setChecked(False)
        self.fig_api.canvas.draw_idle()  # to delete

    def reset_button_api(self):
        try:
            self.reset_api_figure()
            self.dt_flag = 0
            self.xy_flag = 0
            self.en_flag = 0
            self.df_current = self.df_api.copy()
            self.df_previous = self.df_api.copy()
            self.initialize_plots_api()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def send_to_spect_api(self):
        try:
            self.reset_spect_params()
            if self.sims_flag or self.calibrated_flag:
                self.e_units = "MeV"
                self.spect = sp.Spectrum(
                    counts=self.api_gam, energies=self.api_gam_x, e_units=self.e_units
                )
            else:
                self.spect = sp.Spectrum(counts=self.api_gam)
                self.e_units = "channels"
            self.e_units_orig = deepcopy(self.e_units)
            self.spect_orig = deepcopy(self.spect)
            self.create_graph(fit=False, reset=True)
            self.search = 0
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def plot_energy_hist_api(self, df):
        self.api_gam, self.api_edg = np.histogram(
            df[self.api_ekey], bins=self.ebins, range=self.erange_api
        )
        self.api_gam_x = (self.api_edg[1:] + self.api_edg[:-1]) / 2
        self.ax_api_spe.plot(self.api_gam_x, self.api_gam, color="green")
        self.ax_api_spe.set_yscale(self.api_spect_scale)
        if self.sims_flag or self.calibrated_flag:
            self.ax_api_spe.set_xlabel("Energy (Mev)")
        else:
            self.ax_api_spe.set_xlabel("Channels")

    def plot_time_hist_api(self, df):
        low, high = np.percentile(df["dt"], [0.2, 99.5])
        df["dt"].plot.hist(
            bins=self.tbins,
            ax=self.ax_api_dt,
            range=[low, high],
            alpha=0.7,
            edgecolor="black",
        )
        self.ax_api_dt.set_xlabel("dt (ns)")

    def plot_xy_api(self, df, cbar=True, logxy=False, **kwargs):
        if logxy:
            kwargs["bins"] = "log"
        df.plot.hexbin(
            x=self.api_xkey,
            y=self.api_ykey,
            gridsize=self.hexbins,
            cmap=self.colormap,
            ax=self.ax_api_xy,
            colorbar=cbar,
            extent=self.xyplane,
            title=f"Total counts = {df.shape[0]}",
            **kwargs,
        )
        self.fig_api.canvas.draw_idle()

    def api_xylog(self):
        try:
            if self.api_xy_scale == "linear":
                self.plot_xy_api(self.df_current, cbar=False, logxy=True)
                self.api_xy_scale = "log"
            elif self.api_xy_scale == "log":
                self.plot_xy_api(self.df_current, cbar=False, logxy=False)
                self.api_xy_scale = "linear"
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def api_on_returnXY(self):
        try:
            self.vmax = int(self.api_vmax_txt.text())
            self.plot_xy_api(self.df_current, cbar=False, logxy=False, vmax=self.vmax)
            self.api_xy_scale = "linear"
            self.api_button_logxy.setChecked(False)
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def api_spect_yscale(self):
        try:
            if self.api_spect_scale == "log":
                self.ax_api_spe.set_yscale("linear")
                self.api_spect_scale = "linear"
            else:
                self.ax_api_spe.set_yscale("log")
                self.api_spect_scale = "log"
            self.fig_api.canvas.draw_idle()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def api_window_filters(self):
        from ..dialogs import WindowAPIfilters
        self.w_api_filt = WindowAPIfilters()
        self.w_api_filt.show()
        try:
            self.w_api_filt.button_api_apply.clicked.connect(self.apply_api_filters)
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def apply_api_filters(self):
        xmin_txt = self.w_api_filt.xmin_txt.text()
        xmax_txt = self.w_api_filt.xmax_txt.text()
        ymin_txt = self.w_api_filt.ymin_txt.text()
        ymax_txt = self.w_api_filt.ymax_txt.text()
        tmin_txt = self.w_api_filt.tmin_txt.text()
        tmax_txt = self.w_api_filt.tmax_txt.text()
        emin_txt = self.w_api_filt.emin_txt.text()
        emax_txt = self.w_api_filt.emax_txt.text()
        if (
            (xmin_txt != "")
            and (xmax_txt != "")
            and (ymin_txt != "")
            and (ymax_txt != "")
        ):
            xmin = float(xmin_txt)
            xmax = float(xmax_txt)
            ymin = float(ymin_txt)
            ymax = float(ymax_txt)
            self.apply_xy_filter(xmin, xmax, ymin, ymax)
        if (tmin_txt != "") and (tmax_txt != ""):
            tmin = float(tmin_txt)
            tmax = float(tmax_txt)
            self.apply_t_filter(tmin, tmax)
        if (emin_txt != "") and (emax_txt != ""):
            emin = int(emin_txt)
            emax = int(emax_txt)
            self.apply_energy_filter(emin, emax)

    def espan_api(self):
        self.span_api = SpanSelector(
            self.ax_api_spe,
            self.enselect_api,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.3, facecolor="yellow"),
        )

    def enselect_api(self, xmin, xmax):
        xmin = round(xmin, 4)
        xmax = round(xmax, 4)
        print("xmin: ", xmin)
        print("xmax: ", xmax)
        self.apply_energy_filter(xmin, xmax)

    def apply_energy_filter(self, xmin, xmax):
        self.ax_api_dt.clear()
        self.ax_api_xy.clear()
        if self.en_flag == 0:  # energy filter has not been used before
            elim = (self.df_current[self.api_ekey] > xmin) & (
                self.df_current[self.api_ekey] < xmax
            )
            self.df_previous = self.df_current.copy()
            self.df_current = self.df_current[elim]
            self.en_flag = 1  # set energy filter to used
        else:
            elim = (self.df_previous[self.api_ekey] > xmin) & (
                self.df_previous[self.api_ekey] < xmax
            )
            self.df_current = self.df_previous[elim]
        self.df_current.reset_index(drop=True, inplace=True)
        self.plot_time_hist_api(df=self.df_current)
        self.plot_xy_api(self.df_current, cbar=False)
        self.fig_api.canvas.draw_idle()

    def xyspan_api(self):
        self.toggle_selector = RectangleSelector(
            self.ax_api_xy,
            self.xyselect_api,
            useblit=True,
            button=[1, 3],  # don't use middle button
            minspanx=0,
            minspany=0,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="white", edgecolor="black", alpha=0.1, fill=True),
        )

    def xyselect_api(self, eclick, erelease):
        "eclick and erelease are the press and release events"
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        self.apply_xy_filter(x1, x2, y1, y2)

    def apply_xy_filter(self, x1, x2, y1, y2):
        self.ax_api_spe.clear()
        self.ax_api_dt.clear()
        self.ax_api_xy.clear()
        if self.xy_flag == 0:  # xy filter not used before
            xlim = (self.df_current[self.api_xkey] > x1) & (
                self.df_current[self.api_xkey] < x2
            )
            ylim = (self.df_current[self.api_ykey] > y1) & (
                self.df_current[self.api_ykey] < y2
            )
            self.df_previous = self.df_current.copy()
            self.df_current = self.df_current[xlim & ylim]
            self.xy_flag = 1  # set energy filter to used
        else:
            xlim = (self.df_previous[self.api_xkey] > x1) & (
                self.df_previous[self.api_xkey] < x2
            )
            ylim = (self.df_previous[self.api_ykey] > y1) & (
                self.df_previous[self.api_ykey] < y2
            )
            self.df_current = self.df_previous[xlim & ylim]
        self.df_current.fillna(0)
        self.df_current.reset_index(drop=True, inplace=True)
        self.plot_energy_hist_api(df=self.df_current)
        self.plot_time_hist_api(df=self.df_current)
        self.plot_xy_api(df=self.df_previous, cbar=False)
        self.fig_api.canvas.draw_idle()

    def tspan_api(self):
        self.tspan = SpanSelector(
            self.ax_api_dt,
            self.tselect_api,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.4, facecolor="yellow"),
        )

    def tselect_api(self, tmin, tmax):
        print("tmin: ", round(tmin, 3))
        print("tmax: ", round(tmax, 3))
        self.apply_t_filter(tmin, tmax)

    def apply_t_filter(self, tmin, tmax):
        self.ax_api_spe.clear()
        self.ax_api_xy.clear()
        if self.dt_flag == 0:
            tlim = (self.df_current["dt"] > tmin) & (self.df_current["dt"] < tmax)
            self.df_previous = self.df_current.copy()
            self.df_current = self.df_current[tlim]
            self.dt_flag = 1
        else:
            tlim = (self.df_previous["dt"] > tmin) & (self.df_previous["dt"] < tmax)
            self.df_current = self.df_previous[tlim]
        self.df_current.reset_index(drop=True, inplace=True)
        self.plot_energy_hist_api(df=self.df_current)
        self.plot_xy_api(self.df_current, cbar=False)
        self.fig_api.canvas.draw_idle()

    # API MCA
    ##TODO
    def open_api_mca(self):
        self.w_api_mca.activateWindow()
        self.w_api_mca.show()

    def activate_api_mca(self):
        try:
            self.load_api_mca()
        except Exception as e:
            print("ERROR: Could not load MCA file")
            print("An unknown error occurred:", str(e))

    def load_api_mca(self):
        self.initialize_plot_api_mca_all()
        date = self.w_api_mca.edit_date.text()
        runnr = int(self.w_api_mca.edit_run.text())
        print(date)
        print(runnr)
        self.mca_data = helper_api.read_mca(date=date, runnr=runnr)
        self.create_plot_mca_all()

    def initialize_plot_api_mca_all(self):
        self.fig_mca_all = self.w_api_mca.plot_mca_all.canvas.figure
        self.fig_mca_all.clear()
        self.fig_mca_all.set_constrained_layout(True)
        self.fig_mca_all.patch.set_alpha(0)
        self.ax_mca_all = self.fig_mca_all.add_subplot()
        self.fig_mca_all.canvas.draw_idle()
        plt.rcParams.update({"font.size": 18})

    def create_plot_mca_all(self):
        for i, row in enumerate(self.mca_data):
            self.ax_mca_all.plot(row, label=f"Spectrum #{i}")
        self.ax_mca_all.legend()
        self.ax_mca_all.set_xlabel("Channels")
        self.ax_mca_all.set_yscale("log")
        self.fig_mca_all.canvas.draw_idle()

    def activate_api_mca_select(self):
        try:
            self.load_api_mca_select()
        except Exception as e:
            print("ERROR: Could not obtain selected spectrum")
            print("An unknown error occurred:", str(e))

    def load_api_mca_select(self):
        self.initialize_plot_api_mca_select()
        self.spec_no = int(self.w_api_mca.edit_select.text())
        self.mca_data_select = self.mca_data[self.spec_no]
        self.create_plot_mca_select()

    def initialize_plot_api_mca_select(self):
        self.fig_mca_select = self.w_api_mca.plot_mca_select.canvas.figure
        self.fig_mca_select.clear()
        self.fig_mca_select.set_constrained_layout(True)
        self.fig_mca_select.patch.set_alpha(0)
        self.ax_mca_select = self.fig_mca_select.add_subplot()
        self.fig_mca_select.canvas.draw_idle()
        plt.rcParams.update({"font.size": 18})

    def create_plot_mca_select(self):
        self.ax_mca_select.plot(self.mca_data_select, label=f"Spectrum #{self.spec_no}")
        self.ax_mca_select.set_xlabel("Channels")
        self.ax_mca_select.legend()
        self.ax_mca_select.set_yscale("log")
        self.fig_mca_select.canvas.draw_idle()

    def reset_mca_plots(self):
        try:
            self.ax_mca_all.clear()
            self.ax_mca_select.clear()
            self.fig_mca_all.canvas.draw_idle()
            self.fig_mca_select.canvas.draw_idle()
        except Exception as e:
            print("ERROR: Could not reset MCA plots")
            print("An unknown error occurred:", str(e))

    def send_to_spect_mca(self):
        try:
            self.reset_spect_params()
            self.spect = sp.Spectrum(counts=self.mca_data_select)
            self.e_units = "channels"
            self.e_units_orig = deepcopy(self.e_units)
            self.spect_orig = deepcopy(self.spect)
            self.create_graph(fit=False, reset=True)
            self.search = 0
        except Exception as e:
            print("ERROR: Could not send to Spectrum tab")
            print("An unknown error occurred:", str(e))

    # API Binary
    # TODO
    def open_api_bin(self):
        self.w_api_bin.activateWindow()
        self.w_api_bin.show()

    def activate_api_bin(self):
        try:
            self.load_api_bin()
        except Exception as e:
            print("ERROR: Could not load file")
            print("An unknown error occurred:", str(e))

    def load_api_bin(self):
        self.initialize_plot_api_bin_erg()
        self.initialize_plot_api_bin_rand_traces()
        self.initialize_plot_api_bin_own_erg()
        self.initialize_plot_api_bin_erg_tr()
        date = self.w_api_bin.edit_date.text()
        runnr = int(self.w_api_bin.edit_run.text())
        print(date)
        print(runnr)
        if self.w_api_bin.checkBox_trace_data.isChecked():
            self.df_apibin = helper_api.read_trace_data(date=date, runnr=runnr)
        elif self.w_api_bin.checkBox_bin_data.isChecked():
            self.df_apibin = helper_api.read_binary_data(date=date, runnr=runnr)
        else:
            self.df_apibin = read_parquet_api.read_parquet_file(date=date, runnr=runnr)
        print(self.df_apibin.shape)
        self.df_apibin_ch = self.df_apibin
        self.plot_apibin_energy()

    def send_to_spect_api_bin1(self):
        try:
            self.reset_spect_params()
            self.spect = sp.Spectrum(
                energies=self.api_gam_x_bin, counts=self.spect_api_bin1
            )
            self.e_units = "channels"
            self.e_units_orig = deepcopy(self.e_units)
            self.spect_orig = deepcopy(self.spect)
            self.create_graph(fit=False, reset=True)
            self.search = 0
        except Exception as e:
            print("ERROR: Could not send spectrum")
            print("An unknown error occurred:", str(e))

    def send_to_spect_api_bin2(self):
        try:
            self.reset_spect_params()
            self.spect = sp.Spectrum(
                energies=self.api_gam_x_bin2, counts=self.spect_api_bin2
            )
            self.e_units = "channels"
            self.e_units_orig = deepcopy(self.e_units)
            self.spect_orig = deepcopy(self.spect)
            self.create_graph(fit=False, reset=True)
            self.search = 0
        except Exception as e:
            print("ERROR: Could not send spectrum")
            print("An unknown error occurred:", str(e))

    def update_yscale_api_bin(self):
        if self.yscale_api_bin == "log":
            self.ax_bin_erg.set_yscale("linear")
            self.yscale_api_bin = "linear"
        elif self.yscale_api_bin == "linear":
            self.ax_bin_erg.set_yscale("log")
            self.yscale_api_bin = "log"
        self.fig_bin_erg.canvas.draw_idle()

    def select_ch_api_bin(self):
        try:
            print(self.w_api_bin.edit_ch.text())
            ch = int(self.w_api_bin.edit_ch.text())
            self.df_apibin_ch = self.df_apibin[self.df_apibin["channel"] == ch]
            print(self.df_apibin_ch.shape)
            self.plot_apibin_energy()
        except Exception as e:
            print("ERROR: Could not read channel")
            print("An unknown error occurred:", str(e))

    def normalize_baseline_api_bin(self):
        self.ax_bin_erg_tr.clear()
        baseline = self.w_api_bin.edit_baseline_erg_tr.text()
        if baseline == "":
            baseline = 100
            self.w_api_bin.edit_baseline_erg_tr.setText(str(baseline))
        traces = self.df_bin_tr.trace.to_numpy()
        traces = np.vstack(traces)
        baseline_pts = int(int(baseline) / 2)  # in ns
        traces_norm = traces - traces[:, 0:baseline_pts].mean(axis=1).reshape(-1, 1)
        t = np.arange(0, self.df_bin_tr.trace.iloc[0].shape[0], 1) * 2
        for tr in traces_norm:
            self.ax_bin_erg_tr.plot(t, tr)
        self.ax_bin_erg_tr.set_xlabel("Time (ns)")
        self.ax_bin_erg_tr.set_ylabel("Amplitude (a.u)")
        self.fig_bin_erg_tr.canvas.draw_idle()

    def apply_fft(self):
        if self.w_api_bin.button_FFT.isChecked():
            try:
                self.calculate_fft()
                self.plot_fft()
            except Exception as e:
                print("ERROR: Could not calculate FFT")
                print("An unknown error occurred:", str(e))
        else:
            self.plot_apibin_rand_traces()

    def calculate_fft(self):
        self.ffts = []
        for tr in self.df_rand.trace:
            freqs, magnitude = apicalc.compute_fft(signal=tr, sampling_rate=500e6)
            self.ffts.append(magnitude)
        self.freqs = freqs

    def apply_rand_traces_api_bin(self):
        self.w_api_bin.button_FFT.setChecked(False)
        try:
            self.sample_rand_traces()
            self.plot_apibin_rand_traces()
        except Exception as e:
            print("ERROR: Could not plot random traces")
            print("An unknown error occurred:", str(e))

    def sample_rand_traces(self):
        n_txt = self.w_api_bin.edit_no_traces.text()
        if n_txt == "":
            n = 10
            self.w_api_bin.edit_no_traces.setText("10")
        else:
            n = int(n_txt)
        if self.w_api_bin.chk_pileup_rand.isChecked():
            dfx = self.df_apibin_ch[self.df_apibin_ch.pileup == True]
            if dfx.shape[0] < n:
                n = dfx.shape[0]
            df_rand = dfx.sample(n)
        elif self.w_api_bin.chk_cfderr_rand.isChecked():
            dfx = self.df_apibin_ch[self.df_apibin_ch.CFD_error == True]
            if dfx.shape[0] < n:
                n = dfx.shape[0]
            df_rand = dfx.sample(n)
        elif self.w_api_bin.chk_flag_rand.isChecked():
            dfx = self.df_apibin_ch[self.df_apibin_ch.trace_flag == True]
            if dfx.shape[0] < n:
                n = dfx.shape[0]
            df_rand = dfx.sample(n)
        else:
            df_rand = self.df_apibin_ch.sample(n)
            df_rand = df_rand.sort_index()
        self.df_rand = df_rand

    def cal_own_erg_api_bin(self):
        try:
            self.plot_apibin_energy_from_traces()
        except Exception as e:
            print("ERROR: Could not calculate own energy")
            print("An unknown error occurred:", str(e))

    def initialize_plot_api_bin_erg(self):
        self.fig_bin_erg = self.w_api_bin.plot_energy.canvas.figure
        self.fig_bin_erg.clear()
        self.fig_bin_erg.set_constrained_layout(True)
        self.fig_bin_erg.patch.set_alpha(0)
        self.ax_bin_erg = self.fig_bin_erg.add_subplot()
        self.fig_bin_erg.canvas.draw_idle()
        plt.rcParams.update({"font.size": 18})

    def initialize_plot_api_bin_rand_traces(self):
        self.fig_bin_rand_traces = self.w_api_bin.plot_random_traces.canvas.figure
        self.fig_bin_rand_traces.clear()
        self.fig_bin_rand_traces.set_constrained_layout(True)
        self.fig_bin_rand_traces.patch.set_alpha(0)
        self.ax_bin_rand_traces = self.fig_bin_rand_traces.add_subplot()
        self.fig_bin_rand_traces.canvas.draw_idle()
        plt.rcParams.update({"font.size": 18})

    def initialize_plot_api_bin_own_erg(self):
        self.fig_bin_own_erg = self.w_api_bin.plot_own_energy.canvas.figure
        self.fig_bin_own_erg.clear()
        self.fig_bin_own_erg.set_constrained_layout(True)
        self.fig_bin_own_erg.patch.set_alpha(0)
        self.ax_bin_own_erg = self.fig_bin_own_erg.add_subplot()
        self.fig_bin_own_erg.canvas.draw_idle()
        plt.rcParams.update({"font.size": 18})

    def initialize_plot_api_bin_erg_tr(self):
        self.fig_bin_erg_tr = self.w_api_bin.plot_energy_traces.canvas.figure
        self.fig_bin_erg_tr.clear()
        self.fig_bin_erg_tr.set_constrained_layout(True)
        self.fig_bin_erg_tr.patch.set_alpha(0)
        self.ax_bin_erg_tr = self.fig_bin_erg_tr.add_subplot()
        self.fig_bin_erg_tr.canvas.draw_idle()
        plt.rcParams.update({"font.size": 18})

    def plot_apibin_energy(self):
        self.ax_bin_erg.clear()
        channels = self.df_apibin_ch.channel.unique()
        channels.sort()
        no_bins_txt = self.w_api_bin.edit_bins.text()
        if no_bins_txt == "":
            no_bins = 4098
            self.w_api_bin.edit_bins.setText("4098")
        else:
            no_bins = int(no_bins_txt)

        for c in channels:
            df_tmp = self.df_apibin_ch[self.df_apibin_ch.channel == c]
            cts, ed = np.histogram(df_tmp.energy, bins=no_bins)
            sum_cts = cts.sum()
            self.api_gam_x_bin = (ed[1:] + ed[:-1]) / 2
            self.ax_bin_erg.plot(
                self.api_gam_x_bin, cts, label=f"Channel: {c}, counts: {sum_cts}"
            )
        self.spect_api_bin1 = cts
        self.ax_bin_erg.legend()
        self.ax_bin_erg.set_yscale(self.yscale_api_bin)
        self.ax_bin_erg.set_xlabel("Channels")
        self.ax_bin_erg.set_ylabel("Counts")
        self.espan_api_bin()
        self.fig_bin_erg.canvas.draw_idle()

    def espan_api_bin(self):
        self.span_api_bin = SpanSelector(
            self.ax_bin_erg,
            self.enselect_api_bin,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.3, facecolor="green"),
        )

    def enselect_api_bin(self, xmin, xmax):
        xmin = round(xmin, 4)
        xmax = round(xmax, 4)
        print("xmin: ", xmin)
        print("xmax: ", xmax)
        self.apply_energy_filter_bin(xmin, xmax)

    def apply_energy_filter_bin(self, xmin, xmax):
        self.ax_bin_erg_tr.clear()
        elim = (self.df_apibin_ch["energy"] > xmin) & (self.df_apibin_ch["energy"] < xmax)
        self.df_bin_tr = self.df_apibin_ch[elim]
        self.df_bin_tr.reset_index(drop=True, inplace=True)
        no_traces = self.df_bin_tr.shape[0]
        self.w_api_bin.edit_no_events_erg_tr.setText(str(no_traces))
        self.plot_apibin_erg_tr()

    def plot_apibin_erg_tr(self):
        self.ax_bin_erg_tr.clear()
        t = np.arange(0, self.df_bin_tr.trace.iloc[0].shape[0], 1) * 2
        for i in self.df_bin_tr.index:
            erg = self.df_bin_tr.loc[i].energy
            pileup = self.df_bin_tr.loc[i].pileup
            tf = self.df_bin_tr.loc[i].trace_flag
            cfderr = self.df_bin_tr.loc[i].CFD_error
            lab = f"i:{i}; erg: {erg}; pu:{pileup}, flag:{tf}, CFDerr:{cfderr}"
            self.ax_bin_erg_tr.plot(t, self.df_bin_tr.loc[i].trace, label=lab)
        self.ax_bin_erg_tr.set_xlabel("Time (ns)")
        self.ax_bin_erg_tr.set_ylabel("Amplitude (a.u)")
        self.legend_api_erg_tr = self.ax_bin_erg_tr.legend()
        self.legend_api_erg_tr.set_visible(False)
        if self.w_api_bin.chk_legend_erg_tr.isChecked():
            self.legend_api_erg_tr.set_visible(True)
        self.w_api_bin.chk_legend_erg_tr.toggled.connect(self.toggle_legend_apibin_erg_tr)
        self.fig_bin_erg_tr.canvas.draw_idle()

    def toggle_legend_apibin_erg_tr(self, checked: bool):
        self.legend_api_erg_tr.set_visible(checked)
        self.fig_bin_erg_tr.canvas.draw_idle()

    def plot_apibin_rand_traces(self):
        self.ax_bin_rand_traces.clear()
        t = np.arange(0, self.df_rand.trace.iloc[0].shape[0], 1) * 2
        for i in self.df_rand.index:
            erg = self.df_rand.loc[i].energy
            pileup = self.df_rand.loc[i].pileup
            tf = self.df_rand.loc[i].trace_flag
            cfderr = self.df_rand.loc[i].CFD_error
            lab = f"i:{i}; erg: {erg}; pu:{pileup}, flag:{tf}, CFDerr:{cfderr}"
            self.ax_bin_rand_traces.plot(t, self.df_rand.loc[i].trace, label=lab)
        self.ax_bin_rand_traces.set_xlabel("Time (ns)")
        self.ax_bin_rand_traces.set_ylabel("Amplitude (a.u)")
        self.legend_api_rand_traces = self.ax_bin_rand_traces.legend()
        self.legend_api_rand_traces.set_visible(False)
        if self.w_api_bin.chk_legend_rand.isChecked():
            self.legend_api_rand_traces.set_visible(True)
        self.w_api_bin.chk_legend_rand.toggled.connect(self.toggle_legend_apibin)
        self.fig_bin_rand_traces.canvas.draw_idle()

    def plot_fft(self):
        self.ax_bin_rand_traces.clear()
        for fft in self.ffts:
            self.ax_bin_rand_traces.plot(self.freqs / 1e6, fft)
        self.ax_bin_rand_traces.set_xlabel("Frequency (MHz)")
        self.ax_bin_rand_traces.set_ylabel("Amplitude (a.u)")
        self.ax_bin_rand_traces.set_yscale("log")
        self.fig_bin_rand_traces.canvas.draw_idle()

    def toggle_legend_apibin(self, checked: bool):
        self.legend_api_rand_traces.set_visible(checked)
        self.fig_bin_rand_traces.canvas.draw_idle()

    def plot_apibin_energy_from_traces(self):
        self.ax_bin_own_erg.clear()
        a = self.w_api_bin.edit_a.text()
        b = self.w_api_bin.edit_b.text()
        baseline = self.w_api_bin.edit_baseline.text()
        no_bins_txt = self.w_api_bin.edit_bins.text()
        if no_bins_txt == "":
            no_bins = 4098
            self.w_api_bin.edit_bins.setText("4098")
        else:
            no_bins = int(no_bins_txt)

        traces = self.df_apibin_ch.trace.to_numpy()
        traces = np.vstack(traces)
        baseline_pts = int(int(baseline) / 2)  # in ns
        traces_norm = traces - traces[:, 0:baseline_pts].mean(axis=1).reshape(-1, 1)
        if a == "" or b == "":  # use the entire trace
            a = 0
            b = int(traces.shape[1])
        else:
            a = int(int(a) / 2)  # in ns
            b = int(int(b) / 2)  # in ns
        erg_norm = traces_norm[:, a:b].sum(axis=1)
        cts, edg = np.histogram(erg_norm, bins=no_bins)
        self.spect_api_bin2 = cts
        self.api_gam_x_bin2 = (edg[1:] + edg[:-1]) / 2

        self.ax_bin_own_erg.plot(cts)
        self.ax_bin_own_erg.set_yscale("log")
        self.ax_bin_own_erg.set_xlabel("Channels")
        self.ax_bin_own_erg.set_ylabel("Counts")
        self.fig_bin_own_erg.canvas.draw_idle()

    # API 3D plot
    def open_api_3D(self):
        from ..dialogs import WindowAPI3D
        self.w_api_3D = WindowAPI3D()
        self.w_api_3D.activateWindow()
        self.w_api_3D.show()
        try:
            self.w_api_3D.button_api_plot3D.clicked.connect(self.create_plot_api3D)
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def create_plot_api3D(self):
        try:
            self.df_current
        except Exception as e:
            print("Make sure you load an API file on the previous window.", str(e))
        if self.sims_flag:
            X, Y, Z = self.df_current["X"], self.df_current["Y"], self.df_current["Z"]
        else:
            X, Y, Z = apicalc.api_xyz(
                df=self.df_current, det_pos=[0, 22.2, -25.515], use_det=False
            )

        # Define the number of bins for the histogram
        num_bins = int(self.w_api_3D.no_bins.text())
        iso_min = float(self.w_api_3D.isomin.text())
        iso_max = float(self.w_api_3D.isomax.text())
        opacity = float(self.w_api_3D.opacity.text())
        surfcount = int(self.w_api_3D.surfcount.text())
        # Compute the 3D histogram
        hist, edges = np.histogramdd((X, Y, Z), bins=num_bins)
        # Compute the bin centers
        x_bin_centers = 0.5 * (edges[0][1:] + edges[0][:-1])
        y_bin_centers = 0.5 * (edges[1][1:] + edges[1][:-1])
        z_bin_centers = 0.5 * (edges[2][1:] + edges[2][:-1])
        # Create a meshgrid of the bin centers
        x_bin_centers, y_bin_centers, z_bin_centers = np.meshgrid(
            x_bin_centers, y_bin_centers, z_bin_centers, indexing="ij"
        )
        # Flatten the arrays for plotting
        x_flat = x_bin_centers.flatten()
        y_flat = y_bin_centers.flatten()
        z_flat = z_bin_centers.flatten()
        values_flat = hist.flatten()
        # Normalize the histogram values for color mapping
        norm_values = (values_flat - values_flat.min()) / (
            values_flat.max() - values_flat.min()
        )
        vol = ndimage.gaussian_filter(norm_values, 4)
        vol /= vol.max()
        # Create a volume plot
        fig = go.Figure(
            go.Volume(
                x=x_flat,
                y=y_flat,
                z=z_flat,
                value=vol,
                isomin=iso_min,
                isomax=iso_max,
                opacity=opacity,
                surface_count=surfcount,
                colorscale="Viridis",
            )
        )
        fig.update_layout(
            scene_xaxis_showticklabels=False,
            scene_yaxis_showticklabels=False,
            scene_zaxis_showticklabels=False,
        )
        self.w_api_3D.browser.setHtml(fig.to_html(include_plotlyjs="cdn"))
        # Render the plot to an HTML file
        html = pio.to_html(fig, full_html=True)
        # Write the HTML to a temporary file and load it in the QWebEngineView
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            f.write(html.encode("utf-8"))
            temp_file_path = f.name
        self.w_api_3D.browser.setUrl(QUrl.fromLocalFile(temp_file_path))
