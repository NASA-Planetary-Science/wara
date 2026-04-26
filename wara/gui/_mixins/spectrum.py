import numpy as np
import pandas as pd
from matplotlib.widgets import SpanSelector
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from wara import peakfit as pf
from wara import peaksearch as ps
from wara import spectrum as sp
from wara import advanced_fit as adv
from ..table import TableModel


class SpectrumMixin:

    def main_info_activate(self):
        self.w_info_main.activateWindow()
        self.w_info_main.show()

    def switch_toolbar(self):
        current = self.tabWidget.currentIndex()
        for i, tb in enumerate(self.toolbars):
            tb.setVisible(i == current)

    def create_graph(self, fit=True, reset=True):
        if reset:
            self.reset_main_figure()
            self.remove_ticks_fit()
            self.reset_span()
        if fit:
            self.search.plot(
                yscale=self.scale, snrs=self.snr_state, ax=self.ax_main
            )
            self.span_select()
        else:
            self.spect.plot(ax=self.ax_main, scale=self.scale)
        self.ax_main.set_yscale(self.scale)
        self.fig.canvas.draw_idle()

    def initialize_main_figure(self):
        self.fig = self.main_plot.canvas.figure  # spectrum
        self.fig_fit = self.fit_plot.canvas.figure  # residual
        self.fig.clear()
        self.fig_fit.clear()
        self.fig.set_constrained_layout(True)
        self.fig_fit.set_constrained_layout(True)
        gs = self.fig_fit.add_gridspec(2, 1, height_ratios=[0.3, 1.5])
        self.fig.patch.set_alpha(0)
        # axes
        self.ax_main = self.fig.add_subplot()
        self.ax_res = self.fig_fit.add_subplot(gs[0, 0])
        self.ax_fit = self.fig_fit.add_subplot(gs[1, 0])

        self.fig.canvas.draw_idle()
        self.fig_fit.canvas.draw_idle()

    def reset_spect_params(self):
        self.spect = 0
        self.search = 0
        self.fit = 0
        self.span = None

    def reset_main_figure(self):
        if self.button_add_peak.isChecked():
            self.fig.canvas.mpl_disconnect(self.cid)
            self.button_add_peak.setChecked(False)
        self.initialize_main_figure()
        self.fig.canvas.draw_idle()
        self.fig_fit.canvas.draw_idle()

    def clear_spectrum(self):
        self.reset_spect_params()
        self.reset_main_figure()
        self.remove_ticks_fit()

    def remove_ticks_fit(self):
        self.ax_fit.set_xticks([])
        self.ax_fit.set_yticks([])
        self.ax_res.set_xticks([])
        self.ax_res.set_yticks([])

    def reset_span(self):
        if self.span is not None:
            self.span.set_active(False)
            self.span.set_visible(False)

    def update_scale(self):
        if self.scale == "log":
            self.ax_main.set_yscale("linear")
            self.scale = "linear"
        elif self.scale == "linear":
            self.ax_main.set_yscale("log")
            self.scale = "log"
        self.fig.canvas.draw_idle()

    def add_peak(self):
        if self.button_add_peak.isChecked():
            print("Waiting to add a new peak...")
            self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
            if self.span is not None:
                self.span.set_active(False)
        else:
            self.fig.canvas.mpl_disconnect(self.cid)

    def onclick(self, event):
        xnew = event.xdata
        if self.spect.energies is not None:
            xnew = np.where(self.spect.energies >= xnew)[0][0]
        else:
            xnew = np.where(self.spect.channels >= xnew)[0][0]
        print(xnew)
        # if no search object initialized, initialize a dummy one
        if self.search == 0:
            self.search = ps.PeakSearch(self.spect, 420, 3, min_snr=1e6, method="scipy")
            self.span_select()
        x_idx = np.searchsorted(self.search.peaks_idx, xnew)
        self.search.peaks_idx = np.insert(self.search.peaks_idx, x_idx, xnew)
        fwhm_guess_new = self.search.fwhm(xnew)
        self.search.fwhm_guess = np.insert(self.search.fwhm_guess, x_idx, fwhm_guess_new)
        if self.spect.energies is not None:
            self.ax_main.axvline(
                x=self.spect.energies[xnew], color="red", linestyle="--", alpha=0.2
            )
        else:
            self.ax_main.axvline(
                x=self.spect.channels[xnew], color="red", linestyle="--", alpha=0.2
            )
        self.fig.canvas.draw_idle()
        self.fig.canvas.mpl_disconnect(self.cid)
        self.button_add_peak.setChecked(False)
        self.span.set_active(True)

    ## Advanced fitting
    ## TODO
    def open_adv_fit(self):
        self.w_adv_fit.activateWindow()
        self.parea = adv.PeakAreaLinearBkg(spectrum=self.spect,
                                           x1=self.idxmin_fit, x2=self.idxmax_fit)
        self.w_adv_fit.show()
        self.reset_plot_adv_fit()
        self.plot_adv_fit(bool_area=False)

    def reset_plot_adv_fit(self):
        self.fig_adv_fit = self.w_adv_fit.plot_adv_fit.canvas.figure
        self.fig_adv_fit.clear()
        self.fig_adv_fit.set_constrained_layout(True)
        self.fig_adv_fit.patch.set_alpha(0)
        self.ax_area = self.fig_adv_fit.add_subplot()

    def plot_adv_fit(self, bool_area=False):
        self.parea.plot(ax=self.ax_area, areas=bool_area)
        self.fig_adv_fit.canvas.draw_idle()

    def calculate_peak_area(self):
        x1 = self.w_adv_fit.edit_x1.text()
        x2 = self.w_adv_fit.edit_x2.text()
        if self.isevaluable(x1) and self.isevaluable(x2) and self.parea is not None:
            peak_range = [eval(x1), eval(x2)]
            self.parea.calculate_peak_area(x1=peak_range[0], x2=peak_range[1])
            self.reset_plot_area()
            self.plot_area(bool_area=True)

    ## Customize plots
    def customize_info_activate(self):
        self.w_cust_info.activateWindow()
        self.w_cust_info.show()

    def new_window_custom(self):
        self.w_cust.activateWindow()
        self.w_cust.show()

    def try_cust_labels(self):
        try:
            self.cust_labels()
        except Exception as e:
            print("Invalid label/constant values")
            print("An unknown error occurred:", str(e))

    def try_cust_smooth(self):
        self.cust_smooth()

    def try_cust_countRate(self):
        try:
            self.cust_countRate()
        except Exception as e:
            print("Invalid count rate values")
            print("An unknown error occurred:", str(e))

    def try_cust_shift(self):
        try:
            self.cust_shift()
        except Exception as e:
            print("Invalid gain shift values")
            print("An unknown error occurred:", str(e))

    def cust_labels(self):
        description = self.w_cust.descript_txt.text()
        xlabel = self.w_cust.x_label_txt.text()
        ylabel = self.w_cust.y_label_txt.text()
        legend = self.w_cust.legend_txt.text().split(",")
        yconst = self.w_cust.yconst_txt.text()
        xconst = self.w_cust.xconst_txt.text()
        xunits = self.w_cust.xunits_txt.text()
        if description != "":
            self.spect.description = description
        if xlabel != "":
            self.ax_main.set_xlabel(xlabel)
            self.spect.x_units = xlabel
            self.fig.canvas.draw_idle()
        if ylabel != "":
            self.ax_main.set_ylabel(ylabel)
            self.spect.y_label = ylabel
            self.fig.canvas.draw_idle()
        if legend != [""]:
            legend = [x.strip() for x in legend]
            self.ax_main.legend(self.ax_main.get_legend_handles_labels()[0], legend)
            self.spect.label = legend[-1]
            self.fig.canvas.draw_idle()
        if yconst != "":
            self.spect.counts = self.spect.counts * eval(yconst)
            self.create_graph(fit=False, reset=True)
            self.search = 0
        if xconst != "":
            if self.spect.e_units is None:
                new_x = self.spect.channels * eval(xconst)
                self.e_units = f"Channels * {xconst}"
                self.spect = sp.Spectrum(
                    counts=self.spect.counts, energies=new_x, e_units=self.e_units
                )
            else:
                new_x = self.spect.energies * eval(xconst)
                self.spect = sp.Spectrum(
                    counts=self.spect.counts, energies=new_x, e_units=self.e_units
                )
            self.create_graph(fit=False, reset=True)
            self.search = 0
        if xunits != "":
            self.e_units = xunits
            self.spect.e_units = xunits
            self.spect.x_units = f"Energy ({self.e_units})"
            self.create_graph(fit=False, reset=True)
            self.search = 0

    def cust_smooth(self):
        rebin = self.w_cust.rebin_txt.text()
        moving_avg = self.w_cust.smooth_txt.text()
        if rebin != "":
            nbins = eval(rebin)
            self.spect.rebin(by=nbins)
            self.create_graph(fit=False, reset=True)
            self.search = 0
        if moving_avg != "":
            n = int(eval(moving_avg))
            self.spect.smooth(num=n)
            self.create_graph(fit=False, reset=True)
            self.search = 0

    def cust_countRate(self):
        livetime = self.w_cust.livetime_txt.text()
        if self.w_cust.checkBox_CR.isChecked():
            yl = "CPS"
            self.ax_main.set_ylabel(yl)
            self.spect.y_label = yl
            self.spect.cps = True
            self.fig.canvas.draw_idle()
        if livetime != "" and self.isevaluable(livetime) and self.spect.cps:
            self.spect.livetime = eval(livetime)
            self.create_graph(fit=False, reset=True)
            self.search = 0

    def cust_shift(self):
        gain_shift = self.w_cust.shiftBy_txt.text()
        if self.w_cust.checkBox_erg.isChecked():
            bool_erg = True
        else:
            bool_erg = False
        if gain_shift != "":
            gs = eval(gain_shift)
            self.spect.gain_shift(by=gs, energy=bool_erg)
            self.create_graph(fit=False, reset=True)
            self.search = 0

    ## Add/subtract spectra
    def add_sub_info_activate(self):
        self.w_add_sub_info.activateWindow()
        self.w_add_sub_info.show()

    def activate_window_add_subtract(self):
        self.w_add_sub.activateWindow()
        self.w_add_sub.show()

    def load_file1(self):
        try:
            self.file1, self.e_units1, self.spect1 = self.open_file()
        except Exception as e:
            self.file1 = "**ERROR loading file**"
            print("An unknown error occurred:", str(e))
        disp1 = self.file1.split("/")[-1]
        self.w_add_sub.filename1.setText(disp1)

    def load_file2(self):
        try:
            self.file2, self.e_units2, self.spect2 = self.open_file()
        except Exception as e:
            self.file2 = "**ERROR loading file**"
            print("An unknown error occurred:", str(e))
        disp2 = self.file2.split("/")[-1]
        self.w_add_sub.filename2.setText(disp2)

    def add_sub_plot(self):
        try:
            if self.w_add_sub.checkBox_add.isChecked():
                cts = self.spect1.counts + self.spect2.counts
            elif self.w_add_sub.checkBox_sub.isChecked():
                cts = self.spect1.counts - self.spect2.counts

            if self.spect1.energies is not None:
                x = self.spect1.energies
                self.spect = sp.Spectrum(counts=cts, energies=x, e_units=self.e_units1)
                self.e_units = self.e_units1
            else:
                self.spect = sp.Spectrum(counts=cts, e_units=self.e_units1)
            self.create_graph(fit=False, reset=True)
            self.search = 0
        except Exception as e:
            print("Could not perform operation")
            print("An unknown error occurred:", str(e))

    ## Fitting
    def which_button(self):
        if self.pushButton_poly1.isChecked():
            self.bg = "poly1"
        elif self.pushButton_poly2.isChecked():
            self.bg = "poly2"
        elif self.pushButton_poly3.isChecked():
            self.bg = "poly3"
        elif self.pushButton_poly4.isChecked():
            self.bg = "poly4"
        elif self.pushButton_poly5.isChecked():
            self.bg = "poly5"
        elif self.pushButton_exp.isChecked():
            self.bg = "exponential"

    def which_button_gauss(self):
        if self.pushButton_gauss2.isChecked():
            self.sk_gauss = True
        else:
            self.sk_gauss = False

    def update_poly(self):
        if len(self.list_xrange) != 0:
            self.which_button()
            try:
                self.fit = pf.PeakFit(
                    self.search, self.list_xrange[-1], bkg=self.bg, skew=self.sk_gauss
                )
                self.ax_res.clear()
                self.ax_fit.clear()
                self.fit.plot(
                    fig=self.fig_fit,
                    ax_res=self.ax_res,
                    ax_fit=self.ax_fit,
                )
                data = self.get_values_table_fit()
                self.activate_fit_table(data)
            except Exception as e:
                print("update_poly: could not perform fit")
                print("An unknown error occurred:", str(e))
            self.fig_fit.canvas.draw_idle()

    def update_gauss(self):
        if len(self.list_xrange) != 0:
            self.which_button_gauss()
            try:
                self.fit = pf.PeakFit(
                    self.search, self.list_xrange[-1], bkg=self.bg, skew=self.sk_gauss
                )
                self.ax_res.clear()
                self.ax_fit.clear()
                self.fit.plot(
                    fig=self.fig,
                    ax_res=self.ax_res,
                    ax_fit=self.ax_fit,
                )
                data = self.get_values_table_fit()
                self.activate_fit_table(data)
            except Exception as e:
                print("update_gauss:could not perform fit")
                print("An unknown error occurred:", str(e))
            self.fig_fit.canvas.draw_idle()

    def span_select(self):
        self.span = SpanSelector(
            self.ax_main,
            self.onselect,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.3, facecolor="green"),
        )

    def enable_fitButtons(self, boolV):
        self.button_adv_fit.setEnabled(boolV)
        self.button_add_cal.setEnabled(boolV)
        self.button_add_fwhm.setEnabled(boolV)
        self.button_add_eff.setEnabled(boolV)

    def onselect(self, xmin, xmax):
        self.idxmin_fit = round(xmin, 4)
        self.idxmax_fit = round(xmax, 4)
        xrange = [self.idxmin_fit, self.idxmax_fit]
        self.list_xrange.append(xrange)
        print("xmin: ", self.idxmin_fit)
        print("xmax: ", self.idxmax_fit)
        self.ax_res.clear()
        self.ax_fit.clear()
        self.fit = pf.PeakFit(self.search, xrange, bkg=self.bg, skew=self.sk_gauss)
        self.fit.plot(
            fig=self.fig_fit,
            ax_res=self.ax_res,
            ax_fit=self.ax_fit,
        )
        data = self.get_values_table_fit()
        self.activate_fit_table(data)
        self.enable_fitButtons(True)
        self.fig_fit.canvas.draw_idle()

    def get_values_table_fit(self):
        cols = [
            "Mean",
            "Peak area",
            "FWHM",
            "FWHM (%)",
            "+/- Area",
            "+/- Area (%)",
        ]
        mean = []
        area = []
        fwhm = []
        fwhm_p = []
        std_mu = []
        std_A = []
        std_A_p = []
        std_fwhm = []
        for i, e in zip(self.fit.peak_info, self.fit.peak_err):
            ls = list(i.values())
            lse = list(e.values())
            mean.append(ls[0])
            area.append(ls[1])
            fwhm.append(ls[2])
            fwhm_p.append(ls[2] / ls[0] * 100)
            std_A.append(lse[1])
            std_A_p.append(lse[1] / ls[1] * 100)

        rs = np.array([mean, area, fwhm, fwhm_p, std_A, std_A_p]).T
        df = pd.DataFrame(columns=cols, data=rs)
        df = df.round(decimals=3)
        return df

    def activate_fit_table(self, data):
        self.scroll_fit = self.table_fit_area
        self.table_fit = QtWidgets.QTableView()
        self.table_fit.setAlternatingRowColors(True)
        self.table_fit.setSortingEnabled(True)
        stylesheet_header = "::section{Background-color:lightgreen}"
        self.table_fit.horizontalHeader().setStyleSheet(stylesheet_header)
        self.table_fit.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.model_fit = TableModel(data)
        self.table_fit.setModel(self.model_fit)
        self.scroll_fit.setWidget(self.table_fit)
        stylesheet_ix = "::section{Background-color:lightgoldenrodyellow}"
        self.table_fit.setStyleSheet(stylesheet_ix)
        self.table_fit.setColumnWidth(0, 150)
        self.table_fit.setColumnWidth(5, 160)
