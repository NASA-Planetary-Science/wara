from copy import deepcopy
from wara import diagnostics


class DiagnosticsMixin:

    def initialize_plots_diagnostics(self):
        self.fig_diag00 = self.diag_plot00.canvas.figure
        self.fig_diag01 = self.diag_plot01.canvas.figure
        self.fig_diag02 = self.diag_plot02.canvas.figure
        self.fig_diag03 = self.diag_plot03.canvas.figure
        self.fig_diag10 = self.diag_plot10.canvas.figure
        self.fig_diag11 = self.diag_plot11.canvas.figure
        self.fig_diag12 = self.diag_plot12.canvas.figure
        self.fig_diag13 = self.diag_plot13.canvas.figure
        self.figs_diag = [
            self.fig_diag00,  # spectrum
            self.fig_diag01,  # counts/cr
            self.fig_diag02,  # measurement times
            self.fig_diag03,  # live time
            self.fig_diag10,  # peak xrange
            self.fig_diag11,  # counts/cr (peak)
            self.fig_diag12,  # centroid/max
            self.fig_diag13,
        ]  # fwhm (%)
        for fig in self.figs_diag:
            fig.clear()
            fig.set_constrained_layout(True)
            fig.patch.set_alpha(0)
        self.ax00 = self.fig_diag00.add_subplot()
        self.ax01 = self.fig_diag01.add_subplot()
        self.ax02 = self.fig_diag02.add_subplot()
        self.ax03 = self.fig_diag03.add_subplot()
        self.ax10 = self.fig_diag10.add_subplot()
        self.ax11 = self.fig_diag11.add_subplot()
        self.ax12 = self.fig_diag12.add_subplot()
        self.ax13 = self.fig_diag13.add_subplot()
        for fig in self.figs_diag:
            fig.canvas.draw_idle()

    def load_diagnostic_data(self):
        self.diag_data = diagnostics.Diagnostics(self.folder_path)
        self.button_cts_cr.setEnabled(True)
        self.button_combine_send.setEnabled(True)

    def change_cts_cr_diag(self):
        if self.flag_tot_cts:
            self.ax01.clear()
            self.diag_data.plot_count_rates(time=False, ax=self.ax01)
            self.fig_diag01.canvas.draw_idle()
            self.flag_tot_cts = False
        else:
            self.ax01.clear()
            self.diag_data.plot_counts(time=False, ax=self.ax01)
            self.flag_tot_cts = True
            self.fig_diag01.canvas.draw_idle()

    def combine_and_send_to_spect(self):
        self.spect = self.diag_data.combine_spects()
        self.e_units = self.spect.e_units
        self.e_units_orig = deepcopy(self.e_units)
        self.spect_orig = deepcopy(self.spect)
        self.create_graph(fit=False, reset=True)
        self.search = 0
        self.setWindowTitle("wara: combined spectrum from Diagnostics tab")

    def plot_diagnostic_total(self):
        self.diag_data.plot_spectra(ax=self.ax00)
        self.diag_data.plot_counts(time=False, ax=self.ax01)
        self.diag_data.plot_measurement_times(time=False, ax=self.ax02)
        self.diag_data.plot_livetime_percent(time=False, ax=self.ax03)
        self.flag_tot_cts = True

    def try_fit_peaks_diagnostics(self, integral=False):
        try:
            self.fit_peaks_diagnostics(integral)
            self.button_diag_save.setEnabled(True)
        except Exception as e:
            print("ERROR: Could not perform fit or integral")
            print("An unknown error occurred:", str(e))

    def fit_peaks_diagnostics(self, integral=False):
        self.ax10.clear()
        self.ax11.clear()
        self.ax12.clear()
        self.ax13.clear()
        xmid_txt = self.xmin_diag.text()
        width_txt = self.xmax_diag.text()
        if xmid_txt == "" or width_txt == "":
            print("Must specify X-range bounds")
        elif self.isevaluable(xmid_txt) and self.isevaluable(width_txt):
            xmid = eval(xmid_txt)
            width = eval(width_txt)
        if integral:
            self.diag_data.calculate_integral(xmid=xmid, width=width)
        else:
            self.diag_data.fit_peaks(xmid=xmid, width=width)
        self.diag_data.plot(ax=self.ax10)
        self.diag_data.plot_fit_counts(time=False, ax=self.ax11)
        self.diag_data.plot_centroid(time=False, ax=self.ax12)
        self.diag_data.plot_fwhm(time=False, ax=self.ax13)
        self.fig_diag10.canvas.draw_idle()
        self.fig_diag11.canvas.draw_idle()
        self.fig_diag12.canvas.draw_idle()
        self.fig_diag13.canvas.draw_idle()
        self.flag_fit_cts = True
        self.flag_centroid = True
        self.button_cts_cr_fit.setEnabled(True)
        self.button_centroid_max.setEnabled(True)

    def change_cts_cr_fit_diag(self):
        if self.flag_fit_cts:
            self.ax11.clear()
            self.diag_data.plot_fit_count_rates(time=False, ax=self.ax11)
            self.fig_diag11.canvas.draw_idle()
            self.flag_fit_cts = False
        else:
            self.ax11.clear()
            self.diag_data.plot_fit_counts(time=False, ax=self.ax11)
            self.flag_fit_cts = True
            self.fig_diag11.canvas.draw_idle()

    def change_centroid_max_diag(self):
        if self.flag_centroid:
            self.ax12.clear()
            self.diag_data.plot_max(time=False, ax=self.ax12)
            self.fig_diag12.canvas.draw_idle()
            self.flag_centroid = False
        else:
            self.ax12.clear()
            self.diag_data.plot_centroid(time=False, ax=self.ax12)
            self.flag_centroid = True
            self.fig_diag12.canvas.draw_idle()
