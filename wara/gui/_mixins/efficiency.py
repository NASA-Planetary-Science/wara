import datetime
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from wara import efficiency
from wara import resolution
from wara import peakfit as pf
from ..table import TableModel


class EfficiencyMixin:

    def eff_info_activate(self):
        self.w_info_eff.activateWindow()
        self.w_info_eff.show()

    def reset_eff_figure(self):
        self.fig_eff = self.efficiency_plot.canvas.figure
        self.fig_eff.clear()
        self.fig_eff.set_constrained_layout(True)
        gs2 = self.fig_eff.add_gridspec(2, 1, height_ratios=[0.3, 1.5])
        # axes
        self.ax_eff = self.fig_eff.add_subplot(gs2[1, :])
        self.ax_eff_res = self.fig_eff.add_subplot(gs2[0, :], sharex=self.ax_eff)
        # remove ticks
        self.ax_eff.set_xticks([])
        self.ax_eff.set_yticks([])
        self.ax_eff_res.set_xticks([])
        self.ax_eff_res.set_yticks([])

    def reset_eff_all(self):
        self.reset_eff_figure()
        self.fig_eff.canvas.draw_idle()
        self.eff_vals = []
        self.df_eff = None
        try:
            self.table_eff.setParent(None)
        except Exception as e:
            print("Cannot reset table")
            print(str(e))

    def new_window_eff(self):
        from ..dialogs import WindowEff
        self.w_eff = WindowEff()
        self.w_eff.show()
        self.w_eff.accepted.connect(self.add_eff)

    def add_eff(self):
        try:
            self.get_eff_input()
            self.eff_obj = efficiency.Efficiency(
                t_half=self.t_half,
                A0=self.A0,
                Br=self.B,
                livetime=self.acq_time,
                t_elapsed=self.delta_t_sec,
                which_peak=self.n_peak_eff,
            )
            self.eff_obj.calculate_efficiency(self.fit)
            self.eff_obj.calculate_error(
                self.fit,
                self.t_half_sig,
                self.A0_sig,
                self.B_sig,
                self.delta_t_sig,
                self.acq_time_sig,
            )

            df = self.eff_obj.to_df(e_units=self.e_units)
            self.concat_df_eff(df)
            self.update_eff_table()
            self.ax_eff_res.clear()
            self.plot_eff_points()
        except Exception as e:
            print("ERROR: could not add efficiency value")
            print(e)

    def plot_eff_points(self):
        if self.df_eff.shape[0] > 0:
            self.ax_eff_res.clear()
            x = self.df_eff[f"Energy ({self.e_units})"]
            y = self.df_eff["Efficiency"] * 100
            y_err = self.df_eff["+/- Efficiency"]
            self.ax_eff.clear()
            efficiency.plot_points(
                e_vals=x,
                eff_vals=y,
                err_vals=y_err,
                e_units=self.e_units,
                ax=self.ax_eff,
            )
            self.ax_eff_res.tick_params(axis="x", colors="white")
            self.fig_eff.canvas.draw_idle()

    def concat_df_eff(self, df):
        if df is None:
            self.df_eff = df
        else:
            self.df_eff = pd.concat([self.df_eff, df], ignore_index=True)

    def update_eff_table(self):
        self.df_eff.fillna(value=0, inplace=True)
        self.df_eff.sort_values(
            by=[f"Energy ({self.e_units})"], inplace=True, ignore_index=True
        )
        self.df_eff = self.df_eff.round(6)
        self.df_eff.drop_duplicates(
            subset=[f"Energy ({self.e_units})"], inplace=True, ignore_index=True
        )
        self.activate_eff_table(data=self.df_eff)

    def activate_eff_table(self, data):
        self.selected_rows_eff = []
        self.selected_indexes_eff = []
        self.table_eff = QtWidgets.QTableView()
        self.table_eff.setAlternatingRowColors(True)
        self.table_eff.setSortingEnabled(False)
        stylesheet_header = "::section{Background-color:lightgreen}"
        self.table_eff.horizontalHeader().setStyleSheet(stylesheet_header)
        self.table_eff.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.table_eff.resizeColumnsToContents()
        self.model_eff = TableModel(data)
        self.table_eff.setModel(self.model_eff)
        self.table_eff_area.setWidget(self.table_eff)
        stylesheet_ix = "::section{Background-color:lightgoldenrodyellow}"
        self.table_eff.setStyleSheet(stylesheet_ix)
        self.table_eff.setColumnWidth(10, 150)
        self.table_eff.setColumnWidth(12, 150)
        self.table_eff.selectionModel().selectionChanged.connect(
            self.on_selectionChanged_eff
        )

    def on_selectionChanged_eff(self, selected, deselected):
        for ix in selected.indexes():
            self.selected_rows_eff.append(ix.row())
        for ix in deselected.indexes():
            try:
                self.selected_rows_eff.remove(ix.row())
            except ValueError:
                pass
        self.selected_rows_eff = list(set(self.selected_rows_eff))
        self.selected_indexes_eff = list(
            self.model_eff._data.index[self.selected_rows_eff]
        )
        print(self.selected_indexes_eff)

    def eff_remove_selected(self):
        if len(self.selected_indexes_eff) > 0 and self.df_eff is not None:
            self.df_eff = self.df_eff.drop(self.selected_indexes_eff)
            self.df_eff.reset_index(drop=True, inplace=True)
            self.update_eff_table()
            self.plot_eff_points()
            self.button_fit1_eff.setChecked(False)
            self.button_fit2_eff.setChecked(False)

    def eff_fit1(self):
        if self.button_fit2_eff.isChecked():
            self.button_fit2_eff.setChecked(False)
            self.plot_eff_points()
        if self.df_eff is not None and self.button_fit1_eff.isChecked():
            if self.df_eff.shape[0] > 1:
                x = self.df_eff[f"Energy ({self.e_units})"]
                y = self.df_eff["Efficiency"] * 100
                y_err = self.df_eff["+/- Efficiency"]
                efficiency.eff_fit(
                    x, y, y_err, order=1, ax_fit=self.ax_eff, ax_res=self.ax_eff_res
                )
                self.ax_eff.set_xlabel(self.fit.x_units)
                self.ax_eff.set_ylabel("Efficiency [%]")
                self.fig_eff.canvas.draw_idle()
        elif self.df_eff is not None and not self.button_fit1_eff.isChecked():
            self.plot_eff_points()

    def eff_fit2(self):  # TODO
        if self.button_fit1_eff.isChecked():
            self.button_fit1_eff.setChecked(False)
            self.plot_eff_points()
        if self.df_eff is not None and self.button_fit2_eff.isChecked():
            if self.df_eff.shape[0] > 3:
                x = self.df_eff[f"Energy ({self.e_units})"]
                y = self.df_eff["Efficiency"] * 100
                y_err = self.df_eff["+/- Efficiency"]
                efficiency.eff_fit(
                    x, y, y_err, order=2, ax_fit=self.ax_eff, ax_res=self.ax_eff_res
                )
                self.ax_eff.set_xlabel(self.fit.x_units)
                self.ax_eff.set_ylabel("Efficiency [%]")
                self.fig_eff.canvas.draw_idle()
        elif self.df_eff is not None and not self.button_fit2_eff.isChecked():
            self.plot_eff_points()

    def eff_yscale(self):
        if self.eff_scale == "log":
            self.ax_eff.set_yscale("linear")
            self.eff_scale = "linear"
        else:
            self.ax_eff.set_yscale("log")
            self.eff_scale = "log"
        self.fig_eff.canvas.draw_idle()

    def get_eff_input(self):
        # which peak?
        if self.w_eff.button_first.isChecked():
            self.n_peak_eff = 0
        elif self.w_eff.button_second.isChecked():
            self.n_peak_eff = 1
        elif self.w_eff.button_third.isChecked():
            self.n_peak_eff = 2

        # t_half
        if self.w_eff.button_seconds.isChecked():
            t_fact = 1
        elif self.w_eff.button_minutes.isChecked():
            t_fact = 60
        elif self.w_eff.button_years.isChecked():
            t_fact = 365 * 24 * 3600
        else:
            msg = "ERROR: time units not determined"
            print(msg)
        self.t_half = float(self.w_eff.txt_t_half.text()) * t_fact
        # t_half sigma
        self.t_half_sig = float(self.w_eff.txt_t_half_sig.text())

        # A0
        if self.w_eff.button_Ci.isChecked():
            A_fact = 3.7e10
        elif self.w_eff.button_Bq.isChecked():
            A_fact = 1
        else:
            msg = "ERROR: activity units not determined"
            print(msg)
        self.A0 = float(self.w_eff.txt_activity.text()) * A_fact
        # A0 sigma
        self.A0_sig = float(self.w_eff.txt_activity_sig.text())

        # dates or time since production
        date_str0 = self.w_eff.txt_init_date.text()
        t_duration = self.w_eff.txt_t_duration.text()
        if date_str0 != " ":
            fmt_str = "%Y-%m-%d"
            date0 = datetime.datetime.strptime(date_str0, fmt_str)
            date_str1 = self.w_eff.txt_end_date.text()
            date1 = datetime.datetime.strptime(date_str1, fmt_str)
            delta_t = date1 - date0
            self.delta_t_sec = delta_t.days * 24 * 3600
        elif t_duration != " ":
            self.delta_t_sec = float(t_duration)
        else:
            msg = "ERROR: either dates or time since production must be specified"
            print(msg)
        self.delta_t_sig = float(self.w_eff.txt_t_duration_sig.text())

        # branching ratio
        self.B = float(self.w_eff.txt_B.text())
        self.B_sig = float(self.w_eff.txt_B_sig.text())

        # acquisition time
        self.acq_time = float(self.w_eff.txt_acq.text())
        self.acq_time_sig = float(self.w_eff.txt_acq_sig.text())

    ## Resolution (FWHM)
    def reset_fwhm_figure(self):
        self.fig_fwhm = self.resolution_plot.canvas.figure
        self.fig_fwhm.clear()
        self.fig_fwhm.set_constrained_layout(True)
        gs = self.fig_fwhm.add_gridspec(
            2, 2, width_ratios=[0.5, 0.5], height_ratios=[0.5, 0.5]
        )

        # axes
        self.ax_fwhm = self.fig_fwhm.add_subplot(gs[:, 0])
        self.ax_fwhm_gauss = self.fig_fwhm.add_subplot(gs[0, 1])
        self.ax_fwhm_tab = self.fig_fwhm.add_subplot(gs[1, 1])

        # remove ticks
        for ax in [self.ax_fwhm, self.ax_fwhm_tab, self.ax_fwhm_gauss]:
            ax.set_xticks([])
            ax.set_yticks([])

    @staticmethod
    def get_fwhm_vals(fit):
        fwhm_lst = []
        for d in fit.peak_info:
            keys = list(d)
            fwhm_val = d[keys[2]]
            fwhm_lst.append(fwhm_val)
        return fwhm_lst

    def which_button_fwhm(self):
        if self.button_fwhm1.isChecked():
            self.n_fwhm = 1
        elif self.button_fwhm2.isChecked():
            self.n_fwhm = 2
        else:
            self.n_fwhm = 1

    def perform_fwhm(self):
        try:
            self.which_button_fwhm()
            self.fwhm_fit = resolution.fwhm_vs_erg(
                energies=self.fwhm_x,
                fwhms=self.fwhm,
                x_units=self.spect.x_units,
                e_units=self.e_units,
                order=self.n_fwhm,
                fig=self.fig_fwhm,
                ax=self.ax_fwhm,
            )
            self.update_table_fwhm()
            self.update_gauss_fwhm()
            self.fig_fwhm.canvas.draw_idle()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def update_table_fwhm(self):
        self.ax_fwhm_tab.clear()
        resolution.fwhm_table(
            x_lst=self.fwhm_x,
            fwhm_lst=self.fwhm,
            e_units=self.e_units,
            ax=self.ax_fwhm_tab,
            fig=self.fig_fwhm,
        )

    def update_gauss_fwhm(self):
        self.ax_fwhm_gauss.clear()
        gauss_comp = pf.GaussianComponents(fit_obj_lst=self.fit_lst)
        gauss_comp.plot_gauss(plot_type="fwhm", fig=self.fig_fwhm, ax=self.ax_fwhm_gauss)

    def add_fwhm(self):
        try:
            self.ax_fwhm.clear()
            xvals = self.get_mean_vals()
            fwhms = self.get_fwhm_vals(self.fit)

            for x, f in zip(xvals, fwhms):
                if x not in self.fwhm_x and f not in self.fwhm:
                    self.fwhm_x.append(x)
                    self.fwhm.append(f)

            self.fit_lst.append(self.fit)
            self.fwhm_x.sort()
            self.fwhm.sort()
            self.perform_fwhm()
        except Exception as e:
            print("An unknown error occurred:", str(e))

    def update_fwhm_figure(self):
        self.ax_fwhm.clear()
        self.which_button_fwhm()
        self.perform_fwhm()

    def set_origin_fwhm(self):
        self.ax_fwhm.clear()
        if 0 in self.fwhm_x:
            self.fwhm_x.pop(0)
            self.fwhm.pop(0)
        else:
            self.fwhm_x.insert(0, 0)
            self.fwhm.insert(0, 0)

        if len(self.fwhm_x) > 1:
            self.perform_fwhm()

    def reset_fwhm(self):
        print("Reseting calibration")
        self.fit_lst = []
        self.fwhm_x = [0]
        self.fwhm = [0]
        for ax in [self.ax_fwhm, self.ax_fwhm_tab, self.ax_fwhm_gauss]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        self.fig_fwhm.canvas.draw_idle()

    def extrapolate_fwhm(self):
        try:
            if self.button_extrapolate.isChecked():
                resolution.fwhm_extrapolate(
                    energies=self.spect.x,
                    fit=self.fwhm_fit,
                    order=self.n_fwhm,
                    ax=self.ax_fwhm,
                    fig=self.fig_fwhm,
                )
                self.fig_fwhm.canvas.draw_idle()
            else:
                self.update_fwhm_figure()
        except Exception as e:
            print("An unknown error occurred:", str(e))
