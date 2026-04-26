import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from wara import energy_calibration as ecal
from wara import spectrum as sp
from ..table import TableModel


class CalibrationMixin:

    def cal_info_activate(self):
        self.w_info_cal.activateWindow()
        self.w_info_cal.show()

    def reset_cal_figure(self):
        self.fig_cal = self.cal_plot.canvas.figure
        self.fig_cal.clear()
        self.fig_cal.set_constrained_layout(True)
        gs2 = self.fig_cal.add_gridspec(2, 1, height_ratios=[0.3, 1.5])

        # axes
        self.ax_cal = self.fig_cal.add_subplot(gs2[1, :])
        self.ax_cal_res = self.fig_cal.add_subplot(gs2[0, :])

        # remove ticks
        self.ax_cal.set_xticks([])
        self.ax_cal.set_yticks([])
        self.ax_cal_res.set_xticks([])
        self.ax_cal_res.set_yticks([])

    def remove_cal(self):
        try:
            self.e_units = "channels"
            self.spect.remove_calibration()
            self.create_graph(fit=False, reset=True)
            self.search = 0
        except Exception as e:
            print("Cannot remove calibration")
            print("An unknown error occurred:", str(e))

    def get_mean_vals(self):
        mean_lst = []
        for d in self.fit.peak_info:
            keys = list(d)
            mean_ch = d[keys[0]]
            if self.spect.energies is not None:
                ix_ch = np.where(self.spect.energies >= mean_ch)[0][0]
                mean_ch = self.spect.channels[ix_ch]
            mean_lst.append(mean_ch)
        return mean_lst

    def append_to_list_not_repeated(self, list_e_in, list_m_in, list_e_out, list_m_out):
        for ch, e in zip(list_m_in, list_e_in):
            e2 = e.strip()  # remove white spaces
            if e2 == "_" or e2 == "-":
                next
            else:
                num_e = float(e2)
                if num_e not in list_e_out and ch not in list_m_out:
                    list_e_out.append(round(num_e, 3))
                    list_m_out.append(round(ch, 3))

    def perform_cal(self, df):
        self.ax_cal_res.clear()
        self.ax_cal.clear()
        self.cal = ecal.EnergyCalibration(mean_vals=df["Centroid"],
                                          erg=df[f"Energy ({self.cal_e_units})"],
                                          channels=self.spect.channels,
                                          n=self.n,
                                          e_units=self.e_units)
        self.cal.plot(residual=True, ax_fit=self.ax_cal, ax_res=self.ax_cal_res)
        # C0, C1,...
        coeffs = list(self.cal.fit.best_values.values())
        coeffs_str = []
        for i, c in enumerate(coeffs):
            coeffs_str.append(f"{c:.4E}ch^{i}")
        self.ecal_eqn = " + ".join(coeffs_str)

    def add_point_dont_fit(self, erg_lst=None, ch_lst=None):
        if erg_lst is not None and ch_lst is not None:
            en, mean_lst = self.retrieve_evals()
            self.append_to_list_not_repeated(
                en, mean_lst, self.e_vals_not_fit, self.mean_vals_not_fit
            )
        self.df_ecal_not_fit = pd.DataFrame()
        self.df_ecal_not_fit["Centroid"] = self.mean_vals_not_fit
        self.df_ecal_not_fit[f"Energy ({self.cal_e_units})"] = self.e_vals_not_fit
        self.df_ecal_not_fit[f"Sigma ({self.cal_e_units})"] = [0] * len(
            self.mean_vals_not_fit
        )
        if self.df_ecal_fit is not None:
            self.df_ecal = pd.concat(
                [self.df_ecal_fit, self.df_ecal_not_fit], ignore_index=True
            )
        else:
            self.df_ecal = self.df_ecal_not_fit

    def plot_point_dont_fit(self):
        if len(self.mean_vals_not_fit) > 0:
            self.ax_cal.plot(
                self.mean_vals_not_fit,
                self.e_vals_not_fit,
                "o",
                color="black",
                ms=8,
                markerfacecolor="yellow",
                label="Not used for fit",
            )
            if self.flag_e_legend == 0:
                self.ax_cal.legend()
                self.flag_e_legend = 1
            self.fig_cal.canvas.draw_idle()

    def add_point_fit(self, erg_lst=None, ch_lst=None):
        if len(self.mean_vals) > 0:
            if erg_lst is not None and ch_lst is not None:
                en, mean_lst = self.retrieve_evals()
                self.append_to_list_not_repeated(
                    en, mean_lst, self.e_vals, self.mean_vals
                )
            self.df_ecal_fit = pd.DataFrame()
            self.df_ecal_fit["Centroid"] = self.mean_vals
            self.df_ecal_fit[f"Energy ({self.cal_e_units})"] = self.e_vals
            self.df_ecal_fit[f"Sigma ({self.cal_e_units})"] = [0] * len(self.mean_vals)
            self.perform_cal(self.df_ecal_fit)
            self.df_ecal_fit[f"Sigma ({self.cal_e_units})"] = list(
                self.cal.fit.eval_uncertainty()
            )
            if self.df_ecal_not_fit is not None:
                self.df_ecal = pd.concat(
                    [self.df_ecal_not_fit, self.df_ecal_fit], ignore_index=True
                )
            else:
                self.df_ecal = self.df_ecal_fit

    def add_cal(self):
        try:
            self.check_radioButton()
            self.flag_check_e_units = 1
            cols = [
                "Centroid",
                f"Energy ({self.cal_e_units})",
                f"Sigma ({self.cal_e_units})",
            ]
            self.df_ecal = pd.DataFrame(columns=cols)
            erg, mean_vals = self.retrieve_evals()
            if self.w_cal.checkBox_cal.isChecked():
                self.add_point_dont_fit(erg, mean_vals)
                self.plot_point_dont_fit()
            else:
                self.add_point_fit(erg, mean_vals)
            self.update_ecal_table()
            self.fig_cal.canvas.draw_idle()
        except Exception as e:
            print("Not valid entry for energy calibration")
            print("An unknown error occurred:", str(e))

    def retrieve_evals(self):
        erg = self.w_cal.energy_txt.text().split(",")
        mean_lst = self.get_mean_vals()
        return erg, mean_lst

    def new_window_cal(self):
        from ..dialogs import WindowCal
        self.w_cal = WindowCal()
        self.w_cal.show()
        self.w_cal.accepted.connect(self.add_cal)
        if self.flag_check_e_units == 1:
            if self.e_units == "eV":
                self.w_cal.radio_button_ev.setChecked(True)
            if self.e_units == "keV":
                self.w_cal.radio_button_kev.setChecked(True)
            if self.e_units == "MeV":
                self.w_cal.radio_button_mev.setChecked(True)
            self.disable_checkRadioButtons()
            txt = "Note: calibration has already been initialized"
            self.w_cal.label_ecal_warning.setText(txt)
            self.w_cal.label_ecal_warning.setStyleSheet("color: red ; font: 10")

    def check_radioButton(self):
        if self.w_cal.radio_button_ev.isChecked():
            self.e_units = "eV"
            self.cal_e_units = "eV"
        elif self.w_cal.radio_button_kev.isChecked():
            self.e_units = "keV"
            self.cal_e_units = "keV"
        elif self.w_cal.radio_button_mev.isChecked():
            self.e_units = "MeV"
            self.cal_e_units = "MeV"
        else:
            msg = "ERROR: energy units not determined. Will use channels instead"
            print(msg)

    def disable_checkRadioButtons(self):
        self.w_cal.radio_button_ev.setEnabled(False)
        self.w_cal.radio_button_kev.setEnabled(False)
        self.w_cal.radio_button_mev.setEnabled(False)

    def update_ecal_table(self):
        self.df_ecal.fillna(value=0, inplace=True)
        self.df_ecal.sort_values(by=["Centroid"], inplace=True, ignore_index=True)
        self.df_ecal = self.df_ecal.round(3)
        self.activate_ecal_table(data=self.df_ecal)

    def activate_ecal_table(self, data):
        self.selected_rows_ecal = []
        self.table_ecal = QtWidgets.QTableView()
        self.table_ecal.setAlternatingRowColors(True)
        self.table_ecal.setSortingEnabled(False)
        stylesheet_header = "::section{Background-color:lightgreen}"
        self.table_ecal.horizontalHeader().setStyleSheet(stylesheet_header)
        self.table_ecal.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.model_ecal = TableModel(data)
        ix_not_fit = self.get_index_not_fit_ecal_table()
        self.color_table_rows(ix_not_fit)
        self.table_ecal.setModel(self.model_ecal)
        self.table_ecal_area.setWidget(self.table_ecal)
        stylesheet_ix = "::section{Background-color:lightgoldenrodyellow}"
        self.table_ecal.setStyleSheet(stylesheet_ix)
        self.table_ecal.setColumnWidth(0, 150)
        self.table_ecal.setColumnWidth(1, 150)
        self.table_ecal.setColumnWidth(2, 150)
        self.table_ecal.selectionModel().selectionChanged.connect(
            self.on_selectionChanged_ecal
        )

    def get_index_not_fit_ecal_table(self):
        ix = []
        for row in self.df_ecal.index:
            erg = self.df_ecal.iloc[row][f"Energy ({self.cal_e_units})"]
            if erg in self.e_vals_not_fit:
                ix.append(row)
        return ix

    def color_table_rows(self, index):
        self.model_ecal.set_highlighted_rows(index)

    def on_selectionChanged_ecal(self, selected, deselected):
        for ix in selected.indexes():
            self.selected_rows_ecal.append(ix.row())
        for ix in deselected.indexes():
            try:
                self.selected_rows_ecal.remove(ix.row())
            except ValueError:
                pass
        self.selected_rows_ecal = list(set(self.selected_rows_ecal))
        self.selected_indexes_ecal = list(
            self.model_ecal._data.index[self.selected_rows_ecal]
        )

    def set_origin(self):
        self.ax_cal.clear()
        if 0 in self.mean_vals:
            self.mean_vals.pop(0)
            self.e_vals.pop(0)
        else:
            self.mean_vals.insert(0, 0)
            self.e_vals.insert(0, 0)

        if len(self.mean_vals) > 1:
            self.add_point_fit()
            self.plot_point_dont_fit()
            self.update_ecal_table()
            self.fig_cal.canvas.draw_idle()

    def which_button_cal(self):
        if self.button_cal1.isChecked():
            self.n = 1
        elif self.button_cal2.isChecked():
            self.n = 2
        elif self.button_cal3.isChecked():
            self.n = 3

    def update_cal(self):
        try:
            self.which_button_cal()
            self.ax_cal.clear()
            self.add_point_fit()
            self.add_point_dont_fit()
            self.plot_point_dont_fit()
            self.update_ecal_table()
            self.fig_cal.canvas.draw_idle()
        except Exception as e:
            print("Calibration data not updated")
            print("An unknown error occurred:", str(e))

    def new_window_cal_eqns(self):
        from ..dialogs import WindowCalEqns
        self.w_cal_eqns = WindowCalEqns()
        self.w_cal_eqns.show()
        self.w_cal_eqns.accepted.connect(self.add_cal_eqns)

    def add_cal_eqns(self):
        try:
            a_txt = self.w_cal_eqns.a.text()
            b_txt = self.w_cal_eqns.b.text()
            c_txt = self.w_cal_eqns.c.text()
            d_txt = self.w_cal_eqns.d.text()
            ch = self.spect.channels
            erg = 0
            self.ecal_eqn = ""
            if self.isevaluable(a_txt):
                erg = erg + eval(a_txt)
                self.ecal_eqn = self.ecal_eqn + f"{eval(a_txt)}"
            if self.isevaluable(b_txt):
                erg = erg + eval(b_txt) * ch
                self.ecal_eqn = self.ecal_eqn + f" + {eval(b_txt)}x"
            if self.isevaluable(c_txt):
                erg = erg + eval(c_txt) * ch**2
                self.ecal_eqn = self.ecal_eqn + f" + {eval(c_txt)}x^2"
            if self.isevaluable(d_txt):
                erg = erg + eval(d_txt) * ch**3
                self.ecal_eqn = self.ecal_eqn + f" + {eval(d_txt)}x^3"
            self.cal.predicted = erg
            self.which_button_cal_eqn()
            self.plot_cal_eqns(self.ecal_eqn)
        except Exception as e:
            print("Could not set calibration equations")
            print("An unknown error occurred:", str(e))

    def plot_cal_eqns(self, label):
        self.reset_cal()
        self.ax_cal.clear()
        self.ax_cal.plot(self.spect.channels, self.cal.predicted, lw=3, label=label)
        self.ax_cal.set_xlabel("Channels")
        self.ax_cal.set_ylabel("Energy (keV)")
        self.ax_cal.legend(fontsize=16)
        self.fig_cal.canvas.draw_idle()

    def which_button_cal_eqn(self):
        if self.w_cal_eqns.radio_button_ev.isChecked():
            self.cal_e_units = "eV"
        elif self.w_cal_eqns.radio_button_kev.isChecked():
            self.cal_e_units = "keV"
        elif self.w_cal_eqns.radio_button_mev.isChecked():
            self.cal_e_units = "MeV"

    def reset_cal(self):
        print("Reseting calibration")
        self.ax_cal.clear()
        self.ax_cal_res.clear()
        self.mean_vals = [0]
        self.e_vals = [0]
        self.mean_vals_not_fit = []
        self.e_vals_not_fit = []
        self.flag_check_e_units = 0
        cols = ["Centroid", "Energy", "Sigma"]
        self.df_ecal = pd.DataFrame(columns=cols)
        self.update_ecal_table()
        # remove ticks
        self.ax_cal.set_xticks([])
        self.ax_cal.set_yticks([])
        self.ax_cal_res.set_xticks([])
        self.ax_cal_res.set_yticks([])
        self.fig_cal.canvas.draw_idle()

    def apply_cal(self):
        try:
            if self.cal_e_units is None:
                e_units = self.spect.e_units
            else:
                e_units = self.cal_e_units

            self.spect = sp.Spectrum(
                counts=self.spect.counts,
                energies=self.cal.predicted,
                e_units=e_units,
                realtime=self.spect.realtime,
                livetime=self.spect.livetime,
                cps=self.spect.cps,
                acq_date=self.spect.acq_date,
                energy_cal=self.ecal_eqn,
                description=self.spect.description,
                label=self.spect.label,
            )
            self.create_graph(fit=False, reset=True)
            self.search = 0
        except Exception as e:
            print("Could not apply calibration")
            print("An error occurred:", str(e))

    def ecal_remove_selected(self):
        if (
            len(self.selected_rows_ecal) > 0
            and self.df_ecal is not None
            and self.df_ecal.shape[0] > 2
        ):
            for row in self.selected_rows_ecal:
                erg = self.df_ecal.iloc[row][f"Energy ({self.cal_e_units})"]
                val = self.df_ecal.iloc[row]["Centroid"]
                if erg in self.e_vals:
                    self.mean_vals.remove(val)
                    self.e_vals.remove(erg)
                elif erg in self.e_vals_not_fit:
                    self.mean_vals_not_fit.remove(val)
                    self.e_vals_not_fit.remove(erg)
                    if len(self.e_vals_not_fit) == 0:  # if empty, reset flag
                        self.flag_e_legend = 0
            if len(self.mean_vals) > 1:
                self.update_cal()

    def new_window_cal_add_point(self):
        from ..dialogs import WindowCalAddPoint
        self.w_cal_addpoint = WindowCalAddPoint()
        self.w_cal_addpoint.show()
        self.w_cal_addpoint.accepted.connect(self.ecal_add_point)

    def ecal_add_point(self):
        try:
            ch_txt = self.w_cal_addpoint.edit_channel.text().split(",")
            erg_txt = self.w_cal_addpoint.edit_energy.text().split(",")
            if len(ch_txt) == len(erg_txt):
                for ch, erg in zip(ch_txt, erg_txt):
                    self.mean_vals.append(eval(ch))
                    self.e_vals.append(eval(erg))
                self.update_cal()
            else:
                print("Channels and energies must be the same length")
        except Exception as e:
            print("Could not add point manually")
            print("An unknown error occurred:", str(e))
