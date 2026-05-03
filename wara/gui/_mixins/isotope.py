import traceback
import re
import pandas as pd
from importlib.resources import files
from PyQt5 import QtWidgets
from wara import parse_NIST
from wara import peaksearch as ps
from ..table import TableModel
from .. import theme


class IsotopeMixin:

    ## Peak finder
    def peakFind_info_activate(self):
        self.w_peak_find_info.activateWindow()
        self.w_peak_find_info.show()

    def activate_peak_finder(self):
        self.w_peak_find.activateWindow()
        txt = f"Xrange (optional) [{self.e_units}]:"
        self.w_peak_find.label_units.setText(txt)
        self.w_peak_find.show()

    def peakFind_kernel_apply(self):
        self.fwhm_at_0 = 1
        try:
            if self.w_peak_find.checkBox_km.isChecked():
                method = "km"
            else:
                method = "fast"
            self.min_snr = float(self.w_peak_find.edit_snr.text())
            self.ref_x = float(self.w_peak_find.edit_ref_ch.text())
            self.ref_fwhm = float(self.w_peak_find.edit_ref_fwhm.text())

            x0 = self.w_peak_find.edit_x0.text()
            x1 = self.w_peak_find.edit_x1.text()
            if x0 and x1:
                self.x0 = float(x0)
                self.x1 = float(x1)
                self.search = ps.PeakSearch(
                    self.spect,
                    self.ref_x,
                    self.ref_fwhm,
                    self.fwhm_at_0,
                    min_snr=self.min_snr,
                    xrange=[self.x0, self.x1],
                    method=method,
                )
            else:
                if method == "km" and len(self.spect.channels) < 9000:
                    self.search = ps.PeakSearch(
                        self.spect,
                        self.ref_x,
                        self.ref_fwhm,
                        self.fwhm_at_0,
                        min_snr=self.min_snr,
                        method=method,
                    )
                elif method == "fast":
                    self.search = ps.PeakSearch(
                        self.spect,
                        self.ref_x,
                        self.ref_fwhm,
                        self.fwhm_at_0,
                        min_snr=self.min_snr,
                        method=method,
                    )
                else:
                    pass
            if self.w_peak_find.checkBox_SNR.isChecked():
                self.snr_state = "on"
            else:
                self.snr_state = "off"
            self.create_graph(fit=True, reset=True)
        except Exception:
            print(
                "ERROR: non-numeric entry or if more than 9000 channels, "
                "constrain the range using x0 and x1."
            )
            traceback.print_exc()

    def peakFind_check_hpge(self):
        self.w_peak_find.edit_snr.setText("5")
        self.w_peak_find.edit_ref_ch.setText("420")
        self.w_peak_find.edit_ref_fwhm.setText("3")

    def peakFind_check_labr(self):
        self.w_peak_find.edit_snr.setText("5")
        self.w_peak_find.edit_ref_ch.setText("420")
        self.w_peak_find.edit_ref_fwhm.setText("12")

    def peakFind_check_nai(self):
        self.w_peak_find.edit_snr.setText("5")
        self.w_peak_find.edit_ref_ch.setText("420")
        self.w_peak_find.edit_ref_fwhm.setText("15")

    def peakFind_check_plastic(self):
        self.w_peak_find.edit_snr.setText("5")
        self.w_peak_find.edit_ref_ch.setText("420")
        self.w_peak_find.edit_ref_fwhm.setText("20")

    ## Isotope ID
    def isotID_info_activate(self):
        self.w_isotID_info.activateWindow()
        self.w_isotID_info.show()

    def activate_isotope_id(self):
        self.w_isot_id.activateWindow()
        self.w_isot_id.show()

    def isotID_apply(self):
        df_files = self.isotID_retrieve_data()
        if len(df_files) == 0:
            print("Cannot retrieve data")
        else:
            df_element = self.isotID_filter_by_element(df_files)
            df_energy = self.isotID_filter_by_energy(df_element)
            self.df_isotID = df_energy
            self.df_isotID = self.df_isotID.round(3)
            self.activate_table_gammas(self.df_isotID)

    def isotID_clear(self):
        "Clear entries, uncheck boxes"
        self.w_isot_id.edit_isot.clear()
        self.w_isot_id.edit_energy.clear()
        self.w_isot_id.edit_erange.clear()
        self.w_isot_id.lab_src.setChecked(False)
        self.w_isot_id.delayed_activation.setChecked(False)
        self.w_isot_id.natural_rad.setChecked(False)
        self.w_isot_id.neutron_capt.setChecked(False)
        self.w_isot_id.neutron_capt_IAEA.setChecked(False)
        self.w_isot_id.neutron_talys.setChecked(False)
        self.w_isot_id.neutron_inl_baghdad.setChecked(False)

    @staticmethod
    def join_gamma_files(files):
        # to be deleted
        data = pd.read_csv(files[0])
        if len(files) == 1:
            return data
        else:
            for f in files[1:]:
                df0 = pd.read_csv(f)
                data = pd.concat([data, df0], ignore_index=True)
            data["sort"] = data["Isotope"].str.extract(r"(\d+)", expand=False).astype(int)
            print(data.columns)
            data.sort_values("sort", inplace=True, ascending=True)
            data = data.drop("sort", axis=1)
            data.reset_index(inplace=True, drop=True)
            return data

    def isotID_retrieve_data(self):
        # Define the base directory once
        data_dir = files("wara").joinpath("nuclear-data")
        # Assign the files using the base directory
        lab_src_file     = str(data_dir.joinpath("Common_lab_sources.csv"))
        delay_act_file   = str(data_dir.joinpath("Delayed_activation_IAEA.csv"))
        nat_rad_file     = str(data_dir.joinpath("Natural_radiation.csv"))
        capt_file        = str(data_dir.joinpath("Capture_CapGam.csv"))
        capt_IAEA_file   = str(data_dir.joinpath("Capture_IAEA.csv"))
        talys_file       = str(data_dir.joinpath("Talys-14MeV.csv"))
        inl_baghdad_file = str(data_dir.joinpath("Inelastic_Baghdad.csv"))
        file = 0
        if self.w_isot_id.lab_src.isChecked():
            file = lab_src_file
        elif self.w_isot_id.delayed_activation.isChecked():
            file = delay_act_file
        elif self.w_isot_id.natural_rad.isChecked():
            file = nat_rad_file
        elif self.w_isot_id.neutron_capt.isChecked():
            file = capt_file
        elif self.w_isot_id.neutron_capt_IAEA.isChecked():
            file = capt_IAEA_file
        elif self.w_isot_id.neutron_talys.isChecked():
            file = talys_file
        elif self.w_isot_id.neutron_inl_baghdad.isChecked():
            file = inl_baghdad_file
        else:
            print("ERROR: Select a database from the menu")
        if file == 0:
            return []
        else:
            data = pd.read_csv(file)
            if "Info" not in data.columns:
                data["Info"] = ""
            return data

    def isotID_filter_by_element(self, df):
        elements = self.w_isot_id.edit_isot.text()
        if elements == "":
            return df
        elements_lst = elements.split(",")
        ixs = []  # indices
        for el in elements_lst:
            elm = el.strip(" ").lower()
            isot = re.findall(r"\d+", elm)
            if len(isot) == 0:
                ix = list(
                    df.index[df["Isotope"].str.match(f"(\\d+){elm}(\\b)", case=False)]
                )
                ixs.append(ix)
            else:
                # Normalize element-first formats (co60, co-60) to mass-first (60co)
                match = re.match(r"^([a-zA-Z]+)-?(\d+)$", elm)
                if match:
                    elm = match.group(2) + match.group(1)
                ix = list(df.index[df["Isotope"].str.lower() == elm])
                ixs.append(ix)
        ixs_flat = [item for sublist in ixs for item in sublist]
        ixs_uniq = sorted(list(set(ixs_flat)))
        df = df.loc[ixs_uniq]
        df.reset_index(inplace=True, drop=True)
        return df

    @staticmethod
    def isevaluable(s):
        "Check if eval(s) is possible"
        try:
            eval(s)
            return True
        except Exception:
            return False

    def isotID_filter_by_energy(self, df):
        energy_txt = self.w_isot_id.edit_energy.text()
        if energy_txt == "" or self.isevaluable(energy_txt) is False:
            return df
        energy = eval(energy_txt)
        erange_txt = self.w_isot_id.edit_erange.text()
        if erange_txt == "" or self.isevaluable(erange_txt) is False:
            erange = 0.01  # keV
        else:
            erange = eval(erange_txt)
        filt = (df["Energy (keV)"] > energy - erange) & (
            df["Energy (keV)"] < energy + erange
        )
        df = df[filt]
        df.reset_index(inplace=True, drop=True)
        return df

    ## list of gamma rays
    def activate_table_gammas(self, data):
        self.selected_rows = []
        self.scroll = self.w_isot_id.scrollArea_gammas
        self.table = QtWidgets.QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStyleSheet(theme.TABLE_HEADER_STYLE)
        self.table.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self.model = TableModel(data)
        self.table.setModel(self.model)
        self.scroll.setWidget(self.table)
        self.table.selectionModel().selectionChanged.connect(self.on_selectionChanged)

    def on_selectionChanged(self, selected, deselected):
        for ix in selected.indexes():
            self.selected_rows.append(ix.row())
        for ix in deselected.indexes():
            try:
                self.selected_rows.remove(ix.row())
            except ValueError:
                pass
        self.selected_rows = list(set(self.selected_rows))
        self.selected_indexes = list(self.model._data.index[self.selected_rows])

    def isotID_getColor(self):
        if self.w_isot_id.button_blue.isChecked():
            return "blue"
        elif self.w_isot_id.button_orange.isChecked():
            return "C1"
        elif self.w_isot_id.button_green.isChecked():
            return "green"
        elif self.w_isot_id.button_red.isChecked():
            return "red"
        return "blue"

    def isotID_plot_vlines(self):
        "Plot vertical lines of energies of selected rows"
        if len(self.df_isotID) == 0:  # do nothing if empty
            pass
        else:
            if len(self.selected_rows) == 0:  # do all if none selected
                df_candidates = self.df_isotID.copy()
            else:
                df_candidates = self.df_isotID.loc[self.selected_indexes].copy()

            # Filter out entries already plotted
            if len(self.df_isotID_selected) > 0:
                already = set(
                    zip(
                        self.df_isotID_selected["Isotope"],
                        self.df_isotID_selected["Energy (keV)"],
                    )
                )
                mask = [
                    (iso, e) not in already
                    for iso, e in zip(df_candidates["Isotope"], df_candidates["Energy (keV)"])
                ]
                self.df_isotID_plot = df_candidates[mask].reset_index(drop=True)
            else:
                self.df_isotID_plot = df_candidates

            self.df_isotID_selected = pd.concat(
                [self.df_isotID_selected, self.df_isotID_plot], ignore_index=True
            )

            sep_checked = self.w_isot_id.checkBox_sep.isChecked()
            dep_checked = self.w_isot_id.checkBox_dep.isChecked()
            compton_edge = self.w_isot_id.checkBox_CE.isChecked()
            for row in self.df_isotID_plot.index:
                isot0 = self.df_isotID_plot.loc[row, "Isotope"]
                e0 = self.df_isotID_plot.loc[row, "Energy (keV)"]
                info0 = self.df_isotID_plot.loc[row, "Info"]

                photopeak = self.ax_main.axvline(
                    x=e0,
                    color=self.isotID_getColor(),
                    linestyle="-",
                    alpha=0.5,
                    label=f"{isot0}{info0} : {round(e0,5)} keV",
                )
                if sep_checked and e0 > 1022:
                    e1 = e0 - 511
                    sep = self.ax_main.axvline(
                        x=e1,
                        color=self.isotID_getColor(),
                        linestyle="--",
                        alpha=0.5,
                        label=f"{isot0}{info0} : {round(e1,5)} keV (SEP)",
                    )
                    self.isotID_vlines.append(sep)
                if dep_checked and e0 > 1022:
                    e11 = e0 - 511 * 2
                    dep = self.ax_main.axvline(
                        x=e11,
                        color=self.isotID_getColor(),
                        linestyle="-.",
                        alpha=0.5,
                        label=f"{isot0}{info0} : {round(e11,5)} keV (DEP)",
                    )
                    self.isotID_vlines.append(dep)

                if compton_edge:
                    e111 = e0 - e0 / (1 + 2 * e0 / 511)
                    ce = self.ax_main.axvline(
                        x=e111,
                        color=self.isotID_getColor(),
                        linestyle=":",
                        alpha=0.5,
                        label=f"{isot0}{info0} : {round(e111,5)} keV (CE)",
                    )
                    self.isotID_vlines.append(ce)

                self.isotID_vlines.append(photopeak)
                if len(self.isotID_vlines) > 300:
                    print("Cannot display more than 300 vertical lines.")
                    break
            if self.w_isot_id.checkBox_labels.isChecked():
                self.ax_main.legend()
            self.fig.canvas.draw_idle()

    def isotID_remove_vlines(self):
        if len(self.isotID_vlines) != 0:
            self.df_isotID_selected = pd.DataFrame()
            for line in self.isotID_vlines:
                line.remove()
            self.isotID_vlines = []
            self.ax_main.legend()
            self.fig.canvas.draw_idle()

    def isotID_textSearch(self):
        element = self.w_isot_id.edit_element_search.text()
        if element == "":
            pass
        else:
            out_text = parse_NIST.isotopic_abundance_str(element)
            self.w_isot_id.text_search.setText(out_text)
