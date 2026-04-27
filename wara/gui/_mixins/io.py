import traceback
from copy import deepcopy
from PyQt5.QtWidgets import QFileDialog
from wara import file_reader
from wara import spectrum as sp


class IOMixin:

    def saveReport(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save", "", "All Files (*);;Text Files (*.txt)", options=options
        )
        if fileName and len(self.list_xrange) != 0:
            print("Report file: ", fileName)
            self.save_fit(fileName)

    def save_fit(self, fileName):
        try:
            res = self.fit.fit_result
            report = res.fit_report()
            fn = f"Data_filename: {self.fileName}\n"
            xunits = f"x_units: {self.spect.x_units}\n"
            ROI = f"ROI_start_stop: {self.idxmin_fit}, {self.idxmax_fit}\n"
            npeaks = f"Num_peaks: {len(self.fit.peak_info)}\n"
            bkgd = f"Background_type: {self.bg}\n"
            report2 = fn + xunits + ROI + npeaks + bkgd + "\n" + report

            with open(f"{fileName}", "w") as text_file:
                text_file.write(report2)
        except Exception:
            print("Cannot save file with fit info")
            traceback.print_exc()

    def save_spect(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save", "", "All Files (*);", options=options
        )
        try:
            if fileName[-4:] == ".txt":
                self.spect.to_txt(fileName)
            else:
                self.spect.to_csv(fileName)
            print("Saved file: ", fileName)
        except Exception:
            print("Cannot save file")
            traceback.print_exc()

    def save_ID_peaks(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save", "", "All Files (*);;Text Files (*.txt)", options=options
        )
        try:
            print("Saved file: ", fileName)
            df = self.df_isotID_selected
            if ".csv" in fileName:
                df.to_csv(f"{fileName}", index=False)
            else:
                df.to_csv(f"{fileName}.csv", index=False)
        except Exception:
            print("Cannot save file with peak info")
            traceback.print_exc()

    def save_multiple_spectra(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save", "", "All Files (*);", options=options
        )
        try:
            print("Saved file: ", fileName)
            df = self.diag_data.create_data_frame()
            if ".csv" in fileName:
                df.to_csv(f"{fileName}", index=False)
            else:
                df.to_csv(f"{fileName}.csv", index=False)
        except Exception:
            print("ERROR: Cannot save file")
            traceback.print_exc()

    def try_display_info_file(self):
        try:
            self.display_info_file()
        except Exception:
            print("Make sure to load a spectrum file.")
            traceback.print_exc()

    def display_info_file(self):
        from ..dialogs import WindowInfoFile
        self.w_info_file = WindowInfoFile()
        self.w_info_file.show()
        tot_cts = self.spect.counts.sum()
        nch = self.spect.counts.shape[0]
        self.w_info_file.label_tc.setText(str(tot_cts))
        self.w_info_file.label_nch.setText(str(nch))
        if self.spect.description is None:
            self.w_info_file.label_descript.setText("N/A")
        else:
            self.w_info_file.label_descript.setText(str(self.spect.description))
        if self.spect.label is None:
            self.w_info_file.label_plot.setText("N/A")
        else:
            self.w_info_file.label_plot.setText(str(self.spect.label))
        if self.spect.livetime is None:
            self.w_info_file.label_lt.setText("N/A")
        else:
            count_rate = tot_cts / self.spect.livetime
            self.w_info_file.label_lt.setText(str(self.spect.livetime))
            self.w_info_file.label_cr.setText(str(count_rate))
        if self.spect.realtime is None:
            self.w_info_file.label_rt.setText("N/A")
        else:
            self.w_info_file.label_rt.setText(str(self.spect.realtime))
        if self.spect.livetime is None or self.spect.realtime is None:
            self.w_info_file.label_dt.setText("N/A")
            self.w_info_file.label_cr.setText("N/A")
        else:
            deadtime = round((1 - self.spect.livetime / self.spect.realtime) * 100, 3)
            self.w_info_file.label_dt.setText(str(deadtime))
        if self.spect.acq_date is None:
            self.w_info_file.label_date.setText("N/A")
        else:
            self.w_info_file.label_date.setText(str(self.spect.acq_date))
        if self.spect.energy_cal is None:
            self.w_info_file.label_energy_cal.setText("N/A")
        else:
            self.w_info_file.label_energy_cal.setText(str(self.spect.energy_cal))

    def reset_spectrum(self):
        try:
            self.reset_spect_params()
            self.e_units = deepcopy(self.e_units_orig)
            self.spect = deepcopy(self.spect_orig)
            self.create_graph(fit=False, reset=True)
        except Exception:
            print("Could not reset file")
            traceback.print_exc()

    def load_spe_file(self):
        try:
            self.fileName, self.e_units, self.spect = self.open_file()
            self.e_units_orig = deepcopy(self.e_units)
            self.spect_orig = deepcopy(self.spect)
            self.create_graph(fit=False, reset=False)
            self.setWindowTitle(f"wara: {self.fileName}")
        except Exception:
            print("File could not be opened")
            traceback.print_exc()

    def open_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.folder_path:
            print(f"Selected folder path: {self.folder_path}")
        try:
            self.initialize_plots_diagnostics()
            self.load_diagnostic_data()
            self.plot_diagnostic_total()
        except Exception:
            print("Could not open folder")
            traceback.print_exc()

    def open_file(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open spectrum file",
            "",
            "All Files (*);;",
            options=options,
        )

        if fileName == "":
            pass
        else:
            try:
                print("Opening file: ", fileName)
                fileName_lc = fileName.lower()
                if fileName_lc[-4:] == ".csv":
                    try:
                        spect = file_reader.read_csv(fileName)
                        e_units = spect.e_units
                    except Exception:
                        self.lynxcsv = file_reader.ReadLynxCsv(fileName)
                        spect = self.lynxcsv.spect
                        e_units = spect.e_units

                elif fileName_lc[-4:] == ".cnf":
                    spect = file_reader.read_cnf(fileName)
                    e_units = spect.e_units
                elif fileName_lc[-4:] == ".mca":
                    spect = file_reader.read_mca(fileName)
                    e_units = "channels"
                elif fileName_lc[-4:] == ".spe":
                    spect = file_reader.read_spe(fileName)
                    e_units = "channels"
                elif fileName_lc[-8:] == ".pha.txt":
                    spect = file_reader.read_multiscan(fileName)
                    e_units = spect.e_units

                elif fileName_lc[-4:] == ".txt" and fileName_lc[-8:] != ".pha.txt":
                    spect = file_reader.read_txt(fileName)
                    e_units = spect.e_units
                else:
                    print("Could not open file")
                    fileName = fileName + " ***INVALID FILE TYPE***"
            except Exception:
                print("Could not open file")
                traceback.print_exc()
        return fileName, e_units, spect
