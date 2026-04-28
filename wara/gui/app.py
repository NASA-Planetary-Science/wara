"""
Usage:
  wara <file_name> [options]
  wara [options]

  options:
      -o                        open a blank window
      --fwhm_at_0=<fwhm0>       fwhm value at x=0
      --min_snr=<msnr>          min SNR
      --ref_x=<xref>            x reference for fwhm_ref
      --ref_fwhm=<ref_fwhm>     fwhm ref corresponding to x_ref
      --cebr                    detector type (cerium bromide)
      --labr                    detector type (lanthanum bromide)
      --hpge                    detector type (HPGe)


Reads a csv file with the following column format: counts | energy_EUNITS,
where EUNITS can be for examle keV or MeV. It can also read a CSV file with a
single column named "counts". No need to have channels because
they are automatically infered starting from channel = 0.

If detector type is defined e.g. --cebr then the code guesses the x_ref and
fwhm_ref based on the known detector characteristics.

Note that the detector type input parameters must be changed depending on the
particular electronic gain used. The examples here are for our specific
detector configurations.
"""
import time
import traceback
import docopt
import pandas as pd
import matplotlib.pyplot as plt
from importlib.resources import files
from PyQt5.QtWidgets import QApplication, QMainWindow, QButtonGroup, QSplashScreen
from PyQt5.uic import loadUi
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from wara import param_handle

from .dialogs import (
    WindowMainInfo,
    WindowCust,
    WindowCustInfo,
    WindowAddSubtract,
    WindowAddSubtractInfo,
    WindowPeakFinder,
    WindowPeakFinderInfo,
    WindowCalInfo,
    WindowEffInfo,
    WindowAdvFit,
    WindowAPImca,
    WindowAPIbin,
    WindowIsotID,
    WindowIsotIDInfo,
    WindowPNGInfo,
)
from ._mixins.spectrum import SpectrumMixin
from ._mixins.calibration import CalibrationMixin
from ._mixins.efficiency import EfficiencyMixin
from ._mixins.diagnostics import DiagnosticsMixin
from ._mixins.api import ApiMixin
from ._mixins.png import PngMixin
from ._mixins.isotope import IsotopeMixin
from ._mixins.io import IOMixin


class WaraApp(
    SpectrumMixin,
    CalibrationMixin,
    EfficiencyMixin,
    DiagnosticsMixin,
    ApiMixin,
    PngMixin,
    IsotopeMixin,
    IOMixin,
    QMainWindow,
):
    def __init__(self, commands):
        super().__init__()
        ui_file = str(files("wara").joinpath("ui/qt_gui.ui"))
        loadUi(ui_file, self)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowTitle("wara")
        icon_file = str(files("wara").joinpath("ui/wara-logo.png"))
        self.setWindowIcon(QtGui.QIcon(icon_file))
        self.scale = "linear"
        self.snr_state = "off"
        self.pushButton_scale.clicked.connect(self.update_scale)
        self.pushButton_scale.setStyleSheet("background-color : lightgoldenrodyellow")
        self.button_add_peak.setStyleSheet("background-color : silver")
        self.button_add_peak.setCheckable(True)
        self.button_add_peak.clicked.connect(self.add_peak)
        # push buttons
        self.enable_fitButtons(False)
        self.bg = "poly1"
        self.e_units = None
        self.pushButton_poly1.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly1.setCheckable(True)
        self.pushButton_poly1.setChecked(True)
        self.pushButton_poly2.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly2.setCheckable(True)
        self.pushButton_poly3.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly3.setCheckable(True)
        self.pushButton_poly4.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly4.setCheckable(True)
        self.pushButton_poly5.setStyleSheet("background-color : lightgreen")
        self.pushButton_poly5.setCheckable(True)
        self.pushButton_exp.setStyleSheet("background-color : lightgreen")
        self.pushButton_exp.setCheckable(True)
        self.btn_grp = QButtonGroup()
        self.btn_grp.setExclusive(True)
        self.btn_grp.addButton(self.pushButton_poly1)
        self.btn_grp.addButton(self.pushButton_poly2)
        self.btn_grp.addButton(self.pushButton_poly3)
        self.btn_grp.addButton(self.pushButton_poly4)
        self.btn_grp.addButton(self.pushButton_poly5)
        self.btn_grp.addButton(self.pushButton_exp)
        self.btn_grp.buttonClicked.connect(self.update_poly)

        # info button (main window)
        self.w_info_main = WindowMainInfo()
        self.button_info_main.clicked.connect(self.main_info_activate)

        # reset main figure
        self.button_reset.clicked.connect(self.reset_spectrum)

        # Gaussian or skewed Gaussian?
        self.sk_gauss = False
        self.pushButton_gauss.setStyleSheet("background-color : silver")
        self.pushButton_gauss.setCheckable(True)
        self.pushButton_gauss.setChecked(True)
        self.pushButton_gauss2.setStyleSheet("background-color : silver")
        self.pushButton_gauss2.setCheckable(True)
        self.btn_grp_gauss = QButtonGroup()
        self.btn_grp_gauss.setExclusive(True)
        self.btn_grp_gauss.addButton(self.pushButton_gauss)
        self.btn_grp_gauss.addButton(self.pushButton_gauss2)
        self.btn_grp_gauss.buttonClicked.connect(self.update_gauss)
        self.button_remove_cal.setStyleSheet("background-color : lightcoral")

        # customize
        self.w_cust = WindowCust()
        self.button_customize.setStyleSheet("background-color : lightsteelblue")
        self.button_customize.clicked.connect(self.new_window_custom)
        self.w_cust_info = WindowCustInfo()
        self.w_cust.button_info.clicked.connect(self.customize_info_activate)
        self.w_cust.button_apply_lab.clicked.connect(self.try_cust_labels)
        self.w_cust.button_apply_smooth.clicked.connect(self.try_cust_smooth)
        self.w_cust.button_apply_CR.clicked.connect(self.try_cust_countRate)
        self.w_cust.button_apply_shift.clicked.connect(self.try_cust_shift)

        # add/subtract spectra
        self.button_add_subtract.clicked.connect(self.activate_window_add_subtract)
        self.w_add_sub = WindowAddSubtract()
        self.w_add_sub.button_load1.clicked.connect(self.load_file1)
        self.w_add_sub.button_load2.clicked.connect(self.load_file2)
        self.w_add_sub.button_plot.clicked.connect(self.add_sub_plot)
        self.w_add_sub_info = WindowAddSubtractInfo()
        self.w_add_sub.button_info.clicked.connect(self.add_sub_info_activate)

        # navigation toolbars
        self.toolbars = []
        for tb in [
            self.main_plot,
            self.cal_plot,
            self.efficiency_plot,
            self.resolution_plot,
            self.api_plot,
            self.diag_plot00,
            self.plot_dt_png,
        ]:
            tmp = NavigationToolbar(tb.canvas, self)
            tmp.setVisible(False)
            tmp.setStyleSheet("font-size: 24px; background-color : wheat")
            self.addToolBar(tmp)
            self.toolbars.append(tmp)
        self.toolbars[0].setVisible(True)

        self.tabWidget.currentChanged.connect(self.switch_toolbar)
        self.tabWidget.setStyleSheet("background-color : whitesmoke")

        # menu bar
        self.saveFitReport.triggered.connect(self.saveReport)
        self.openFile.triggered.connect(self.load_spe_file)
        self.saveSpect.triggered.connect(self.save_spect)
        self.saveIDpeaks.triggered.connect(self.save_ID_peaks)
        self.info_file.triggered.connect(self.try_display_info_file)
        self.clearSpectrum.triggered.connect(self.clear_spectrum)

        self.initialize_main_figure()
        self.reset_main_figure()
        self.reset_spect_params()
        # remove ticks
        self.remove_ticks_fit()

        self.list_xrange = []
        self.commands = commands
        self.search = 0  # initialize dummy search object
        self.spect_orig = None
        self.e_units_orig = None
        self.fileName = None
        # peak fitting

        get_input = param_handle.get_spect_search(self.commands)
        if get_input is not None:
            spect, search, ref_x, fwhm_at_0, ref_fwhm = get_input
            self.spect = spect
            self.e_units = self.spect.e_units
            self.fileName = self.commands["<file_name>"]
            if search is not None:
                self.search = search
                self.ref_x = ref_x
                self.fwhm_at_0 = fwhm_at_0
                self.ref_fwhm = ref_fwhm
                self.create_graph(fit=True, reset=True)
            else:
                self.create_graph(fit=False, reset=True)
        if self.commands["--min_snr"] is not None:
            self.min_snr = float(self.commands["--min_snr"])
        else:
            self.min_snr = 3.0
            print("Welcome to Wara GUI")

        self.button_remove_cal.clicked.connect(self.remove_cal)
        # peak search range
        self.x0 = None
        self.x1 = None

        # advanced fitting
        # TODO
        self.parea = None
        self.w_adv_fit = WindowAdvFit()
        self.button_adv_fit.clicked.connect(self.open_adv_fit)
        self.w_adv_fit.button_perform_fit.clicked.connect(self.calculate_peak_area)

        ## energy calibration
        self.mean_vals = [0]
        self.e_vals = [0]
        self.e_sigs = [0]
        self.pred_erg = 0
        self.mean_vals_not_fit = []
        self.e_vals_not_fit = []
        self.selected_rows_ecal = []
        self.cal_e_units = None
        self.df_ecal = None
        self.df_ecal_not_fit = None
        self.df_ecal_fit = None
        self.flag_check_e_units = 0
        self.flag_e_legend = 0
        self.ecal_eqn = None

        self.w_info_cal = WindowCalInfo()
        self.button_info_cal.clicked.connect(self.cal_info_activate)

        # push butons
        self.button_add_cal.setStyleSheet("background-color : lightblue")
        self.button_origin.setStyleSheet("background-color : lightgoldenrodyellow")
        self.button_apply_cal.setStyleSheet("background-color : sandybrown")
        self.button_reset_cal.setStyleSheet("background-color : lightcoral")
        self.button_cal1.setStyleSheet("background-color : lightgreen")
        self.button_cal2.setStyleSheet("background-color : lightgreen")
        self.button_cal3.setStyleSheet("background-color : lightgreen")
        self.button_cal_eqns.setStyleSheet("background-color : silver")
        self.button_origin.clicked.connect(self.set_origin)
        self.button_remove_selected.clicked.connect(self.ecal_remove_selected)

        self.button_cal1.setCheckable(True)
        self.button_cal2.setCheckable(True)
        self.button_cal3.setCheckable(True)

        self.btn_grp2 = QButtonGroup()
        self.btn_grp2.setExclusive(True)
        self.btn_grp2.addButton(self.button_cal1)
        self.btn_grp2.addButton(self.button_cal2)
        self.btn_grp2.addButton(self.button_cal3)
        self.btn_grp2.buttonClicked.connect(self.update_cal)
        self.n = 1

        self.reset_cal_figure()

        self.button_add_cal.clicked.connect(self.new_window_cal)
        # reset calibration
        self.button_reset_cal.clicked.connect(self.reset_cal)
        # apply calibration
        self.button_apply_cal.clicked.connect(self.apply_cal)
        # set calibration equation
        self.button_cal_eqns.clicked.connect(self.new_window_cal_eqns)
        # Add calibration point
        self.button_cal_add_point.clicked.connect(self.new_window_cal_add_point)

        ## Resolution
        # push butons
        self.button_fwhm1.setStyleSheet("background-color : lightblue")
        self.button_fwhm2.setStyleSheet("background-color : lightblue")
        self.button_add_fwhm.setStyleSheet("background-color : khaki")
        self.button_origin_fwhm.setStyleSheet("background-color : lightgoldenrodyellow")
        self.button_reset_fwhm.setStyleSheet("background-color : lightcoral")
        self.button_extrapolate.setStyleSheet("background-color : lightgreen")

        self.button_fwhm1.setCheckable(True)
        self.button_fwhm2.setCheckable(True)
        self.button_extrapolate.setCheckable(True)

        self.btn_grp3 = QButtonGroup()
        self.btn_grp3.setExclusive(True)
        self.btn_grp3.addButton(self.button_fwhm1)
        self.btn_grp3.addButton(self.button_fwhm2)

        self.btn_grp3.buttonClicked.connect(self.update_fwhm_figure)
        self.reset_fwhm_figure()
        self.fit_lst = []

        self.button_add_fwhm.clicked.connect(self.add_fwhm)
        self.button_origin_fwhm.clicked.connect(self.set_origin_fwhm)
        self.button_reset_fwhm.clicked.connect(self.reset_fwhm)
        self.button_extrapolate.clicked.connect(self.extrapolate_fwhm)

        self.fwhm_x = [0]
        self.fwhm = [0]

        ## Efficiency
        self.eff_vals = []
        self.selected_rows_eff = []
        self.eff_scale = "log"
        self.df_eff = None
        # push butons
        self.w_info_eff = WindowEffInfo()
        self.button_info_eff.clicked.connect(self.eff_info_activate)
        self.button_yscale_eff.setStyleSheet("background-color : lightgoldenrodyellow")
        self.button_yscale_eff.clicked.connect(self.eff_yscale)
        self.button_reset_eff.setStyleSheet("background-color : lightcoral")
        self.button_reset_eff.clicked.connect(self.reset_eff_all)
        self.button_fit1_eff.setStyleSheet("background-color : lightgreen")
        self.button_fit1_eff.clicked.connect(self.eff_fit1)
        self.button_fit2_eff.setStyleSheet("background-color : lightgreen")
        self.button_fit2_eff.clicked.connect(self.eff_fit2)
        self.button_add_eff.setStyleSheet("background-color : sandybrown")
        self.button_add_eff.clicked.connect(self.new_window_eff)
        self.button_remove_selected_eff.clicked.connect(self.eff_remove_selected)
        self.reset_eff_figure()

        ## Diagnostics
        self.folder_path = None
        self.button_cts_cr.setEnabled(False)
        self.button_cts_cr_fit.setEnabled(False)
        self.button_centroid_max.setEnabled(False)
        self.button_combine_send.setEnabled(False)
        self.button_diag_save.setEnabled(False)
        self.button_load_folder.clicked.connect(self.open_folder)
        self.button_fit_diag.clicked.connect(
            lambda: self.try_fit_peaks_diagnostics(integral=False)
        )
        self.button_integral_diag.clicked.connect(
            lambda: self.try_fit_peaks_diagnostics(integral=True)
        )
        self.button_diag_clear.clicked.connect(self.initialize_plots_diagnostics)
        self.button_cts_cr.clicked.connect(self.change_cts_cr_diag)
        self.button_cts_cr_fit.clicked.connect(self.change_cts_cr_fit_diag)
        self.button_centroid_max.clicked.connect(self.change_centroid_max_diag)
        self.button_combine_send.clicked.connect(self.combine_and_send_to_spect)
        self.button_diag_save.clicked.connect(self.save_multiple_spectra)

        ## API
        self.api_spect_scale = "linear"
        self.api_xy_scale = "linear"
        self.api_yscale.clicked.connect(self.api_spect_yscale)
        self.api_button_logxy.setCheckable(True)
        self.api_button_logxy.clicked.connect(self.api_xylog)
        self.api_vmax_txt.returnPressed.connect(self.api_on_returnXY)
        self.api_send_to_spect.clicked.connect(self.send_to_spect_api)
        self.api_load_file.clicked.connect(self.activate_load_api_file)
        self.api_reset.clicked.connect(self.reset_button_api)
        self.api_button_filters.clicked.connect(self.api_window_filters)
        # API 3D
        self.button_api_3D.clicked.connect(self.open_api_3D)

        # API MCA
        self.w_api_mca = WindowAPImca()
        self.button_api_mca.clicked.connect(self.open_api_mca)
        self.w_api_mca.button_apply_all.clicked.connect(self.activate_api_mca)
        self.w_api_mca.button_apply_select.clicked.connect(self.activate_api_mca_select)
        self.w_api_mca.button_reset.clicked.connect(self.reset_mca_plots)
        self.w_api_mca.button_send_to_spect.clicked.connect(self.send_to_spect_mca)

        # API Binary
        self.w_api_bin = WindowAPIbin()
        self.button_api_binary.clicked.connect(self.open_api_bin)
        self.w_api_bin.button_load.clicked.connect(self.activate_api_bin)
        self.w_api_bin.button_select.clicked.connect(self.select_ch_api_bin)
        self.w_api_bin.button_norm_baseline.clicked.connect(
            self.normalize_baseline_api_bin
        )
        self.w_api_bin.button_apply.clicked.connect(self.apply_rand_traces_api_bin)
        self.w_api_bin.button_FFT.clicked.connect(self.apply_fft)
        self.w_api_bin.button_calc_erg.clicked.connect(self.cal_own_erg_api_bin)
        self.w_api_bin.button_send_spec_1.clicked.connect(self.send_to_spect_api_bin1)
        self.w_api_bin.button_send_spec_2.clicked.connect(self.send_to_spect_api_bin2)
        self.w_api_bin.button_yscale.clicked.connect(self.update_yscale_api_bin)
        self.yscale_api_bin = "log"
        # Add individual toolbars
        self.toolbar_apibin1 = NavigationToolbar(self.w_api_bin.plot_energy.canvas, self)
        self.w_api_bin.layout_erg.addWidget(self.toolbar_apibin1)
        self.toolbar_apibin2 = NavigationToolbar(
            self.w_api_bin.plot_energy_traces.canvas, self
        )
        self.w_api_bin.layout_erg_traces.addWidget(self.toolbar_apibin2)
        self.toolbar_apibin3 = NavigationToolbar(
            self.w_api_bin.plot_random_traces.canvas, self
        )
        self.w_api_bin.layout_rand_traces.addWidget(self.toolbar_apibin3)
        self.toolbar_apibin4 = NavigationToolbar(
            self.w_api_bin.plot_own_energy.canvas, self
        )
        self.w_api_bin.layout_own_erg.addWidget(self.toolbar_apibin4)

        ## Find peaks
        self.button_find_peaks.setStyleSheet("background-color : mediumaquamarine")
        self.button_find_peaks.clicked.connect(self.activate_peak_finder)
        self.w_peak_find = WindowPeakFinder()
        self.w_peak_find.button_kernel_apply.clicked.connect(self.peakFind_kernel_apply)
        self.w_peak_find.radioButton_hpge.clicked.connect(self.peakFind_check_hpge)
        self.w_peak_find.radioButton_labr.clicked.connect(self.peakFind_check_labr)
        self.w_peak_find.radioButton_nai.clicked.connect(self.peakFind_check_nai)
        self.w_peak_find.radioButton_plastic.clicked.connect(self.peakFind_check_plastic)
        self.w_peak_find_info = WindowPeakFinderInfo()
        self.w_peak_find.button_info.clicked.connect(self.peakFind_info_activate)

        ## Isotope ID
        self.df_isotID_selected = pd.DataFrame()
        self.df_isotID = pd.DataFrame()
        self.isotID_vlines = []
        self.button_identify_peaks.setStyleSheet("background-color : navajowhite")
        self.button_identify_peaks.clicked.connect(self.activate_isotope_id)
        self.w_isot_id = WindowIsotID()
        self.w_isot_id.isotID_button_apply.setStyleSheet("background-color : lightblue")
        self.w_isot_id.isotID_button_clear.setStyleSheet("background-color : lightcoral")
        self.w_isot_id.isotID_button_apply.clicked.connect(self.isotID_apply)
        self.w_isot_id.isotID_button_clear.clicked.connect(self.isotID_clear)
        self.w_isot_id.button_remove_vlines.setStyleSheet("background-color : lightcoral")
        self.w_isot_id.button_plot_vlines.clicked.connect(self.isotID_plot_vlines)
        self.w_isot_id.button_remove_vlines.clicked.connect(self.isotID_remove_vlines)
        self.w_isot_id.edit_element_search.returnPressed.connect(self.isotID_textSearch)
        self.w_isotID_info = WindowIsotIDInfo()
        self.w_isot_id.button_info.clicked.connect(self.isotID_info_activate)

        ## PNG
        self.w_info_png = WindowPNGInfo()
        self.button_info_png.clicked.connect(self.png_info_activate)
        self.png_spect_scale = "linear"
        self.button_yscale_png.clicked.connect(self.png_spect_yscale)
        self.button_loadFile_png.clicked.connect(self.load_file_png)
        self.button_reset_png.clicked.connect(self.reset_png)
        self.pushButton_apply_tbins_png.clicked.connect(self.change_tbins_png)
        self.pushButton_apply_ebins_png.clicked.connect(self.change_ebins_png)
        self.pushButton_apply_t_png.clicked.connect(self.apply_trange_png)
        self.pushButton_apply_e_png.clicked.connect(self.apply_erange_png)
        self.pushButton_apply_dieaway_png.clicked.connect(self.apply_die_away)
        self.pushButton_apply_fit_png.clicked.connect(self.apply_die_away_fit)
        self.button_send_to_spec_png.clicked.connect(self.send_to_spect_png)
        self.initialize_plots_png()


def main():
    commands = docopt.docopt(__doc__)

    plt.rc("font", size=14)
    plt.style.use("seaborn-v0_8-darkgrid")

    app = QApplication([])
    icon_file = str(files("wara").joinpath("ui/wara-logo.png"))
    splash = QSplashScreen(QtGui.QPixmap(icon_file))
    splash.show()
    font = QtGui.QFont("Helvetica", 24, QtGui.QFont.Bold)
    splash.setFont(font)
    splash.showMessage("Initializing...", Qt.AlignBottom | Qt.AlignHCenter,
                       QtGui.QColor("#39FF14"))
    app.processEvents()
    time.sleep(3)

    window = WaraApp(commands)
    window.show()
    splash.finish(window)
    app.exec_()


if __name__ == "__main__":
    main()
