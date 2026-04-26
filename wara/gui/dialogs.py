#import pkg_resources
from importlib.resources import files
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt


class Dialog_from_UI(QDialog):
    """Create a dialog from a UI file with a given window title."""

    def __init__(self):
        super().__init__()
        self.define_ui_vars()
        #ui_file = pkg_resources.resource_filename("wara", f"ui/{self.ui_name}")
        ui_file = str(files("wara").joinpath(f"ui/{self.ui_name}"))
        loadUi(ui_file, self)
        self.setWindowTitle(self.window_title)


class WindowMainInfo(Dialog_from_UI):
    def define_ui_vars(self):
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.ui_name = "win_info_main.ui"
        self.window_title = "Spectrum info"


class WindowCust(Dialog_from_UI):
    def define_ui_vars(self):
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.ui_name = "win_customize.ui"
        self.window_title = "Customize plot"


class WindowCustInfo(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_info_customize.ui"
        self.window_title = "Customize info"


class WindowAddSubtract(Dialog_from_UI):
    def define_ui_vars(self):
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.ui_name = "win_add_subtract.ui"
        self.window_title = "Add/Subtract Spectra"


class WindowAddSubtractInfo(Dialog_from_UI):
    def define_ui_vars(self):
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.ui_name = "win_info_add_subt.ui"
        self.window_title = "Add/Subtract Info"


class WindowPeakFinder(Dialog_from_UI):
    def define_ui_vars(self):
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.ui_name = "win_peak_find.ui"
        self.window_title = "Peak finder"


class WindowPeakFinderInfo(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_info_peak_find.ui"
        self.window_title = "Peak finder info"


class WindowCal(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_erg_cal.ui"
        self.window_title = "Energy calibration"


class WindowCalInfo(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_info_ecal.ui"
        self.window_title = "Energy calibration information"


class WindowCalEqns(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_erg_cal_eqns.ui"
        self.window_title = "Energy calibration: set equations"


class WindowCalAddPoint(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_erg_cal_add_point.ui"
        self.window_title = "Energy calibration: add point"


class WindowEff(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_eff.ui"
        self.window_title = "Efficiency calibration"


class WindowEffInfo(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_info_eff.ui"
        self.window_title = "Efficiency calibration information"


class WindowInfoFile(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_info_file.ui"
        self.window_title = "File information"


class WindowIsotID(Dialog_from_UI):
    def define_ui_vars(self):
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.ui_name = "win_isot_id.ui"
        self.window_title = "Isotope ID"


class WindowIsotIDInfo(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_info_isot_id.ui"
        self.window_title = "Isotope ID info"


class WindowAdvFit(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_adv_fit.ui"
        self.window_title = "Advanced Fitting"


class WindowAPImca(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_api_mca.ui"
        self.window_title = "API MCA"


class WindowAPIbin(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_api_binary.ui"
        self.window_title = "API Binary File"


class WindowAPI3D(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_api_3D.ui"
        self.window_title = "API 3D plot"


class WindowAPIfilters(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_api_filters.ui"
        self.window_title = "Apply API filters"


class WindowPNGInfo(Dialog_from_UI):
    def define_ui_vars(self):
        self.ui_name = "win_info_png.ui"
        self.window_title = "PNG information"
