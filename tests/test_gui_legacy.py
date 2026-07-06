"""
Pytest tests for the wara GUI entry point and UI loading.
"""

import sys
import pytest
import docopt
import matplotlib
matplotlib.use("Agg")


def test_gui_import():
    """wara.gui_legacy.app can be imported without errors."""
    import wara.gui_legacy.app  # noqa: F401


def test_gui_module_docstring():
    """Module docstring exists and is non-empty (required by docopt)."""
    import wara.gui_legacy.app as m
    assert m.__doc__ is not None
    assert len(m.__doc__) > 0


def test_docopt_no_args():
    """docopt parses correctly with no CLI arguments (plain 'wara' invocation)."""
    import wara.gui_legacy.app as m
    argv = []
    result = docopt.docopt(m.__doc__, argv=argv)
    assert result["<file_name>"] is None
    assert result["-o"] is False


def test_docopt_open_flag():
    """docopt parses -o flag correctly."""
    import wara.gui_legacy.app as m
    result = docopt.docopt(m.__doc__, argv=["-o"])
    assert result["-o"] is True
    assert result["<file_name>"] is None


def test_docopt_file_arg(tmp_path):
    """docopt parses a file argument correctly."""
    import wara.gui_legacy.app as m
    fake_file = str(tmp_path / "test.csv")
    result = docopt.docopt(m.__doc__, argv=[fake_file])
    assert result["<file_name>"] == fake_file


def test_ui_file_loads():
    """Qt UI file loads without errors (catches stale module references like old package names)."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication
    from PyQt5.uic import loadUi
    from importlib.resources import files

    app = QApplication.instance() or QApplication(sys.argv)
    ui_file = str(files("wara").joinpath("ui/qt_gui.ui"))
    from PyQt5.QtWidgets import QMainWindow
    win = QMainWindow()
    loadUi(ui_file, win)
