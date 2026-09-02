"""wara.gui.theme: global tooltip word-wrapping (QWidget.setToolTip patch)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QPushButton

from wara.gui import theme as T  # noqa: F401 -- import installs the patch


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_short_tooltip_is_left_alone(qapp):
    b = QPushButton()
    b.setToolTip("short tip")
    assert b.toolTip() == "short tip"


def test_long_tooltip_is_wrapped_as_rich_text(qapp):
    b = QPushButton()
    long_text = "x" * 100
    b.setToolTip(long_text)
    assert b.toolTip() != long_text
    assert "<div" in b.toolTip()
    assert long_text in b.toolTip()


def test_already_rich_text_tooltip_is_not_double_wrapped(qapp):
    b = QPushButton()
    rich = "<b>already rich</b> " + "x" * 100
    b.setToolTip(rich)
    assert b.toolTip() == rich


def test_stylesheet_suppresses_the_platform_focus_rectangle():
    """Arrowing down the nav rail used to leave the platform's own focus box
    drawn inside the button's border, on top of the themed look, until focus
    moved on. `outline: none` on the base rule turns it off everywhere -- a
    QSS type selector matches subclasses, so the QWidget rule reaches every
    widget in the app. (Not checkable by rendering: the offscreen platform
    doesn't paint focus rects at all, so a pixel test would pass either way.)"""
    base = T.STYLESHEET.split("QWidget#nav_panel")[0]
    assert "QMainWindow, QWidget {" in base
    assert "outline: none;" in base
