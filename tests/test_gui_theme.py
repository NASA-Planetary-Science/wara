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
