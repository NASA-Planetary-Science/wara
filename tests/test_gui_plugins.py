"""Plug-in host (wara.plugins + the PLUG-INS nav in wara.gui.app).

No real plug-in is involved: discovery is fed fake entry points, and the GUI
tests pass PluginEntry records straight to WaraApp. What is pinned here is the
promise to users who install a plug-in: a broken one never stops wara from
starting, and a working one is built only when its tab is first opened.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

from wara import plugins as plug
from wara.spectrum import Spectrum


class _FakeEntryPoint:
    def __init__(self, name, load):
        self.name = name
        self.dist = None
        self._load = load

    def load(self):
        return self._load()


def _raise(exc):
    raise exc


def _plugin(name="Sim", build=None, api=plug.PLUGIN_API):
    return plug.Plugin(name=name, build=build or (lambda host: (None, QWidget())),
                       color="#123456", api=api)


def test_discover_reports_broken_plugins_without_raising(monkeypatch):
    monkeypatch.delenv(plug.DISABLE_ENV, raising=False)
    eps = [
        _FakeEntryPoint("good", lambda: _plugin("Sim")),
        _FakeEntryPoint("missing", lambda: _raise(ImportError("no module named vtk"))),
        _FakeEntryPoint("future", lambda: _plugin("Future", api=plug.PLUGIN_API + 1)),
        _FakeEntryPoint("junk", lambda: object()),
        _FakeEntryPoint("clash", lambda: _plugin("Spectrum")),
    ]
    monkeypatch.setattr(plug, "entry_points", lambda group: eps)

    found = {e.name: e for e in plug.discover(reserved=["Spectrum"])}

    assert found["Sim"].ok
    assert "ImportError: no module named vtk" in found["missing"].error
    assert "Upgrade wara" in found["Future"].error
    assert "not a wara Plugin" in found["junk"].error
    assert "already used" in found["Spectrum"].error
    assert sum(e.ok for e in found.values()) == 1


def test_discover_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv(plug.DISABLE_ENV, "1")
    monkeypatch.setattr(plug, "entry_points",
                        lambda group: pytest.fail("entry points were read"))
    assert plug.discover() == []


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Page(QWidget):
    def __init__(self):
        super().__init__()
        self.activations = 0

    def on_activated(self):
        self.activations += 1


def test_plugin_tabs_build_lazily_and_failures_stay_contained(qapp):
    from wara.gui.app import WaraApp
    from wara.gui.theme import NAV_SECTIONS

    calls = []

    def build(host):
        calls.append(host)
        return QLabel("options"), _Page()

    entries = [
        plug.PluginEntry("Sim", plugin=_plugin("Sim", build), source="simpkg 0.1"),
        plug.PluginEntry("Crashy", plugin=_plugin(
            "Crashy", lambda host: _raise(RuntimeError("boom")))),
        plug.PluginEntry("Broken", error="ImportError: no module named vtk"),
    ]
    w = WaraApp(plugins=entries)
    try:
        first = len(NAV_SECTIONS)
        # Built-in tabs keep their indices; plug-ins follow them.
        assert w._sections[:first] == [n for n, _ in NAV_SECTIONS]
        broken = w.nav_group.button(first + 2)
        assert not broken.isEnabled()
        assert "no module named vtk" in broken.toolTip()
        assert "simpkg 0.1" in w.nav_group.button(first).toolTip()
        assert calls == []                      # nothing built at startup

        w._on_nav(first)
        page = w.stack.currentWidget()
        assert isinstance(page, _Page) and page.activations == 1
        assert w.opt_stack.currentWidget().text() == "options"
        assert not w.opt_panel.isHidden()
        w._on_nav(0); w._on_nav(first)
        assert len(calls) == 1 and page.activations == 2

        # The host is the plug-in's way to hand a spectrum to the Spectrum tab.
        calls[0].send_spectrum(Spectrum(counts=np.arange(1.0, 65.0)), "F8 sim")
        assert w._active_name == "F8 sim" and len(w.spect.counts) == 64

        w._on_nav(first + 1)                    # build raises: error page, no crash
        assert "boom" in w.stack.currentWidget().findChild(QLabel).text()
        assert w.opt_panel.isHidden()
    finally:
        w.close()
