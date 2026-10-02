"""
Plug-ins: optional tabs that separately installed packages add to the GUI.

A plug-in is a package that declares an entry point in the ``wara.plugins``
group, naming a :class:`Plugin`::

    # the plug-in's pyproject.toml
    [project.entry-points."wara.plugins"]
    mcstudio = "mcstudio.wara_plugin:plugin"

wara never imports a plug-in by name. Installing the package is what makes its
tab appear, and uninstalling it makes the tab go away.

**The module the entry point names must be light.** It is imported while the
window is being built, whether or not the user ever opens the tab. Put the
:class:`Plugin` record there, and import the heavy widgets inside ``build``,
which runs the first time the tab is opened.

``build(host)`` returns ``(options, page)``: the widget for the fixed-width
options column (or None for a tab without one) and the widget for the plot
column. If the page has an ``on_activated()`` method, it is called every time
the tab is shown. A plug-in reaches wara only through :class:`Host`, so wara's
internals can change without breaking plug-ins. For colors and the stylesheet,
use :mod:`wara.gui.theme`.

This module does not import Qt, so a plug-in's metadata module can import
:class:`Plugin` without pulling in the GUI.
"""
import os
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable

#: The entry-point group plug-ins register under.
GROUP = "wara.plugins"

#: The version of the contract this wara offers: :class:`Plugin` and
#: :class:`Host`. Raise it when either changes incompatibly.
PLUGIN_API = 1

#: Set to a non-empty value other than ``0`` to start without plug-ins.
#: ``wara --no-plugins`` sets it, and the test suite sets it so a developer's
#: installed plug-ins do not change what the GUI tests see.
DISABLE_ENV = "WARA_DISABLE_PLUGINS"


@dataclass
class Plugin:
    """What a plug-in's entry point names.

    *name* is the nav label. *build* is called as ``build(host)`` the first
    time the tab is opened and returns ``(options_widget_or_None, page)``.
    *api* is the :data:`PLUGIN_API` the plug-in was written against.
    """

    name: str
    build: Callable
    color: str = "#9e9e9e"
    tooltip: str = ""
    api: int = PLUGIN_API


@dataclass
class PluginEntry:
    """One discovered plug-in: the record, or the reason there is none.

    A plug-in that fails to import, or asks for a newer contract, still gets an
    entry. The GUI shows it greyed out with *error* as its tooltip rather than
    hiding it, so a broken install is visible instead of mysterious.
    """

    name: str
    plugin: Plugin | None = None
    error: str | None = None
    source: str = ""          # "distribution version", for the tooltip

    @property
    def ok(self):
        return self.plugin is not None and self.error is None


def plugins_disabled():
    """True when :data:`DISABLE_ENV` asks wara to start without plug-ins."""
    return os.environ.get(DISABLE_ENV, "").strip() not in ("", "0")


def _source(ep):
    dist = getattr(ep, "dist", None)
    if dist is None:
        return ""
    return f"{dist.metadata['Name']} {dist.version}"


def check(plugin, name="plug-in"):
    """The reason *plugin* cannot be used, or None if it can."""
    if not isinstance(plugin, Plugin):
        return f"{name}: entry point is a {type(plugin).__name__}, not a wara Plugin"
    if not callable(plugin.build):
        return f"{plugin.name}: build is not callable"
    if plugin.api > PLUGIN_API:
        return (f"{plugin.name} needs plug-in API {plugin.api}; this wara "
                f"provides {PLUGIN_API}. Upgrade wara.")
    if plugin.api < 1:
        return f"{plugin.name}: invalid plug-in API {plugin.api}"
    return None


def discover(reserved=()):
    """Every installed plug-in, as a list of :class:`PluginEntry`, sorted by name.

    Never raises. Each entry point is loaded inside its own ``try``, so one
    broken plug-in cannot stop wara, or the other plug-ins, from starting.
    *reserved* holds names that are already taken (the built-in tabs); a
    plug-in that reuses one of them, or the name of an earlier plug-in, is
    reported as an error rather than shown twice.
    """
    if plugins_disabled():
        return []
    try:
        eps = entry_points(group=GROUP)
    except Exception as exc:  # noqa: BLE001 -- broken metadata in the env
        return [PluginEntry(name="plug-ins", error=f"could not list plug-ins: {exc}")]

    taken = set(reserved)
    found = []
    for ep in sorted(eps, key=lambda e: e.name):
        source = _source(ep)
        try:
            plugin = ep.load()
        except Exception as exc:  # noqa: BLE001 -- any import failure
            found.append(PluginEntry(name=ep.name, source=source,
                                     error=f"{type(exc).__name__}: {exc}"))
            continue
        error = check(plugin, ep.name)
        name = plugin.name if isinstance(plugin, Plugin) else ep.name
        if error is None and name in taken:
            error = f"{name}: the name is already used by another tab"
        taken.add(name)
        found.append(PluginEntry(name=name, source=source,
                                 plugin=None if error else plugin, error=error))
    return found


class Host:
    """The plug-in's view of the wara window.

    Kept deliberately small. Anything a plug-in needs beyond this belongs here,
    added as a method, rather than being reached for on the window.
    """

    def __init__(self, window):
        self._window = window

    @property
    def window(self):
        """The main window, to parent dialogs on."""
        return self._window

    @property
    def options_width(self):
        """The fixed width (px) of the options column."""
        return self._window.OPT_W

    def send_spectrum(self, spect, name, switch_tab=False):
        """Make *spect* (a :class:`wara.Spectrum`) the Spectrum tab's active
        spectrum, labelled *name*."""
        self._window.load_external_spectrum(spect, name, switch_tab=switch_tab)

    def send_spectra(self, specs, switch_tab=False):
        """Send ``[(spect, name), ...]``: the first becomes active and the rest
        are overlaid."""
        self._window.load_external_spectra(specs, switch_tab=switch_tab)

    def status(self, message):
        """Show *message* in the status bar."""
        self._window.statusBar().showMessage(f"  {message}")
