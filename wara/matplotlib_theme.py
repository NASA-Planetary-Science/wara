"""
matplotlib_theme.py — Centralised Matplotlib style for wara.

Three built-in themes
---------------------
SEABORN      : seaborn-v0_8-darkgrid (original wara default)
DARK         : deep-navy dark theme (from gamma_viewer scratch)
PUBLICATION  : clean, print-ready style for journal figures

Usage
-----
    from wara.matplotlib_theme import set_theme, apply_theme, DARK
    set_theme(DARK)          # change the active theme
    apply_theme()            # call inside any plot method

The active theme is stored as a module-level string; the default is SEABORN.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Theme name constants ───────────────────────────────────────────────────────
SEABORN     = "seaborn"
DARK        = "dark"
PUBLICATION = "publication"

# ── Module-level active theme ──────────────────────────────────────────────────
_active_theme: str = SEABORN

# ── Dark theme palette (matches gamma_viewer_tmp.py) ──────────────────────────
_BG_DARK   = "#0a0a0f"
_BG_PANEL  = "#0f0f1a"
_BORDER    = "#1e1e3a"
_GRID      = "#1a1a2e"
_TEXT_DIM  = "#5c6bc0"
_TEXT_PRI  = "#e8eaf6"

_DARK_PARAMS: dict = {
    "figure.facecolor":  _BG_DARK,
    "axes.facecolor":    "#06060e",
    "axes.edgecolor":    _BORDER,
    "axes.labelcolor":   _TEXT_DIM,
    "axes.grid":         True,
    "grid.color":        _GRID,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.7,
    "xtick.color":       _TEXT_DIM,
    "ytick.color":       _TEXT_DIM,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "text.color":        _TEXT_PRI,
    "figure.edgecolor":  _BG_DARK,
    "savefig.facecolor": _BG_DARK,
    "legend.facecolor":  _BG_PANEL,
    "legend.edgecolor":  _BORDER,
    "legend.labelcolor": _TEXT_PRI,
    "legend.fontsize":   9,
    "lines.linewidth":   1.2,
    "axes.prop_cycle":   mpl.cycler(color=[
        "#00e5ff", "#39ff14", "#ffb300", "#ff3b5c",
        "#b388ff", "#69f0ae", "#ff6d00", "#40c4ff",
    ]),
}

# ── Publication theme ──────────────────────────────────────────────────────────
# Optimised for single- and double-column journal figures (NIM-A, PRC, etc.).
# Wong (2011) colour-blind-safe palette, DejaVu Sans for broad compatibility.
_WONG_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky-blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

_PUBLICATION_PARAMS: dict = {
    "figure.facecolor":        "white",
    "axes.facecolor":          "white",
    "axes.edgecolor":          "black",
    "axes.linewidth":          0.8,
    "axes.labelcolor":         "black",
    "axes.labelsize":          11,
    "axes.titlesize":          11,
    "axes.grid":               True,
    "grid.color":              "#cccccc",
    "grid.linewidth":          0.5,
    "grid.linestyle":          "--",
    "grid.alpha":              0.6,
    "xtick.color":             "black",
    "ytick.color":             "black",
    "xtick.labelsize":         9,
    "ytick.labelsize":         9,
    "xtick.direction":         "in",
    "ytick.direction":         "in",
    "xtick.top":               True,
    "ytick.right":             True,
    "xtick.minor.visible":     True,
    "ytick.minor.visible":     True,
    "xtick.major.width":       0.8,
    "ytick.major.width":       0.8,
    "xtick.minor.width":       0.5,
    "ytick.minor.width":       0.5,
    "xtick.major.size":        4.0,
    "ytick.major.size":        4.0,
    "xtick.minor.size":        2.0,
    "ytick.minor.size":        2.0,
    "text.color":              "black",
    "font.family":             "sans-serif",
    "font.sans-serif":         ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size":               10,
    "figure.edgecolor":        "white",
    "savefig.facecolor":       "white",
    "savefig.dpi":             300,
    "savefig.bbox":            "tight",
    "legend.facecolor":        "white",
    "legend.edgecolor":        "#aaaaaa",
    "legend.fontsize":         9,
    "legend.framealpha":       0.9,
    "lines.linewidth":         1.5,
    "lines.markersize":        5,
    "figure.figsize":          [6.4, 4.0],
    "axes.prop_cycle":         mpl.cycler(color=_WONG_COLORS),
}


def set_theme(theme: str) -> None:
    """Set the active wara theme (SEABORN, DARK, or PUBLICATION)."""
    global _active_theme
    if theme not in (SEABORN, DARK, PUBLICATION):
        raise ValueError(f"Unknown theme '{theme}'. Choose from: SEABORN, DARK, PUBLICATION")
    _active_theme = theme


def get_theme() -> str:
    """Return the name of the currently active theme."""
    return _active_theme


def apply_theme(theme: str | None = None) -> None:
    """
    Apply a theme to the current Matplotlib session.

    Parameters
    ----------
    theme : str, optional
        One of SEABORN, DARK, PUBLICATION.  If omitted the module-level
        active theme (set via :func:`set_theme`) is used.
    """
    target = theme if theme is not None else _active_theme

    if target == SEABORN:
        plt.style.use("seaborn-v0_8-darkgrid")

    elif target == DARK:
        plt.style.use("default")
        mpl.rcParams.update(_DARK_PARAMS)

    elif target == PUBLICATION:
        plt.style.use("default")
        mpl.rcParams.update(_PUBLICATION_PARAMS)

    else:
        raise ValueError(f"Unknown theme '{target}'. Choose from: SEABORN, DARK, PUBLICATION")
