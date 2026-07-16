"""Shared constants and small helpers for the API tab modules.

Split out of ``wara.gui.api`` so the dialog modules (``api_dialogs``,
``api_shifts``, ``api_combine``, ``api_diagnostics``, ``api_selections``)
can use them without importing the controller module.
"""
from matplotlib.colors import LinearSegmentedColormap

from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QSizePolicy

from . import theme as T

API_PLOT_BG = T.BG_PLOT
COLORMAP = "plasma"
# Default histogram bin counts (energy spectrum, dt histogram, X-Y hexbin grid).
# Shown pre-filled in the Display "bins" boxes; high-resolution channels bump the
# energy bins at load (see ApiController._load_file).
DEFAULT_EBINS = 2 ** 12
DEFAULT_TBINS = 512
DEFAULT_HEXBINS = 80
# Faint gray density map for the *base* X-Y hexbin when colored energy
# selections are overlaid on top: low counts fade into the plot background, high
# counts brighten to a dim gray so the selections stay the focus.
GRAY_CMAP = LinearSegmentedColormap.from_list("api_gray", [API_PLOT_BG, T.TEXT_DIM])

# Send-to-Spectrum button label. The click is confirmed with a brief green blink
# (see ApiController._flash_button), not a persistent label change.
SEND_DEFAULT_TEXT = "Send to spectrum"


def _combo_row(label_text, widget):
    """A 'label  [widget]' row where the widget expands to fill the width  --
    unlike labeled_row's fixed-width inputs, so a long combo entry isn't cut
    off."""
    row = QWidget()
    rl = QHBoxLayout(row)
    rl.setContentsMargins(0, 0, 0, 0)
    rl.setSpacing(6)
    rl.addWidget(QLabel(label_text))
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    rl.addWidget(widget, 1)
    return row


def _clear_axis(ax):
    """Clear *ax* without warning. Clearing an axis that is currently log-scaled
    makes matplotlib apply a non-positive default ylim and warn; resetting to a
    linear scale first avoids that."""
    ax.set_yscale("linear")
    ax.clear()


def _draw_axes_placeholder(ax, message, xlabel, style_axis, yscale="log"):
    """Clear *ax*, apply the owning dialog's *style_axis*, and center a dim
    *message*. Shared by the API dialogs that show a hint until the user runs
    something, so the styling stays identical across them."""
    _clear_axis(ax)
    ax.set_ylim(0.1, 1.0)
    style_axis(ax, xlabel, yscale)
    if message:
        # wrap=True keeps a long message inside the axes instead of running
        # off both edges of the plot.
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
                va="center", color=T.TEXT_DIM, fontsize=12, wrap=True)
