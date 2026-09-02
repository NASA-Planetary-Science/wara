"""
The "uncut outline": reading an API cut against the run it came from.

While an interactive cut is applied, the API GUI tab draws the *uncut* run
faintly in grey behind the energy, alpha and dt spectra, with the cut data in
colour on top. A selection means little without the whole it was taken from:
the outline is what says how much of the run survived the gate, and it holds
each panel on the scale of the whole so the cut reads as a fraction of it
instead of being re-stretched to fill the axes. The X-Y map is left alone --
a grey density either hides behind the hexbins or washes them out, and the cut
region is already marked there.

The GUI's DISPLAY panel has a "Show uncut outline" checkbox (on by default);
it only takes effect once a cut is applied, since before that the outline would
exactly overlap the data.

This script reproduces that view statically for one X-Y (region) cut.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wara import apicalc

RNG = np.random.default_rng(5)
N = 40000
EBINS = 512     # gamma "Energy bins"
TBINS = 128     # "dt bins"
HEXBINS = 60    # "X-Y bins"
REGION = (-0.25, 0.25, -0.25, 0.25)     # the rectangle dragged on the X-Y map

# How the outline is drawn, matching wara.gui.api.ApiController.GHOST_LINE
# (grey, thin, half transparent).
GHOST = {"color": "grey", "alpha": 0.55, "linewidth": 0.8, "zorder": 1}

# ---- Synthetic API event DataFrame --------------------------------------
df_raw = pd.DataFrame({
    "energy_orig": RNG.uniform(0, 6000, N),
    "dt":          RNG.normal(0.0, 15.0, N),        # ns
    "A":           RNG.uniform(0.1, 1.0, N),
    "B":           RNG.uniform(0.1, 1.0, N),
    "C":           RNG.uniform(0.1, 1.0, N),
    "D":           RNG.uniform(0.1, 1.0, N),
})
df = apicalc.calc_own_pos(df_raw)       # X2/Y2, as the GUI does on load

# ---- The cut: one region of the alpha image -----------------------------
xlo, xhi, ylo, yhi = REGION
cut = apicalc.dfxy(df, xrange=(xlo, xhi), yrange=(ylo, yhi),
                   xkey="X2", ykey="Y2")     # the GUI's reconstructed position
print(f"Events        : {len(df):,}")
print(f"X-Y region cut: {len(cut):,} events "
      f"({100 * len(cut) / len(df):.1f}% of the run)")

# ---- Three panels, each with its uncut outline --------------------------
fig, (ax_spe, ax_dt, ax_xy) = plt.subplots(1, 3, figsize=(16, 4.5))

# Energy: the whole run as a grey line, the cut on top in green. Both are
# binned on the same axis, so the gap between them is what the cut removed.
erange = [0.0, float(df["energy_orig"].max())]
whole, edges = np.histogram(df["energy_orig"], bins=EBINS, range=erange)
kept, _ = np.histogram(cut["energy_orig"], bins=EBINS, range=erange)
centres = (edges[1:] + edges[:-1]) / 2
ax_spe.plot(centres, whole, label="uncut", **GHOST)
ax_spe.plot(centres, kept, color="green", lw=0.9, label="cut")
ax_spe.set_xlabel("Channels"); ax_spe.set_ylabel("Counts")
ax_spe.set_title("Gamma energy"); ax_spe.legend()

# dt: binned over the *uncut* window, so the outline is whole and the axis
# does not jump every time the gate is narrowed.
low, high = np.percentile(df["dt"], [0.2, 99.5])
gcounts, gedges = np.histogram(df["dt"], bins=TBINS, range=(low, high))
ax_dt.step(gedges, np.append(gcounts, gcounts[-1]), where="post",
           label="uncut", **GHOST)
ax_dt.hist(cut["dt"], bins=TBINS, range=(low, high), color="teal", alpha=0.8,
           label="cut")
ax_dt.set_xlabel("dt (ns)"); ax_dt.set_ylabel("Counts")
ax_dt.set_title("Alpha-gamma time"); ax_dt.legend()

# X-Y: no grey here. A density map under a hexbin either hides behind it or
# washes it out, so the panel just shows the surviving events, with the region
# outlined in red as the GUI marks an active cut.
extent = (-0.9, 0.9, -0.9, 0.9)
ax_xy.hexbin(cut["X2"], cut["Y2"], gridsize=HEXBINS, cmap="plasma", mincnt=1,
             extent=extent)
ax_xy.add_patch(plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False,
                              edgecolor="red", ls="--", lw=1.3))
ax_xy.set_xlim(extent[0], extent[1]); ax_xy.set_ylim(extent[2], extent[3])
ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y")
ax_xy.set_title("Alpha X-Y map")

fig.tight_layout()
plt.show()
