"""
Flat-field (channel 9) alpha spectrum + X-Y map.

A flat-field run records the alpha-detector events with no gamma coincidence, so
it carries no gamma ``energy`` / ``dt`` columns -- only the A/B/C/D anode signals
(for the X-Y alpha map) and, when available, an ``alpha`` energy column.

This mirrors what the API GUI tab does for a ch-9 run that has an ``alpha``
column: it splits the figure side by side into the alpha energy spectrum (left,
binned with the shared "Energy bins" box) and the X-Y alpha map (right).
Reconstruct the position with ``calc_own_pos`` exactly as the GUI does.

In the GUI the two panels cross-filter interactively -- an alpha-energy window
updates the X-Y map and an X-Y region updates the alpha spectrum. Here we show
the equivalent static cut: the alpha spectrum of a central X-Y region overlaid on
the full spectrum.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wara import apicalc

RNG = np.random.default_rng(7)
N = 20000
EBINS = 1024   # shared "Energy bins" count in the GUI

# ---- Synthetic flat-field event DataFrame -------------------------------
# Only the A/B/C/D anode signals plus an alpha energy column: no gamma
# energy/dt columns, matching an on-disk flat-field (ch 9) run.
df_raw = pd.DataFrame({
    "alpha": RNG.normal(loc=2200, scale=180, size=N),   # alpha peak ~2200 ch
    "A":     RNG.uniform(0.1, 1.0, N),
    "B":     RNG.uniform(0.1, 1.0, N),
    "C":     RNG.uniform(0.1, 1.0, N),
    "D":     RNG.uniform(0.1, 1.0, N),
})

# ---- Position reconstruction (X2/Y2 from the quadrants) -----------------
df = apicalc.calc_own_pos(df_raw)
has_alpha = "alpha" in df.columns
print(f"Events            : {len(df):,}")
print(f"Has alpha column  : {has_alpha}")
print(f"X2 range          : [{df['X2'].min():.3f}, {df['X2'].max():.3f}]")

# ---- Split view: alpha spectrum (left) + X-Y map (right) ----------------
if has_alpha:
    fig, (ax_alpha, ax_xy) = plt.subplots(1, 2, figsize=(12, 5))

    erange = [0.0, float(df["alpha"].max())]
    counts, edges = np.histogram(df["alpha"], bins=EBINS, range=erange)
    centers = (edges[1:] + edges[:-1]) / 2
    ax_alpha.plot(centers, counts, color="green", lw=0.9, label="all events")

    # Cross-filter demo: the alpha spectrum of a central X-Y region only. In the
    # GUI this is what dragging that rectangle on the X-Y map produces.
    roi = apicalc.dfxy(df, xrange=(-0.3, 0.3), yrange=(-0.3, 0.3),
                       xkey="X2", ykey="Y2")
    roi_counts, _ = np.histogram(roi["alpha"], bins=EBINS, range=erange)
    ax_alpha.plot(centers, roi_counts, color="orange", lw=0.9,
                  label="central X-Y region")
    ax_alpha.set_xlabel("Alpha energy (channels)")
    ax_alpha.set_ylabel("Counts")
    ax_alpha.set_title(f"Alpha spectrum ({EBINS} bins)")
    ax_alpha.legend()
else:
    # No alpha column: X-Y map only.
    fig, ax_xy = plt.subplots(1, 1, figsize=(6, 5))

ax_xy.hexbin(df["X2"], df["Y2"], gridsize=80, cmap="plasma")
ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y")
ax_xy.set_title("X-Y alpha map")

fig.tight_layout()
plt.show()
