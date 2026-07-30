"""
Alpha energy as a fourth panel of a normal API run.

Many API (gamma-coincidence) parquet files also carry the ch-9 alpha energy of
the event, under either ``energy_ch9`` or ``alpha`` -- the two names mean the
same thing. The API GUI tab detects such a run and offers an "Add alpha energy"
checkbox: ticking it adds the alpha spectrum (2048 bins by default) as a fourth
panel, bottom-right under the X-Y map, cross-linked with the gamma energy, dt
and X-Y panels.

This script reproduces that four-panel view statically, and shows the
cross-filter both ways:

* an alpha-energy band (the interactive span on the alpha panel) cuts the events,
  so the gamma / dt / X-Y panels are drawn from the cut data;
* the alpha spectrum of that same cut is overlaid on the full one, which is what
  a dt or X-Y cut does to the alpha panel in the GUI.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wara import apicalc

RNG = np.random.default_rng(3)
N = 40000
EBINS = 1024    # gamma "Energy bins"
TBINS = 128     # "dt bins"
ABINS = 2048    # "Alpha bins" (the GUI default)
ALPHA_BAND = (1800, 2600)   # the band dragged on the alpha panel, in channels

# ---- Synthetic API event DataFrame --------------------------------------
# A normal (ch 4/5) run: gamma energy + dt + the A/B/C/D anode signals, plus the
# ch-9 alpha energy of the same event.
df_raw = pd.DataFrame({
    "energy_orig": RNG.uniform(0, 6000, N),
    "dt":          RNG.normal(0.0, 15.0, N),            # ns
    "energy_ch9":  RNG.normal(2200, 250, N),            # alpha peak ~2200 ch
    "A":           RNG.uniform(0.1, 1.0, N),
    "B":           RNG.uniform(0.1, 1.0, N),
    "C":           RNG.uniform(0.1, 1.0, N),
    "D":           RNG.uniform(0.1, 1.0, N),
})

df = apicalc.calc_own_pos(df_raw)       # X2/Y2, as the GUI does on load
# Either column name is accepted, exactly as the GUI's detection does.
akey = next((c for c in ("energy_ch9", "alpha") if c in df.columns), None)
print(f"Events              : {len(df):,}")
print(f"Alpha energy column : {akey}")

# ---- The alpha-energy cut ----------------------------------------------
amin, amax = ALPHA_BAND
cut = df[(df[akey] > amin) & (df[akey] < amax)]
print(f"Alpha band [{amin}, {amax}] : {len(cut):,} events "
      f"({100 * len(cut) / len(df):.1f}%)")

# ---- Four panels, laid out like the GUI --------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
(ax_spe, ax_xy), (ax_dt, ax_aspe) = axes

gam, edges = np.histogram(cut["energy_orig"], bins=EBINS,
                          range=[0.0, float(df["energy_orig"].max())])
ax_spe.plot((edges[1:] + edges[:-1]) / 2, gam, color="green", lw=0.9)
ax_spe.set_xlabel("Channels"); ax_spe.set_ylabel("Counts")
ax_spe.set_title("Gamma energy (alpha band only)")

ax_dt.hist(cut["dt"], bins=TBINS, color="teal")
ax_dt.set_xlabel("dt (ns)"); ax_dt.set_ylabel("Counts")
ax_dt.set_title("dt (alpha band only)")

ax_xy.hexbin(cut["X2"], cut["Y2"], gridsize=80, cmap="plasma")
ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y")
ax_xy.set_title("X-Y map (alpha band only)")

# The alpha panel keeps the full spectrum with the band marked -- the GUI marks
# the active cut with dotted red lines so it can be dragged again.
arange = [0.0, float(df[akey].max())]
alp, aedges = np.histogram(df[akey], bins=ABINS, range=arange)
centers = (aedges[1:] + aedges[:-1]) / 2
ax_aspe.plot(centers, alp, color="orange", lw=0.9, label="all events")
alp_cut, _ = np.histogram(cut[akey], bins=ABINS, range=arange)
ax_aspe.plot(centers, alp_cut, color="cyan", lw=0.9, label="selected band")
for x in (amin, amax):
    ax_aspe.axvline(x, color="red", ls=":", lw=1.3)
ax_aspe.set_xlabel("Alpha energy (channels)"); ax_aspe.set_ylabel("Counts")
ax_aspe.set_title(f"Alpha energy ({ABINS} bins)")
ax_aspe.legend()

fig.tight_layout()
plt.show()
