"""Locate the energy-filter sum windows on traces and cross-check the fast filter.

The Pixie stores three raw energy-filter sums per event: trailing, gap and
leading (Fig. 4-1 of the manual), taken over three consecutive windows placed in
time order ``[trailing: L][gap: G][leading: L]``. ``find_LGL_sums`` sweeps that
layout across each trace and finds where the window sums reproduce the stored
values -- with the correct L and G (from the run settings) the absolute sums
match exactly.

We overlay the recovered L / G / L window boundaries on a few traces and compare
their position with ``_fast_filter_ref`` (the reconstructed *fast* trigger). The
two mark different things (energy sums vs fast trigger) but should sit at a
fixed offset -- a consistency check on both.
"""
import numpy as np
import matplotlib.pyplot as plt

from wara import helper_api

DATE, RUNNR, CH = "2026-07-21", 19, 4

# Energy (slow) filter window lengths in ADC samples, from the run settings.
L, G = helper_api.read_slow_filter_geometry(DATE, RUNNR, CH)
print(f"slow-filter windows: L={L}, G={G} ADC samples (2L+G={2*L+G})")

df = helper_api.read_trace_data(DATE, RUNNR, cfd="on")
df = df[(df.energy > 8000) & (df.pileup == 0)].head(4).reset_index(drop=True)

# Fast-trigger reference for the same traces (baseline-subtracted matrix).
T = np.stack([np.asarray(t, float) for t in df.trace])
T -= T[:, :30].mean(axis=1, keepdims=True)
fast_ref = helper_api._fast_filter_ref(T, rise=5, gap=2, threshold=0.2)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
print(f"\n{'ev':>2} {'LGL start':>9} {'gap start':>9} {'lead start':>10} "
      f"{'fast_ref':>8} {'fast-gapstart':>13} {'match err':>9}")
for e, ax in enumerate(axes.ravel()):
    tr = np.asarray(df.trace[e], float)
    res = helper_api.find_LGL_sums(
        tr, df.Esum_trailing[e], df.Esum_gap[e], df.Esum_leading[e], L, G,
        relative=True)
    b = res["bounds"]  # (trailing_start, gap_start, leading_start, leading_end)

    ax.plot(tr, color="0.35", lw=1.0)
    # four vertical boundary lines delimiting the L | G | L windows
    for x in b:
        ax.axvline(x, color="tab:red", ls="--", lw=1.2)
    ax.axvspan(b[0], b[1], color="tab:blue", alpha=0.12)   # trailing (L)
    ax.axvspan(b[1], b[2], color="tab:orange", alpha=0.12)  # gap (G)
    ax.axvspan(b[2], b[3], color="tab:green", alpha=0.12)  # leading (L)
    ax.axvline(fast_ref[e], color="k", ls=":", lw=1.5, label="fast_ref")
    ax.set_title(f"event {e}  (E={df.energy[e]})")
    ax.set_xlabel("sample (2 ns)")
    ax.legend(loc="upper right", fontsize=8)

    print(f"{e:>2} {b[0]:>9} {b[1]:>9} {b[2]:>10} {fast_ref[e]:>8.2f} "
          f"{fast_ref[e]-b[1]:>13.2f} {res['error']:>9.3g}")

axes[0, 0].set_ylabel("ADC")
axes[1, 0].set_ylabel("ADC")
fig.suptitle(f"Energy-sum windows (L|G|L) vs fast trigger -- {DATE} RUN {RUNNR}")
fig.tight_layout()
plt.show()
