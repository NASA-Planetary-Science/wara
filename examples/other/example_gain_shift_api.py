"""
Example API gamma spectra gain shift

Splits a list-mode API run into N equal time segments and corrects for gain
drift over the run by sliding every segment onto the first (earliest) one:

    1. plot_overlay() shows the raw segments — peaks drift with time.
    2. align()        finds, per segment, the correction that best lines its
                      spectrum up with segment 0 and writes the corrected
                      energies to df["energy_cal"].
    3. plot_aligned() shows the corrected segments — the peaks now overlap.

align() supports two correction models:
    method="shift"  (default) a pure additive channel slide — robust, works
                    even with a single peak.
    method="linear" slope + offset — use when the gain itself drifts (peaks at
                    high channels move more than those at low channels). Needs
                    structure spread across the axis (≥2 peaks/regions).
"""
import matplotlib.pyplot as plt
from wara import apicalc
from wara import read_parquet_api


df = read_parquet_api.read_parquet_file(date="2024-08-21", runnr=18, ch=4)

cal = apicalc.GainShift(df, n_segments=40, bins=2**12, erange=(0, 2**16))

cal.plot_overlay()           # before: segments drift with time
cal.align(method="linear")                  # additive shift onto the left-most (segment 0)
# For a true gain drift, use a slope + offset instead:
#     cal.align(method="linear")
# Tip: restrict the alignment to a strong peak with align(xrange=(lo, hi)).
cal.plot_aligned()           # after: segments overlap

plt.show()
