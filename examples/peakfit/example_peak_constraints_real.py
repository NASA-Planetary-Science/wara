"""
Constraining a fit on *real* data: peaks / fix_center / fix_fwhm / ratios.

The companion script ``example_peak_constraints.py`` uses synthetic data with a
known answer. This one applies the same constraint kwargs of
``MultiProfilePeakFit`` to a real HPGe spectrum: the overlapping gamma-ray
doublet near 932.6 and 937.0 keV in an ammonia (NH3) measurement.

The two lines sit only ~4 keV apart — barely two detector resolutions — so a
free fit can trade their widths and areas against each other (here it returns
2.30 and 2.74 keV for two lines that physically share the same ~2.50 keV
resolution). Constraints inject what we know and tighten the result:

  * ``peaks=``      declare the two lines explicitly (reproducible, and works
                   even when one member is too weak for the peak finder);
  * ``fix_fwhm=``   pin both widths to the measured detector resolution — the
                   standard HPGe workflow (Evans et al. 2006 fix peak widths to
                   a resolution curve built from strong, clean lines);
  * ``fix_center=`` lock the centroids found in a first pass;
  * ``ratios=``     tie the areas if the lines have a known intensity ratio.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from wara import file_reader
from wara import peaksearch as ps
from wara.advanced_fit import MultiProfilePeakFit


file = Path(__file__).parent.parent / "data/test_data_hpge_NH3.txt"
spect = file_reader.read_txt(file)
search = ps.PeakSearch(spect, ref_x=935, ref_fwhm=2.5, fwhm_at_0=0.5,
                       min_snr=6, method="km")

XRANGE = [927, 943]
PEAKS = [932.5, 937.0]


def rows(fit):
    """Format both peaks of a fit summary, sorted by energy."""
    s = fit.summary().sort_values("mean")
    return "   ".join(
        f"{r['mean']:7.2f} keV  FWHM {r['fwhm']:.2f}  "
        f"area {r['area']:6.0f}+/-{r['area_err']:.0f}"
        for _, r in s.iterrows()
    )


# --- Measure the detector resolution from strong, clean, isolated lines -----
# FWHM grows smoothly with energy; interpolate the curve to 935 keV.
calib = {}
for e_line, win in {844.9: [838, 852], 1015.4: [1007, 1022],
                    1173.2: [1166, 1181]}.items():
    f = MultiProfilePeakFit(search, win, bkg="poly1", profile="gauss")
    r = f.summary().iloc[(f.summary()["mean"] - e_line).abs().idxmin()]
    calib[r["mean"]] = r["fwhm"]
res_935 = float(np.interp(935.0, list(calib), list(calib.values())))
print("Detector resolution from clean lines: "
      + ", ".join(f"{e:.0f}->{w:.2f} keV" for e, w in calib.items()))
print(f"  => interpolated FWHM at 935 keV = {res_935:.2f} keV\n")

# --- 1. Declare the two lines (number of peaks) ----------------------------
free = MultiProfilePeakFit(search, XRANGE, bkg="poly1", profile="gauss",
                           peaks=PEAKS)
print("1. peaks=[932.5, 937.0]  (free widths)")
print("   " + rows(free))
print("   note the mismatched widths from the overlap; areas carry ~4% error.")

# --- 2. Pin both widths to the measured resolution -------------------------
fixed_w = MultiProfilePeakFit(search, XRANGE, bkg="poly1", profile="gauss",
                              peaks=PEAKS,
                              fix_fwhm={932.5: res_935, 937.0: res_935})
print(f"\n2. fix_fwhm to the resolution ({res_935:.2f} keV)")
print("   " + rows(fixed_w))
print("   widths now consistent and the area uncertainties shrink.")

# --- 3. Lock the centroids (two-pass: determine, then fix) -----------------
c_lo, c_hi = sorted(free.summary()["mean"])
fixed_c = MultiProfilePeakFit(search, XRANGE, bkg="poly1", profile="gauss",
                              peaks=PEAKS,
                              fix_center={c_lo: round(c_lo, 2),
                                          c_hi: round(c_hi, 2)})
print(f"\n3. fix_center to the first-pass positions "
      f"({c_lo:.2f}, {c_hi:.2f} keV)")
print("   " + rows(fixed_c))

# --- 4. Tie the areas by a known intensity ratio ---------------------------
# Suppose nuclear data gives I(937)/I(932) = 0.70 for these lines.
tied = MultiProfilePeakFit(search, XRANGE, bkg="poly1", profile="gauss",
                           peaks=PEAKS, ratios=[(937.0, 932.5, 0.70)])
print("\n4. ratios=[(937.0, 932.5, 0.70)]  (tie the weak line to the strong)")
print("   " + rows(tied))

# --- Plot: free vs resolution-constrained ----------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
m = (spect.energies >= XRANGE[0]) & (spect.energies <= XRANGE[1])
ax.plot(spect.energies[m], spect.counts[m], ".", ms=6, color="0.4",
        label="data")
xf = np.linspace(*XRANGE, 600)
ax.plot(xf, free.fit_result.eval(x=xf), "-", color="tab:orange", lw=1.6,
        label="free widths")
ax.plot(xf, fixed_w.fit_result.eval(x=xf), "-", color="tab:green", lw=1.6,
        label=f"FWHM fixed to {res_935:.2f} keV")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel("Counts")
ax.set_title("Real HPGe doublet (932.6 / 937.0 keV) with width constraint")
ax.legend()
fig.tight_layout()
plt.show()
