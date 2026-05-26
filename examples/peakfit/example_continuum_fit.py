"""
Example usage of ContinuumFit (wara.advanced_fit).

ContinuumFit fits a polynomial baseline to a spectrum window while
masking out the peak regions identified by PeakSearch. It's handy for:

  * estimating the smooth background under one or many peaks
  * subtracting that baseline to get a "peaks-only" view
  * picking a continuum to feed into a separate peak-fit routine

The class takes a PeakSearch (which already knows the peak locations and
widths) and excludes every channel within +/- mask_fwhm * FWHM of each
detected peak before fitting the polynomial.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from wara import file_reader
from wara import peaksearch as ps
from wara.advanced_fit import ContinuumFit


# HPGe spectrum — rich in peaks, so the continuum-vs-peak contrast is clear.
file = Path(__file__).parent.parent / "data/test_data_hpge_NH3.txt"
spect = file_reader.read_txt(file)

search = ps.PeakSearch(spect, ref_x=420, ref_fwhm=3, min_snr=15, method="fast")
print(f"PeakSearch found {len(search.peaks_idx)} peaks across the spectrum.")


# --- 1. Basic continuum fit over a peak-rich window -------------------------
# Pick a moderate window where a low-degree polynomial is a reasonable
# baseline (the full HPGe range has too much dynamic range for a single
# polynomial to track well).
xrange = [180, 540]
cont = ContinuumFit(search, xrange=xrange, degree=3, mask_fwhm=3.0)
in_range = (cont.x >= xrange[0]) & (cont.x <= xrange[1])
print(f"\nContinuum fit on {xrange}:")
print(f"  degree            = {cont.degree}")
print(f"  channels in range = {in_range.sum()}")
print(f"  channels kept     = {cont.continuum_mask.sum()}")
print(f"  channels masked   = {in_range.sum() - cont.continuum_mask.sum()}")
print(f"  redchi            = {cont.fit_result.redchi:.3f}")
print(f"  aic               = {cont.fit_result.aic:.1f}")
cont.plot()


# --- 2. Compare polynomial degrees -----------------------------------------
# Lower degree is more rigid; higher degree may chase noise. AIC/BIC are
# the principled way to pick.
print("\nDegree sweep:")
for d in [1, 2, 3, 4, 5]:
    c = ContinuumFit(search, xrange=xrange, degree=d, mask_fwhm=3.0)
    print(
        f"  degree={d}  redchi={c.fit_result.redchi:8.3f}  "
        f"aic={c.fit_result.aic:9.1f}  bic={c.fit_result.bic:9.1f}"
    )


# --- 3. subtract() — view the peaks alone ----------------------------------
# Useful for sanity-checking that the baseline picks up the slow variation
# and leaves the peaks largely intact.
x_sub, y_sub = cont.subtract()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_sub, y_sub, drawstyle="steps-mid", lw=1)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(spect.x_units)
ax.set_ylabel("counts - continuum")
ax.set_title("Spectrum with polynomial continuum subtracted")


# --- 4. evaluate() — query the baseline at arbitrary x ---------------------
# E.g. estimate the baseline under a specific peak you want to integrate.
# Use the first peak that lies inside our fit window.
peaks_in = search.peaks_idx[in_range[search.peaks_idx]]
peak_x = cont.x[peaks_in[0]]
baseline_at_peak = cont.evaluate([peak_x]).item()
print(f"\nBaseline at peak x={peak_x}: {baseline_at_peak:.2f} counts")


# --- 5. mask_fwhm sensitivity ----------------------------------------------
# Too small → peaks contaminate the fit (redchi balloons). Too large →
# you throw away most of the continuum and the polynomial flaps around.
print("\nmask_fwhm sweep:")
for m in [1.0, 2.0, 3.0, 5.0, 10.0]:
    try:
        c = ContinuumFit(search, xrange=xrange, degree=3, mask_fwhm=m)
        print(
            f"  mask_fwhm={m:4.1f}  kept={c.continuum_mask.sum():5d}  "
            f"redchi={c.fit_result.redchi:8.3f}"
        )
    except ValueError as e:
        print(f"  mask_fwhm={m:4.1f}  FAILED: {e}")

plt.show()

# Estimate the continuum in a region where there are no peaks
xrange2 = [7750, 8500] # no peaks
cont = ContinuumFit(search, xrange=xrange2, degree=1, mask_fwhm=3.0)
in_range = (cont.x >= xrange[0]) & (cont.x <= xrange[1])
print(f"\nContinuum fit on {xrange}:")
print(f"  degree            = {cont.degree}")
print(f"  channels in range = {in_range.sum()}")
print(f"  channels kept     = {cont.continuum_mask.sum()}")
print(f"  channels masked   = {in_range.sum() - cont.continuum_mask.sum()}")
print(f"  redchi            = {cont.fit_result.redchi:.3f}")
print(f"  aic               = {cont.fit_result.aic:.1f}")
cont.plot()