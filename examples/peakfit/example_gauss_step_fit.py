"""
Example usage of GaussStepFit (wara.advanced_fit).

GaussStepFit fits a Gaussian peak on a *step* background instead of the usual
straight line. In gamma-ray spectra the continuum under a peak is often
higher on the low-energy side than the high-energy side: incomplete charge
collection and small-angle Compton scattering pile counts up just below the
full-energy peak. A linear background can't capture that — it either over- or
under-estimates the area depending on which side you anchor it to.

Two step shapes are available:

  * step="sharp"  (default) — a hard Heaviside discontinuity at the peak
        centroid:   bkg(x) = offset + step_amplitude * H(center - x)
  * step="smooth"           — a continuous erfc transition tied to the peak's
        centre and width:
                    bkg(x) = offset + step_amplitude * 0.5 * erfc((x-center)/(sqrt(2)*sigma))

Both drop from ``offset + step_amplitude`` on the low-energy side to
``offset`` on the high-energy side, with the step location tied to the peak.
The only extra free parameters versus a constant background are the step
height and the offset.

The public interface mirrors PeakFit: summary(), fit_quality(), plot(),
save_json()/load_json() all work the same way.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from wara import file_reader
from wara import peaksearch as ps
from wara import peakfit as pf
from wara.advanced_fit import GaussStepFit


# Calibrated CeBr spectrum — the ~1158 keV peak sits on a clear Compton step.
file = Path(__file__).parent.parent / "data/test_data_cebr_cal.csv"
spect = file_reader.read_csv(file)

# PeakSearch parameters are in channels (ref_x, ref_fwhm, fwhm_at_0).
search = ps.PeakSearch(spect, ref_x=1220, ref_fwhm=31, fwhm_at_0=1.0, min_snr=5)


# --- 1. Fit a Gaussian peak on a sharp step background (the default) --------
xrange = [1070, 1260]
fit = GaussStepFit(search, xrange)          # step="sharp" by default

print(f"GaussStepFit (sharp) on {xrange} keV:")
print(fit.summary().to_string(index=False))

bv = fit.fit_result.best_values
print("\nBackground (sharp step):")
print(f"  offset (high-E plateau) = {bv['bkg_offset']:.2f} counts")
print(f"  step height             = {bv['bkg_step_amplitude']:.2f} counts")
print(f"  step edge tied to the peak centroid (center={bv['g1_center']:.2f})")

quality = fit.fit_quality()
print("\nFit quality:")
for k, v in quality.items():
    print(f"  {k:>20s} = {v}")

fit.plot()


# --- 1b. Same peak with the smoothed (erfc) step, for comparison -----------
fit_smooth = GaussStepFit(search, xrange, step="smooth")
print(f"\nGaussStepFit (smooth) redchi = {fit_smooth.fit_result.redchi:.3f}, "
      f"area = {fit_smooth.peak_info[0]['area']:.1f}")
fit_smooth.plot()


# --- 2. Compare against a straight-line background --------------------------
# Same peak, same window, linear background. Look at where each background
# sits on the low- vs high-energy side of the peak: the linear fit has to
# compromise, the step follows the data.
lin = pf.PeakFit(search, xrange, bkg="linear")

print("\nLinear-background fit for comparison:")
print(lin.summary().to_string(index=False))
print(f"  net area  step={fit.peak_info[0]['area']:.1f}  "
      f"linear={lin.peak_info[0]['area']:.1f}")

fig, (ax_res, ax_fit) = plt.subplots(
    2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [1, 4]}
)
fit.plot(fig=fig, ax_res=ax_res, ax_fit=ax_fit)
ax_fit.set_title("Gaussian + step background")


# --- 3. Pin a parameter via hints -------------------------------------------
# hints work exactly as in PeakFit — e.g. fix the centroid to a known line.
fit_pinned = GaussStepFit(
    search, xrange,
    hints={"g1_center": {"value": 1158.0, "vary": False}},
)
print(f"\nWith centroid pinned to 1158 keV: "
      f"area = {fit_pinned.peak_info[0]['area']:.1f}, "
      f"redchi = {fit_pinned.fit_result.redchi:.3f}")

plt.show()
