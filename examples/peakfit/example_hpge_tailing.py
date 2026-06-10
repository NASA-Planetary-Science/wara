"""
HPGe peak-fitting improvements: tailing models and shape diagnostics.

High-resolution HPGe full-energy peaks are not pure Gaussians. Incomplete
charge collection and small-angle scattering add a low-energy *tail*, and the
continuum can *step* down across the peak. Fitting a plain Gaussian to such a
peak leaves structured residuals on the low-energy flank and biases the area.
This example walks through the tools in ``wara.advanced_fit`` for handling
that, all on the strong, visibly-tailed 6129 keV line of a real HPGe spectrum:

  1. shape_metrics / shape_summary — measure FWHM, FWTM and tailing
     numerically from the fitted profile (valid for any line shape).
  2. MultiProfilePeakFit(profile="emg") — exponentially-modified Gaussian.
  3. HypermetFit — Gaussian core + low-energy exponential tail (GF3 style).
  4. FullHPGePeakFit — the Hypermet tail plus a step continuum.

For this peak a plain Gaussian gives reduced chi-square ~50; the tailed
models bring it down to ~2.4.

Note on the step background: this 6129 keV peak is strongly tailed but sits on
a flat continuum (no Compton step), so FullHPGePeakFit's step simply collapses
to ~0 and the tail does the work. The default *sharp* step is what keeps that
fit well-conditioned — a smooth step would share the peak's width with the
tail and the two become degenerate. See example_gauss_step_fit.py for a peak
that genuinely has a step.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from wara import file_reader
from wara import peaksearch as ps
from wara import advanced_fit as adv


# HPGe spectrum, rich in peaks. The 6129 keV line is strong and clearly tailed.
file = Path(__file__).parent.parent / "data/test_data_hpge_NH3.txt"
spect = file_reader.read_txt(file)
search = ps.PeakSearch(spect, ref_x=420, ref_fwhm=3, min_snr=15, method="fast")

XR = [6100, 6150]


# --- 1. Shape diagnostics: how much is this peak tailed? --------------------
# Fit a plain Gaussian first and measure its shape. A pure Gaussian has
# FWTM/FWHM = 1.8226 (adv.GAUSS_FWTM_FWHM) and zero asymmetry.
gauss = adv.MultiProfilePeakFit(search, XR, bkg="linear", profile="gauss")
print(f"Pure-Gaussian reference FWTM/FWHM = {adv.GAUSS_FWTM_FWHM:.4f}")
print(f"\nGaussian fit (reduced chi^2 = {gauss.fit_result.redchi:.2f}):")
print(adv.shape_summary(gauss).to_string(index=False))
print("  -> the *fit* is a Gaussian so its ratio is 1.823 by construction;")
print("     the large reduced chi^2 is what flags the un-modelled tail.")


# --- 2. EMG: an exponentially-modified Gaussian -----------------------------
emg = adv.MultiProfilePeakFit(search, XR, bkg="linear", profile="emg")
print(f"\nEMG fit (reduced chi^2 = {emg.fit_result.redchi:.2f}):")
print(adv.shape_summary(emg).to_string(index=False))


# --- 3. Hypermet: Gaussian core + low-energy tail ---------------------------
hyper = adv.HypermetFit(search, XR, bkg="linear")
tf = hyper.fit_result.best_values["g1_tail_fraction"]
print(f"\nHypermet fit (reduced chi^2 = {hyper.fit_result.redchi:.2f}):")
print(f"  tail fraction = {tf:.3f}  (fraction of the area in the tail)")
print(adv.shape_summary(hyper).to_string(index=False))


# --- 4. Full HPGe peak: Hypermet tail plus a step continuum -----------------
full = adv.FullHPGePeakFit(search, XR)   # step="sharp" by default
bv = full.fit_result.best_values
print(f"\nFull HPGe fit (reduced chi^2 = {full.fit_result.redchi:.2f}):")
print(f"  tail fraction = {bv['g1_tail_fraction']:.3f}")
print(f"  step height   = {bv['bkg_step_amplitude']:.0f} counts "
      f"(continuum drop across the peak — ~0 here, this peak has no step)")
print(adv.shape_summary(full).to_string(index=False))


# --- Summary table of model vs goodness-of-fit ------------------------------
print("\nModel comparison (lower reduced chi^2 = better):")
for name, fit in [("Gaussian", gauss), ("EMG", emg),
                  ("Hypermet", hyper), ("Full HPGe (tail+step)", full)]:
    m = adv.shape_metrics(fit)[0]
    print(f"  {name:24s} redchi={fit.fit_result.redchi:7.2f}  "
          f"area={fit.peak_info[0]['area']:8.2f}  "
          f"FWTM/FWHM={m['fwtm_fwhm']:.3f}  asym={m['asymmetry']:+.3f}")


# --- Plots ------------------------------------------------------------------
gauss.plot()
plt.gcf().suptitle("Gaussian — note the structured low-energy residuals")

full.plot()
plt.gcf().suptitle("Full HPGe — Hypermet tail (step ~0 on this flat continuum)")

plt.show()
