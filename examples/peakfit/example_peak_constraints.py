"""
Constraining a fit region: number of peaks, centroid, FWHM and intensity ratio.

Real spectra often contain overlapping or weak lines that a free fit handles
badly: the peak finder merges or misses them, and even when seeded their
parameters are poorly determined. ``MultiProfilePeakFit`` accepts constraint
kwargs (referenced by energy, so you work in the units you think in) to inject
physical knowledge into the fit:

  * ``peaks=[...]``      — declare exactly how many peaks the region holds and
                          roughly where (replaces the automatic search);
  * ``fix_center={E: v}``— pin a centroid to a known line energy;
  * ``fix_fwhm={E: w}``  — fix a width to the detector resolution (or bound it
                          with a ``(min, max)`` tuple);
  * ``ratios=[(a, b, r)]``— tie areas, ``area(a) = r * area(b)`` (a known
                          branching ratio, escape-peak fraction, ...).

All of them are translated to lmfit parameter hints, so any ``hints=`` you pass
still works and wins on conflicts.

This script builds a *synthetic* spectrum with a known answer — a strong line at
1330 keV and a weak, overlapping companion at 1339 keV (12% of its area) on a
flat background — so each constraint's effect can be checked against the truth.
"""
import numpy as np
import matplotlib.pyplot as plt

from wara.spectrum import Spectrum
from wara import peaksearch as ps
from wara.advanced_fit import MultiProfilePeakFit


# --- A synthetic doublet with a known answer -------------------------------
def gaussian(x, height, center, sigma):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


BIN = 0.5
energy = np.arange(1300.0, 1380.0, BIN)
SIGMA = 2.6                                    # => FWHM 6.12 keV
H_STRONG, H_WEAK = 22000.0, 2600.0             # weak line is ~12% of the strong
TRUE_RATIO = H_WEAK / H_STRONG
TRUE_FWHM = 2.355 * SIGMA

truth = (350.0
         + gaussian(energy, H_STRONG, 1330.0, SIGMA)
         + gaussian(energy, H_WEAK, 1339.0, SIGMA))
counts = np.random.default_rng(12).poisson(truth).astype(float)

spect = Spectrum(counts=counts, energies=energy, e_units="keV", livetime=100)
search = ps.PeakSearch(spect, ref_x=1330, ref_fwhm=TRUE_FWHM, fwhm_at_0=0.5,
                       min_snr=8, method="km")

XRANGE = [1314, 1358]
# Net counts under each true peak (Gaussian integral / bin width).
true_area = lambda h: h * SIGMA * np.sqrt(2 * np.pi) / BIN
print(f"TRUE answer: A(1330)={true_area(H_STRONG):.0f}, "
      f"B(1339)={true_area(H_WEAK):.0f} counts, ratio={TRUE_RATIO:.3f}, "
      f"FWHM={TRUE_FWHM:.2f} keV\n")


def weak_peak_row(fit):
    """Pull the weak (1339 keV) peak's parameters from a fit summary."""
    s = fit.summary()
    b = s.iloc[(s["mean"] - 1339).abs().idxmin()]
    return (f"center={b['mean']:.2f}+/-{b['mean_err']:.2f}  "
            f"FWHM={b['fwhm']:.2f}+/-{b['fwhm_err']:.2f}  "
            f"area={b['area']:.0f}+/-{b['area_err']:.0f}")


# --- 0. Automatic fit: the weak line is missed -----------------------------
auto = MultiProfilePeakFit(search, XRANGE, bkg="poly0", profile="gauss")
print(f"0. Automatic search    : {len(auto.peak_info)} peak(s), "
      f"reduced chi2 = {auto.fit_result.redchi:.1f}  <- weak line missed\n")

# --- 1. number of peaks: declare both lines --------------------------------
free = MultiProfilePeakFit(search, XRANGE, bkg="poly0", profile="gauss",
                           peaks=[1330, 1339])
print("1. peaks=[1330, 1339]  (declare two peaks)")
print(f"   reduced chi2 = {free.fit_result.redchi:.2f}   weak peak: "
      f"{weak_peak_row(free)}")

# --- 2. fix the centroid to a known line energy ----------------------------
fixed_c = MultiProfilePeakFit(search, XRANGE, bkg="poly0", profile="gauss",
                              peaks=[1330, 1339], fix_center={1339: 1339.0})
print("\n2. fix_center={1339: 1339.0}  (pin the weak centroid)")
print(f"   weak peak: {weak_peak_row(fixed_c)}")

# --- 3. fix the FWHM to the detector resolution ----------------------------
fixed_w = MultiProfilePeakFit(search, XRANGE, bkg="poly0", profile="gauss",
                              peaks=[1330, 1339], fix_fwhm={1339: TRUE_FWHM})
print(f"\n3. fix_fwhm={{1339: {TRUE_FWHM:.2f}}}  (pin the weak width)")
print(f"   weak peak: {weak_peak_row(fixed_w)}")

# --- 4. tie the areas by a known ratio -------------------------------------
tied = MultiProfilePeakFit(search, XRANGE, bkg="poly0", profile="gauss",
                           peaks=[1330, 1339],
                           ratios=[(1339, 1330, TRUE_RATIO)])
print(f"\n4. ratios=[(1339, 1330, {TRUE_RATIO:.3f})]  (tie weak area to strong)")
print(f"   weak peak: {weak_peak_row(tied)}")
print("   -> the area uncertainty shrinks the most here: the weak line's "
      "strength\n      is now carried by the well-measured strong line.")

# --- Plot: automatic (blended) vs constrained (resolved) -------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, fit, title in ((ax0, auto, "Automatic: doublet blended into one peak"),
                       (ax1, free, "peaks=[1330, 1339]: doublet resolved")):
    ax.plot(spect.energies, spect.counts, ".", ms=3, color="0.5", label="data")
    xf = np.linspace(*XRANGE, 500)
    ax.plot(xf, fit.fit_result.eval(x=xf), "r-", lw=1.8, label="fit")
    ax.set_xlim(*XRANGE)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Energy (keV)")
    ax.legend()
ax0.set_ylabel("Counts")
fig.tight_layout()
plt.show()
