"""
Example usage of PeakFit's `shared_sigma` constraint.

When fitting several peaks in a tight window, each peak's sigma is often
poorly constrained on its own, especially if the peaks partially overlap.
`shared_sigma=True` links every peak's sigma through the detector's
resolution curve

    fwhm(E) = a + b * sqrt(E),   sigma = fwhm / 2.355

(the same parametrization that PeakSearch uses).  Two shared free
parameters `_fwhm_a` and `_fwhm_b` replace the per-peak sigmas, so the
fit cannot wander into the "one very wide peak swallows the others"
local minimum.

With a single peak in the window the slope `_fwhm_b` is automatically
fixed at the PeakSearch value (otherwise a, b are degenerate).
"""
from pathlib import Path

from wara import file_reader
from wara import peaksearch as ps
from wara import peakfit as pf


# HPGe spectrum — pick a window with two close peaks.
file = Path(__file__).parent.parent / "data/test_data_hpge_NH3.txt"
spect = file_reader.read_txt(file)

search = ps.PeakSearch(spect, ref_x=420, ref_fwhm=3, min_snr=15, method="fast")

# A multi-peak window — adjust if your data differs.
xrange = [946, 975]

# --- Free sigmas (default) -------------------------------------------------
fit_free = pf.PeakFit(search, xrange, bkg="poly1")
print("Free per-peak sigmas:")
print(fit_free.summary())
print(f"  redchi = {fit_free.fit_result.redchi:.4f}\n")

# --- Shared sigma via fwhm = a + b*sqrt(E) ---------------------------------
fit_shared = pf.PeakFit(search, xrange, bkg="poly1", shared_sigma=True)
print("Shared sigma (fwhm = a + b*sqrt(E)):")
print(fit_shared.summary())
print(f"  redchi      = {fit_shared.fit_result.redchi:.4f}")
print(
    f"  _fwhm_a     = {fit_shared.fit_result.params['_fwhm_a'].value:.4f} "
    f"+/- {fit_shared.fit_result.params['_fwhm_a'].stderr}"
)
print(
    f"  _fwhm_b     = {fit_shared.fit_result.params['_fwhm_b'].value:.4f} "
    f"+/- {fit_shared.fit_result.params['_fwhm_b'].stderr}"
)
# AIC penalizes free parameters: shared_sigma trades N per-peak sigmas
# for just 2 shared ones, so a comparable redchi usually means a clear
# AIC/BIC win.
print(
    f"  AIC: free = {fit_free.fit_result.aic:.2f}, "
    f"shared = {fit_shared.fit_result.aic:.2f}"
)

fit_free.plot()
fit_shared.plot()


# --- Pin the slope: hints can override either shared parameter --------------
# If you trust PeakSearch's b but want a to absorb any offset:
fit_pinned = pf.PeakFit(
    search,
    xrange,
    bkg="poly1",
    shared_sigma=True,
    hints={"_fwhm_b": {"vary": False}},
)
print("\nShared sigma with b pinned:")
print(fit_pinned.summary())
print(f"  redchi = {fit_pinned.fit_result.redchi:.4f}")
