"""
Example usage of the chi-squared optimizer.
It finds the x-range that produces the fit with the best chi-squared.

The optimizer searches the left and right edges of the fit window
independently by default (asymmetric grid). Pass `symmetric=True` to
recover the older behaviour of moving both edges by the same offset —
cheaper (n_steps trials vs n_steps**2) but less expressive when one side
of the continuum is harder to capture than the other.
"""
from wara import peaksearch as ps
from wara import peakfit as pf
from wara import file_reader


# dataset 1
file = "../data/test_data_lab_sources.cnf"

# Required input parameters (in channels)
fwhm_at_0 = 1.0
ref_fwhm = 20
ref_x = 420
min_snr = 5

# instantiate a Spectrum object
spect = file_reader.read_cnf(file)

# peaksearch class
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
search.plot()

# peakfit class
bkg0 = "poly1"
xrange = [573, 666]
fit = pf.PeakFit(search, xrange, bkg=bkg0)
print(f"Initial xrange  = {fit.xrange}")
print(f"Reduced chi-squared = {fit.fit_result.redchi:.4f}")
fit.plot()

# --- Asymmetric optimizer (default) ----------------------------------------
# n_steps=10 per edge → 100 trial fits. Each edge of the window is allowed
# to widen by up to max_extend * avg_fwhm independently.
best_range, best_redchi = fit.optimize_xrange(
    max_extend=5.0, n_steps=10, verbose=True
)
fit.plot()
print(f"\nBest xrange (asymmetric) = {best_range}")
print(f"Reduced chi-squared      = {best_redchi:.4f}")


# --- Symmetric optimizer (legacy behaviour) --------------------------------
# Both edges move by the same offset. Cheaper (n_steps trials only) but
# can't compensate when only one side needs more continuum.
fit_sym = pf.PeakFit(search, [573, 666], bkg=bkg0)
sym_range, sym_redchi = fit_sym.optimize_xrange(
    max_extend=5.0, n_steps=50, symmetric=True, verbose=False
)
fit_sym.plot()
print(f"\nBest xrange (symmetric)  = {sym_range}")
print(f"Reduced chi-squared      = {sym_redchi:.4f}")

