"""
Example usage of PeakFit's `hints` mechanism.

`hints` is a dict mapping parameter name to a dict of attributes that
lmfit's `Parameter.set` accepts (`value`, `min`, `max`, `vary`, `expr`).
Hints are applied AFTER the defaults and the shared_sigma expressions,
so they always win.

Common use cases:
  * fix a peak position to a known reference energy
  * bound an amplitude to keep the fit physical
  * tie one parameter to another via an expression
"""
from wara import file_reader
from wara import peaksearch as ps
from wara import peakfit as pf


file = "../data/test_data_lab_sources.cnf"
spect = file_reader.read_cnf(file)

search = ps.PeakSearch(spect, ref_x=420, ref_fwhm=20, fwhm_at_0=1.0, min_snr=5)

xrange = [573, 666]


# --- 1. Free fit (baseline) -------------------------------------------------
fit_free = pf.PeakFit(search, xrange, bkg="poly1")
print("Free fit:")
print(fit_free.summary())


# --- 2. Pin the peak center to a known reference ---------------------------
# Useful when calibrating against a known gamma line.
known_center = fit_free.peak_info[0]["mean"]  # pretend this is the literature value
fit_pinned = pf.PeakFit(
    search,
    xrange,
    bkg="poly1",
    hints={"g1_center": {"value": known_center, "vary": False}},
)
print("\nCenter pinned:")
print(fit_pinned.summary())
# Note that mean_err is NaN for the pinned center — lmfit reports no
# uncertainty for a non-varying parameter.


# --- 3. Bound a parameter --------------------------------------------------
# Keep amplitude strictly positive and constrain sigma to a sensible range.
fit_bounded = pf.PeakFit(
    search,
    xrange,
    bkg="poly1",
    hints={
        "g1_amplitude": {"min": 0.0},
        "g1_sigma": {"min": 0.5, "max": 50.0},
    },
)
print("\nBounded:")
print(fit_bounded.summary())


# --- 4. Tie parameters via expression --------------------------------------
# If your window had a known doublet with a fixed amplitude ratio (say
# branching ratio 0.5), you could tie one peak's amplitude to another:
#
#     hints={"g2_amplitude": {"expr": "0.5 * g1_amplitude"}}
#
# The expression mechanism uses lmfit's asteval; any parameter name in
# the current Parameters set is in scope.
