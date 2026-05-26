"""
Shared-sigma example on an energy-calibrated CeBr spectrum.

CeBr peaks are broader than HPGe, so a window that spans a few hundred
keV often contains several partially-overlapping peaks where individual
sigmas are poorly constrained. `shared_sigma=True` ties them together
through the detector's resolution curve

    fwhm(E) = |a + b * sqrt(E)|

which is what PeakSearch uses internally. This typically stabilises the
fit and yields a clear AIC win because we trade N per-peak sigmas for
just 2 shared ones.

Because the spectrum is energy-calibrated, `xrange` is given in keV. The
script also overlays each fit's per-peak FWHM against the resolution
curve predicted by PeakSearch, so you can see at a glance which case
follows the expected curve most cleanly.
"""
import matplotlib.pyplot as plt
import numpy as np

from wara import file_reader
from wara import peaksearch as ps
from wara import peakfit as pf


# Calibrated CeBr spectrum (units already in keV).
file = "../data/test_data_cebr_cal.csv"
spect = file_reader.read_csv(file)

search = ps.PeakSearch(
    spect, ref_x=420, ref_fwhm=20, fwhm_at_0=1.0, min_snr=3
)
print(f"PeakSearch found {len(search.peaks_idx)} peaks.")

# Energy window in keV.
xrange = [280, 526]

# --- Free per-peak sigmas (default) ----------------------------------------
fit_free = pf.PeakFit(search, xrange, bkg="poly1")
print(f"\nFree per-peak sigmas ({len(fit_free.peak_info)} peaks):")
print(fit_free.summary())
print(f"  redchi = {fit_free.fit_result.redchi:.4f}")
print(f"  aic    = {fit_free.fit_result.aic:.2f}")


# --- Shared sigma via fwhm = |a + b*sqrt(E)| --------------------------------
fit_shared = pf.PeakFit(search, xrange, bkg="poly1", shared_sigma=True)
print(f"\nShared sigma:")
print(fit_shared.summary())
p = fit_shared.fit_result.params
print(f"  redchi   = {fit_shared.fit_result.redchi:.4f}")
print(f"  aic      = {fit_shared.fit_result.aic:.2f}")
print(f"  _fwhm_a  = {p['_fwhm_a'].value:.4f} +/- {p['_fwhm_a'].stderr}")
print(f"  _fwhm_b  = {p['_fwhm_b'].value:.4f} +/- {p['_fwhm_b'].stderr}")
# With several peaks spanning a wide sqrt(E) range, the (a, b) pair is
# well identified — both should have finite, small relative errors. In
# the narrow-window degenerate case the optimizer still converges
# (the abs() in the expression handles negative b), but a and b may
# take unphysical signs while the derived FWHMs remain meaningful.

fit_free.plot()
fit_shared.plot()


# --- Pin b to the PeakSearch value, fit a only ------------------------------
# Useful when you trust the search's slope and only want a constant
# offset to absorb any miscalibration.
fit_pinned = pf.PeakFit(
    search,
    xrange,
    bkg="poly1",
    shared_sigma=True,
    hints={"_fwhm_b": {"vary": False}},
)
print(f"\nShared sigma with b pinned:")
print(fit_pinned.summary())
print(f"  redchi  = {fit_pinned.fit_result.redchi:.4f}")
print(f"  aic     = {fit_pinned.fit_result.aic:.2f}")


# ---------------------------------------------------------------------------
# FWHM vs PeakSearch's expected resolution curve
# ---------------------------------------------------------------------------
# PeakSearch's fwhm() takes a channel and returns FWHM in channels. To
# overlay it on an energy-axis plot we convert each side: channel -> keV
# via the calibration, and FWHM-channels -> FWHM-keV via the local
# dE/dch slope at that channel.
chan = spect.channels
energies = spect.energies

# Smooth expected curve over the fit window.
e_grid = np.linspace(xrange[0], xrange[1], 400)
ch_grid = np.interp(e_grid, energies, chan)
fwhm_ch_grid = search.fwhm(ch_grid)
# Local dE/dch via finite differences on the calibration.
dE_dch = np.gradient(energies, chan)
dE_dch_grid = np.interp(ch_grid, chan, dE_dch)
expected_fwhm_kev = fwhm_ch_grid * dE_dch_grid

cases = [
    ("free",      fit_free,   "o", "C0"),
    ("shared",    fit_shared, "s", "C1"),
    ("b pinned",  fit_pinned, "^", "C2"),
]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(
    e_grid,
    expected_fwhm_kev,
    "k--",
    lw=2,
    label="PeakSearch expected: fwhm(E) curve",
)
for label, fit, marker, color in cases:
    df = fit.summary()
    ax.errorbar(
        df["mean"],
        df["fwhm"],
        xerr=df["mean_err"],
        yerr=df["fwhm_err"],
        fmt=marker,
        color=color,
        ms=9,
        capsize=4,
        label=f"{label} (redchi={fit.fit_result.redchi:.3f})",
    )
ax.set_xlabel(f"Peak mean ({spect.x_units})")
ax.set_ylabel(f"FWHM ({spect.x_units})")
ax.set_title("Fitted FWHM vs PeakSearch resolution curve")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
