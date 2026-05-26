"""
Example usage of MultiProfilePeakFit (wara.advanced_fit).

The basic PeakFit in wara.peakfit is intentionally Gaussian-only.
For HPGe and similar detectors, a Voigt or Pseudo-Voigt line shape
often fits the wings better. MultiProfilePeakFit drops in alongside
PeakFit and accepts a `profile=` kwarg to pick the line shape family.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from wara import file_reader
from wara import peaksearch as ps
from wara.advanced_fit import MultiProfilePeakFit


# HPGe spectrum: narrower peaks, where line-shape choice matters.
file = Path(__file__).parent.parent / "data/test_data_hpge_NH3.txt"
spect = file_reader.read_txt(file)

ref_fwhm = 3
ref_x = 420
min_snr = 15
search = ps.PeakSearch(spect, ref_x, ref_fwhm, min_snr=min_snr, method="fast")

# Pick a single, well-isolated peak so the comparison is fair.
xrange = [2820, 2845]

profiles = ["gauss", "voigt", "pvoigt"]
fits = {}
for prof in profiles:
    fit = MultiProfilePeakFit(search, xrange, bkg="poly1", profile=prof)
    fits[prof] = fit
    q = fit.fit_quality()
    print(
        f"profile={prof:7s}  redchi={q['redchi']:8.4f}  "
        f"aic={q['aic']:9.2f}  bic={q['bic']:9.2f}  "
        f"normaltest_p={q['normaltest_pvalue']:.3f}"
    )

# Lower AIC/BIC = better. For an HPGe peak Voigt or PseudoVoigt usually
# wins; for scintillator data a plain Gaussian is typically fine.
best = min(fits, key=lambda p: fits[p].fit_result.aic)
print(f"\nBest profile by AIC: {best}")

# Plot each fit in its own figure (PeakFit.plot creates a 2-row layout
# for residual + fit on its own).
for prof in profiles:
    fits[prof].plot()
    plt.gcf().suptitle(f"profile = {prof}")
plt.show()
