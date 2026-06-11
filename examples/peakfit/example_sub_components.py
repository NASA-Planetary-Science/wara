"""
Plotting sub-components of the Hypermet peak profile.

The Hypermet model is a sum of a Gaussian core and an exponentially-modified
Gaussian (EMG) tail::

    peak(x) = amplitude * [ (1 - tail_fraction) * Gaussian(x)
                            + tail_fraction      * EMG_left(x) ]

Because the two parts are additive, each can be plotted individually — their
sum equals the full peak.  This example fits the tailed 6129 keV line of a
real HPGe spectrum with a plain Gaussian (for comparison) and then with a
Hypermet, showing the core and tail sub-components on the Hypermet plot.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from wara import file_reader
from wara import peaksearch as ps
from wara import advanced_fit as adv


file = Path(__file__).parent.parent / "data/test_data_hpge_NH3.txt"
spect = file_reader.read_txt(file)
search = ps.PeakSearch(spect, ref_x=420, ref_fwhm=3, min_snr=15, method="fast")

XR = [6100, 6150]

# --- Plain Gaussian for comparison ---
gauss = adv.MultiProfilePeakFit(search, XR, bkg="linear", profile="gauss")
gauss.plot()
plt.gcf().suptitle("Gaussian — note the structured low-energy residuals")

# --- Hypermet: Gaussian core + low-energy tail ---
hyper = adv.HypermetFit(search, XR, bkg="linear")
hyper.plot()
plt.gcf().suptitle("Hypermet — dotted lines show the Gaussian core and EMG tail")

plt.show()
