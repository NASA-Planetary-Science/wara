"""
Add peaks to a peaksearch object with energies and not channels
"""
from wara import spectrum as sp
import pandas as pd
import numpy as np
from wara import peaksearch as ps
from wara import peakfit as pf
from wara import file_reader

# dataset 1
file = "../data/test_data_Fe_API.txt"

# instantiate a Spectrum object
spect = file_reader.read_txt(file)

# Required input parameters (in channels)
fwhm_at_0 = 1
ref_fwhm = 20
ref_x = 420
min_snr = 15

# instantiate a peaksearch object
search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)
search.plot()

# fit
fit = pf.PeakFit(search=search, xrange=[1600, 1926])
fit.plot()


# # assume we want to add a peak at x=1663 keV
xnew_e = 1663  
xnew = np.where(spect.energies >= xnew_e)[0][0] # remove if in channels
x_idx = np.searchsorted(search.peaks_idx, xnew)
search.peaks_idx = np.insert(search.peaks_idx, x_idx, xnew)
fwhm_guess_new = search.fwhm(xnew)
search.fwhm_guess = np.insert(search.fwhm_guess, x_idx, fwhm_guess_new)

search.plot()

# fit
fit = pf.PeakFit(search=search, xrange=[1600, 1926])
fit.plot()
