# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:04:30 2026

@author: mayll
"""

from wara import peaksearch as ps
from wara import peakfit as pf
from wara import file_reader

# Load a spectrum from a CSV file
file = "examples/data/test_data_cebr_cal.csv"
spect = file_reader.read_csv(file)
ax_spe = spect.plot()

# Search for peaks
search = ps.PeakSearch(spectrum=spect, ref_x=420, ref_fwhm=20, min_snr=5)
search.plot()

# Fit the peaks
fit = pf.PeakFit(search=search, xrange=[1080,1450], bkg="poly1")
fit.plot()

# Print some useful information
print(f"Peak 1 Energy: {fit.peak_info[0]['mean']} keV")
print(f"Peak 1 Area: {fit.peak_info[0]['area']} counts")
print(f"Peak 1 FWHM: {fit.peak_info[0]['fwhm']} keV")