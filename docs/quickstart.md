# Quickstart

wara can be used two ways: through its graphical interface, or as a Python
library. This page gives a quick tour of both.

## Launching the GUI

After installation, launch the GUI from any directory:

```bash
wara
```

As of **v2.0**, `wara` opens the redesigned interface — a three-column layout
with a navigation rail (Spectrum, Calibration, Efficiency, Resolution, API,
Planetary), a contextual options panel, and the plot area.

```{note}
The previous interface is still available during the transition period via
`wara --legacy`. (`wara --beta` is also accepted as an explicit alias for the
new GUI.)
```

### A quick tour

1. Click **Open Spectrum** in the navigation rail and load
   **examples → data → test_data_cebr.csv**.
2. On the **Spectrum** tab, expand **Auto-Find Peaks**, choose the
   **LaBr/CeBr** detector preset, and click **Find Peaks**. Several lines
   should be identified in the spectrum.
3. Click on the "Drag and fit" button and drag the mouse over one or more identified lines to perform
   a Gaussian fit over a linear background.

From there, explore the other fitting techniques and tabs — calibration, efficiency, resolution, and
API visualization. Tooltips are placed throughout the GUI to provide
contextual guidance. Simply hover over a feature for a quick explanation.

### Command-line options

You can open a file and preconfigure the peak search directly from the command
line:

```bash
wara examples/data/test_data_cebr.csv --labr --min_snr=5
```

| Option | Description |
|--------|-------------|
| `<file_name>` | Spectrum file to open on startup |
| `--labr` / `--cebr` | Use the LaBr/CeBr detector preset |
| `--hpge` | Use the HPGe detector preset |
| `--min_snr=<n>` | Minimum peak SNR for the peak search |
| `--ref_x=<x>` | Reference channel/energy for the FWHM curve |
| `--ref_fwhm=<f>` | FWHM at the reference point |
| `--fwhm_at_0=<f>` | FWHM value at channel 0 |
| `--legacy` | Launch the pre-2.0 GUI |
| `-h` / `--help` | Show usage and exit |

Run `wara --help` for the complete list.

## Using wara in Python

Every analysis step in the GUI is backed by a Python class you can use
directly in a script or notebook:

```python
from wara import peaksearch as ps
from wara import peakfit as pf
from wara import file_reader

# Load a spectrum from a CSV file
file = "examples/data/test_data_cebr_cal.csv"
spect = file_reader.read_csv(file)
ax_spe = spect.plot()

# Search for peaks given the detector resolution and a minimum SNR
search = ps.PeakSearch(spectrum=spect, ref_x=420, ref_fwhm=20, min_snr=5)
search.plot()

# Fit the peaks in a chosen x-range over a linear background
fit = pf.PeakFit(search=search, xrange=[1080, 1450], bkg="poly1")
fit.plot()

# Print some useful information about the first fitted peak
print(f"Peak 1 Energy: {fit.peak_info[0]['mean']} keV")
print(f"Peak 1 Area:   {fit.peak_info[0]['area']} counts")
print(f"Peak 1 FWHM:   {fit.peak_info[0]['fwhm']} keV")
```

```{tip}
`ref_x` and `ref_fwhm` describe the detector resolution: the FWHM
(`ref_fwhm`, in channels) of a known peak at a reference position (`ref_x`).
Together with `fwhm_at_0` they define the FWHM-vs-channel curve the peak
search uses to distinguish real peaks from noise.
```

See the [User Guide](spectrum.md) for a deeper look at spectra, peak searching,
fitting, and calibration.
