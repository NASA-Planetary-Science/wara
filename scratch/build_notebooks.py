"""
Build replacement Jupyter notebooks from the example Python scripts.
Run from the repo root: python scratch/build_notebooks.py
"""
import json
import uuid

def cell_id():
    return uuid.uuid4().hex[:16]

def md(source):
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(),
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    }

METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

def notebook(cells):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": METADATA, "cells": cells}

def write(path, nb):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Written: {path}")


# ── 1. Spectrum ──────────────────────────────────────────────────────────────
nb1 = notebook([
    md("# Spectrum class example\n\nMinimum working example of the `Spectrum` class, including rebinning and smoothing."),
    code([
        "%matplotlib inline\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "from wara import spectrum as sp\n",
        "from wara import file_reader",
    ]),
    md("Load a calibrated CeBr3 spectrum from a CSV file."),
    code([
        'file = "data/test_data_cebr_cal.csv"\n',
        "df = pd.read_csv(file)",
    ]),
    md("Instantiate a `Spectrum` object. Energies are optional — if omitted, channel numbers are used."),
    code([
        'spect = sp.Spectrum(counts=df["counts"], energies=df["Energy [keV]"], e_units="keV")\n',
        "print(spect)\n",
        "print(spect.metadata())",
    ]),
    md("Plot the original spectrum, then rebin by 2 and apply moving-average smoothing. Rebinning conserves total counts; smoothing does not."),
    code([
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        'spect.label = "Original"\n',
        "spect.plot(ax=ax)\n",
        "\n",
        'spect.label = "Rebinned by 2"\n',
        "spect.rebin(by=2)\n",
        "spect.plot(ax=ax)\n",
        "\n",
        'spect.label = "Smoothed"\n',
        "spect.smooth(num=6)\n",
        "spect.plot(ax=ax)\n",
        "plt.show()",
    ]),
])
write("examples/1.spectrum_example.ipynb", nb1)


# ── 2. Peak search ───────────────────────────────────────────────────────────
nb2 = notebook([
    md("# Peak search example — CeBr3 detector\n\nDemonstrates the `PeakSearch` class on a CeBr3 spectrum."),
    code([
        "%matplotlib inline\n",
        "import matplotlib.pyplot as plt\n",
        "from wara import peaksearch as ps\n",
        "from wara import file_reader",
    ]),
    md("Load the spectrum."),
    code([
        'file = "data/test_data_cebr.csv"\n',
        "spect = file_reader.read_csv(file)",
    ]),
    md("Set the resolution model parameters (all in channels).\n\n"
       "- `fwhm_at_0`: FWHM extrapolated to channel 0\n"
       "- `ref_x`: reference channel for the FWHM measurement\n"
       "- `ref_fwhm`: measured FWHM at `ref_x`"),
    code([
        "fwhm_at_0 = 1\n",
        "ref_fwhm = 20\n",
        "ref_x = 420",
    ]),
    md("Run the peak search and plot the kernel, the detected peaks, and the spectral components."),
    code([
        'search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=5, method="km")\n',
        "search.plot_kernel()\n",
        "plt.show()\n",
        "search.plot()\n",
        "plt.show()\n",
        "search.plot_components()\n",
        "plt.show()",
    ]),
    md("Search within a restricted channel range and inspect metadata."),
    code([
        "search1 = ps.PeakSearch(\n",
        '    spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=5, xrange=[1200, 1600], method="km"\n',
        ")\n",
        'search1.plot(snrs="off")\n',
        "plt.show()\n",
        "print(search1.metadata())",
    ]),
    md("List peaks found within a specific channel range."),
    code([
        "search.peaks_in_range(x_min=0, x_max=300)",
    ]),
])
write("examples/2.peaksearch_example.ipynb", nb2)


# ── 3. Peak fitting ──────────────────────────────────────────────────────────
nb3 = notebook([
    md("# Peak fitting example\n\nDemonstrates the `PeakFit` class and area calculation."),
    code([
        "%matplotlib inline\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from wara import peaksearch as ps\n",
        "from wara import peakfit as pf\n",
        "from wara import file_reader",
    ]),
    md("Load the spectrum and set resolution parameters."),
    code([
        'file = "data/test_data_cebr.csv"\n',
        "spect = file_reader.read_csv(file)\n",
        "\n",
        "fwhm_at_0 = 1.0\n",
        "ref_fwhm = 31\n",
        "ref_x = 1220\n",
        "min_snr = 5",
    ]),
    md("Run the peak search."),
    code([
        "search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=min_snr)\n",
        "search.plot()\n",
        "plt.show()",
    ]),
    md("Fit peaks in the range [1250, 1600] channels using a linear background."),
    code([
        'bkg = "poly1"\n',
        "xrange = [1250, 1600]\n",
        "fit = pf.PeakFit(search, xrange, bkg=bkg)\n",
        "fit.plot()\n",
        "plt.show()\n",
        "print(fit.peak_info)",
    ]),
    md("Compare the peak area from `PeakFit` against an independent Gaussian integration."),
    code([
        "def gaussian(x, A, mu, sig):\n",
        "    return A / np.sqrt(2 * np.pi * sig**2) * np.exp(-((x - mu)**2) / (2 * sig**2))\n",
        "\n",
        "res = fit.fit_result\n",
        "xg = fit.x_data\n",
        'Ag  = res.best_values["g1_amplitude"]\n',
        'mug = res.best_values["g1_center"]\n',
        'sig = res.best_values["g1_sigma"]\n',
        "\n",
        'print(f"Area from PeakFit:          {Ag:.1f}")\n',
        'print(f"Area from Gaussian integral: {gaussian(xg, Ag, mug, sig).sum():.1f}")',
    ]),
])
write("examples/3.peakfitting_example.ipynb", nb3)


# ── 4. Energy calibration ────────────────────────────────────────────────────
nb4 = notebook([
    md("# Energy calibration example\n\nCalibrates a CeBr3 spectrum using Co-60 and Cs-137 gamma lines."),
    code([
        "%matplotlib inline\n",
        "import matplotlib.pyplot as plt\n",
        "from wara import peaksearch as ps\n",
        "from wara import peakfit as pf\n",
        "from wara import energy_calibration as ecal\n",
        "from wara import file_reader",
    ]),
    md("Load the spectrum and run a peak search."),
    code([
        'file = "data/test_data_cebr.csv"\n',
        "spect = file_reader.read_csv(file)\n",
        "\n",
        "fwhm_at_0 = 1.0\n",
        "ref_x = 1315\n",
        "ref_fwhm = 42\n",
        "search = ps.PeakSearch(spect, ref_x, ref_fwhm, fwhm_at_0, min_snr=4)",
    ]),
    md("Fit peaks in the range [680, 1600] channels."),
    code([
        "xrange = [680, 1600]\n",
        'bkg = "poly1"\n',
        "fit = pf.PeakFit(search, xrange=xrange, bkg=bkg)\n",
        "fit.plot()\n",
        "plt.show()\n",
        "print(fit.peak_info)",
    ]),
    md("Map fitted peak centroids to known gamma energies (Cs-137: 662 keV, Co-60: 1173.2 and 1332.5 keV) and build the calibration."),
    code([
        "peak_info = fit.peak_info\n",
        "centroids = [peak_info[0]['mean'], peak_info[2]['mean'], peak_info[3]['mean']]\n",
        "energies_kev = [662, 1173.2, 1332.5]\n",
        "\n",
        "cal = ecal.EnergyCalibration(\n",
        "    mean_vals=centroids, erg=energies_kev,\n",
        "    channels=spect.channels, n=1, e_units='keV'\n",
        ")\n",
        "cal.plot()\n",
        "plt.show()",
    ]),
    md("Apply the calibration to the spectrum and plot."),
    code([
        "spect.apply_calibration(cal)\n",
        "spect.plot()\n",
        'plt.title("Calibrated spectrum")\n',
        "plt.show()",
    ]),
])
write("examples/4.energy-calibration.ipynb", nb4)


# ── 5. Exponential decay ─────────────────────────────────────────────────────
nb5 = notebook([
    md("# Exponential decay fitting example\n\n"
       "Demonstrates single- and double-exponential fits with the `Decay_exp` class using synthetic data."),
    code([
        "%matplotlib inline\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from wara import decay_exponential as decay",
    ]),
    md("Generate synthetic single-decay data with Gaussian noise.\n\n"
       "True parameters: `A=1000`, `k=0.05`, `C=50`."),
    code([
        "RNG = np.random.default_rng(42)\n",
        "\n",
        "t = np.linspace(1, 100, 60)\n",
        "noise = 15.0\n",
        "y_single = decay.single_decay_plus_constant(t, A=1000, k=0.05, C=50)\n",
        "y_single += RNG.normal(0, noise, size=len(t))\n",
        "yerr_single = np.full_like(t, noise)",
    ]),
    md("Fit and plot the single exponential decay."),
    code([
        "exp1 = decay.Decay_exp(x=t, y=y_single, yerr=yerr_single)\n",
        "exp1.fit_single_decay()\n",
        "print(exp1.fit_result.fit_report())\n",
        "\n",
        "ax_fit, ax_res = exp1.plot(show_components=True)\n",
        'ax_fit.set_title("Single exponential decay")\n',
        "plt.show()",
    ]),
    md("Generate synthetic double-decay data.\n\n"
       "True parameters: `A1=700`, `k1=0.04`, `A2=300`, `k2=0.3`, `C=30`."),
    code([
        "y_double = decay.double_decay_plus_constant(t, A1=700, A2=300, k1=0.04, k2=0.3, C=30)\n",
        "y_double += RNG.normal(0, noise, size=len(t))\n",
        "yerr_double = np.full_like(t, noise)",
    ]),
    md("Fit and plot the double exponential decay."),
    code([
        "exp2 = decay.Decay_exp(x=t, y=y_double, yerr=yerr_double)\n",
        "exp2.fit_double_decay()\n",
        "print(exp2.fit_result.fit_report())\n",
        "\n",
        "ax_fit, ax_res = exp2.plot(show_components=True)\n",
        'ax_fit.set_title("Double exponential decay")\n',
        "plt.show()",
    ]),
])
write("examples/5.decay_exponential.ipynb", nb5)
