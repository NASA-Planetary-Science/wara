# Resolution (FWHM vs. Energy)

A detector's resolution — how the peak width grows with energy — is a useful
characterization in its own right and is exactly what the
[peak search](peak_finding.md) needs to tell peaks from noise.
{py:mod}`wara.resolution` fits the FWHM-vs-energy curve from a set of fitted
calibration peaks.

## Collecting the points

Fit your calibration peaks (see [Peak Fitting](peak_fitting.md)) and read each
one's centroid and width straight from the fit:

```python
energies = [info["mean"] for info in fit.peak_info]   # peak positions
fwhms    = [info["fwhm"] for info in fit.peak_info]    # peak FWHMs
```

In practice you collect these across several fits spanning the energy range.

## Fitting the curve

`fwhm_vs_erg` fits one of two standard resolution models and plots the data with
the best-fit curve, returning the underlying lmfit result:

```python
from wara.resolution import fwhm_vs_erg

fit = fwhm_vs_erg(energies, fwhms, x_units="Energy (keV)", e_units="keV", order=2)
print(fit.best_values)        # {"a": ..., "b": ..., "c": ...}
```

| `order` | Model |
|---------|-------|
| `1` | `FWHM = a + b·√E` |
| `2` | `FWHM = a + b·√(E + c·E²)` |

To draw the fitted curve beyond the measured range, pass the returned fit to
`fwhm_extrapolate`:

```python
import numpy as np
from wara.resolution import fwhm_extrapolate

fwhm_extrapolate(np.linspace(0, 3000, 200), fit, order=2)
```

`fwhm_table(energies, fwhms)` renders the points as a formatted table, including
the relative resolution (FWHM as a percentage of energy).

## Feeding resolution back into peak search

The fit's `a` and `b` describe how width scales with position — the same
information [`PeakSearch`](peak_finding.md) encodes through `ref_x`/`ref_fwhm`
and that `PeakFit(..., shared_sigma=True)` uses to tie peak widths to a single
resolution curve. Characterizing resolution once lets you set those parameters
consistently for a given detector.

## In the GUI

The **Resolution** tab in the navigation rail wraps this workflow: send fitted
peaks over from the Spectrum tab and the tab fits and plots FWHM vs. energy.

For the full API, see {py:mod}`wara.resolution` in the API reference.
