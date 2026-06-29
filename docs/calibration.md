# Energy Calibration

Energy calibration maps a spectrum's channel axis to physical energy,
`E = f(channel)`, by anchoring known gamma lines to the channel positions where
they appear. wara fits the mapping with {py:mod}`wara.energy_calibration` and
can apply it back onto a {py:class}`~wara.spectrum.Spectrum`.

## The calibration points

You need two matched lists:

- **channel positions** — the fitted centroids of your calibration peaks
  (`fit.peak_info[i]["mean"]` from [Peak Fitting](peak_fitting.md), in channels
  for an uncalibrated spectrum), and
- **reference energies** — the literature energies of those lines.

```python
channels = [237, 351, 609, 1120, 1460]      # fitted peak centroids
energies = [121.8, 344.3, 778.9, 1112.1, 1408.0]   # keV
```

## Polynomial calibration

`EnergyCalibration` fits a degree-`n` polynomial through the points (default
`n=1`, a straight line). It needs the calibration points plus the spectrum's
full channel array so it can predict an energy for every bin:

```python
from wara.energy_calibration import EnergyCalibration

cal = EnergyCalibration(channels, energies, spect.channels, n=1)
cal.plot()              # calibration curve with residuals
print(cal.rsquared)     # goodness of fit
print(cal.metadata())   # n, units, coefficients, R², points
```

```{tip}
Keep the polynomial degree as low as the data justifies — a linear or quadratic
fit is almost always right for a well-behaved detector. A high-degree polynomial
will chase noise in the calibration points and can become non-monotonic between
them.
```

## Applying a calibration to a spectrum

Once fitted, apply the calibration in place. The spectrum's x-axis switches from
channels to energy, and the fit equation is recorded in its metadata:

```python
spect.apply_calibration(cal)   # spect.x is now energy; x_units -> "Energy (keV)"
spect.plot()

spect.remove_calibration()     # revert to the channel axis
```

`apply_calibration` accepts either an `EnergyCalibration` or a
`PiecewiseLinearCalibration` (below). Downstream tools — peak search, fitting,
ROI counts — then interpret their x-ranges in energy.

## Automatic peak ↔ energy matching

Lining up detected peaks with reference energies by hand is tedious and error
prone, especially when the peak list contains spurious lines or misses some.
Two helpers automate it.

### `smart_calibration`

Best when your two lists are clean and you want to pair *all* of the shorter
list against the best-matching subset of the longer one. It searches every
order-preserving pairing and keeps the highest R² (linear `n=1` or quadratic
`n=2`):

```python
from wara.energy_calibration import smart_calibration

best = smart_calibration(channels, energies, n=1)
# best -> {"c0", "c1", "r2", "channels", "energies", "n", (and "c2" if n=2)}
```

### `smart_calibration_auto`

Best when either list may contain extras — a noise peak among the channels, or a
reference line that isn't actually present. It uses a RANSAC-style search to find
the calibration that maps the **largest consistent subset** of channels onto
energies within a tolerance, leaving outliers unmatched (degrees 1–3):

```python
from wara.energy_calibration import smart_calibration_auto

best = smart_calibration_auto(channels, energies, n=1,
                              channel_range=(0, len(spect.channels)))
print(best["n_matched"], "peaks matched")
print(best["unmatched_channels"], best["unmatched_energies"])
```

```{tip}
Pass the spectrum's full span as `channel_range=(0, len(spect.channels))` so the
fit is required to stay monotonic everywhere it will be applied, not just across
the matched points. `tol` (keV) defaults to a fraction of the smallest gap
between candidate energies — loosen it if good peaks go unmatched.
```

Both helpers return the matched `channels` and `energies` arrays, which feed
straight into `EnergyCalibration` to build the final, applicable object:

```python
cal = EnergyCalibration(best["channels"], best["energies"], spect.channels, n=best["n"])
spect.apply_calibration(cal)
```

## Piecewise linear calibration

Some detectors (or dual-range digitizers) follow two different slopes that a
single polynomial can't capture cleanly. `PiecewiseLinearCalibration` fits two
linear segments that join continuously at an energy breakpoint `e_break`:

```python
from wara.energy_calibration import PiecewiseLinearCalibration

cal = PiecewiseLinearCalibration(channels, energies, spect.channels, e_break=3000.0)
cal.plot()
cal.channel_to_energy(1500)   # convert a channel (or array) directly
spect.apply_calibration(cal)
```

The lower segment (`E < e_break`) is fitted freely; the upper segment is forced
through the breakpoint so the curve is continuous. Each segment needs at least
two calibration points, and `r2_lower` / `r2_upper` report each one's fit
quality.

## In the GUI

The **Calibration** tab in the navigation rail wraps this workflow: send fitted
peak centroids over from the Spectrum tab, enter or look up their energies (the
built-in nuclear database can supply reference lines), and the tab runs the
smart matching and polynomial fit, then applies the result to the loaded
spectrum.

## Related calibrations

Energy calibration is the mapping from channel to energy. Two other detector
characterizations live in their own modules and GUI tabs:

- [**Efficiency Calibration**](efficiency.md) — detection efficiency vs. energy.
- [**Resolution (FWHM vs. Energy)**](resolution.md) — peak width vs. energy.

For the full API, see {py:class}`wara.energy_calibration.EnergyCalibration` and
{py:class}`wara.energy_calibration.PiecewiseLinearCalibration` in the API
reference.
