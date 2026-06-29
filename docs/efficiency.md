# Efficiency Calibration

Detector efficiency is the fraction of emitted gamma rays that are actually
recorded in a full-energy peak. wara measures it from a calibration source of
known activity with {py:mod}`wara.efficiency`: for each reference line you
compare the **detected** counts (a fitted peak area) against the **emitted**
counts predicted from the source's decay, then fit efficiency vs. energy.

## How it works

For a single line, the number of gamma rays emitted into the spectrum during the
measurement is

```
N_emitted = A0 · exp(-λ · t_elapsed) · Br · livetime,   λ = ln2 / t_half
```

where `A0` is the source activity at its reference date, `t_elapsed` is the time
since that date, `Br` the branching ratio of the line, and `livetime` the
acquisition live time. The detected count is the net peak area from a fit, and

```
efficiency = N_detected / N_emitted
```

## Measuring efficiency at one line

First fit the calibration peak (see [Peak Fitting](peak_fitting.md)) so you have
its area, then build an `Efficiency` object with the source parameters in SI
units (seconds, becquerel):

```python
from wara.efficiency import Efficiency, calculate_t_elapsed

# seconds between the source's reference date and the measurement date
t_elapsed = calculate_t_elapsed("2015-01-01", "2026-06-28")

eff = Efficiency(
    t_half=8.04e7,     # half-life [s]
    A0=37000,          # activity at the reference date [Bq]
    Br=0.851,          # branching ratio of this line
    livetime=3600,     # acquisition live time [s]
    t_elapsed=t_elapsed,
    which_peak=0,      # which peak of the fit to use
)

eff.calculate_efficiency(fit)     # sets eff.eff (a fraction)
eff.calculate_error(fit, t_half_sig=1e5, A0_sig=370,
                    Br_sig=0.002, livetime_sig=1, t_elapsed_sig=0)
print(eff.eff, "±", eff.error)
eff.to_df()                        # one-row DataFrame of inputs + result
```

```{note}
`eff.eff` is a fraction (detected / emitted). The plotting and table helpers
below display it as a percentage, so multiply by 100 when collecting points for
them. `calculate_error` propagates the uncertainties of every input
(half-life, activity, branching ratio, live time, elapsed time) together with
the peak-area error in quadrature.
```

## Fitting the efficiency curve

Repeat the single-line measurement for every calibration peak to build matched
lists of energy, efficiency, and uncertainty, then fit and plot the curve:

```python
from wara.efficiency import eff_fit, plot_points

energies = [121.8, 244.7, 344.3, 778.9, 1408.0]   # keV
effs     = [4.1, 2.7, 2.1, 1.1, 0.7]              # percent
errs     = [0.2, 0.1, 0.1, 0.05, 0.04]

plot_points(energies, effs, errs)         # data points with error bars
eff_fit(energies, effs, errs, order=2)    # fit + residual panel
```

The `order` argument selects the model:

| `order` | Model |
|---------|-------|
| `1` | Straight line in efficiency vs. energy |
| `2` | `eff = (a0 + a1·ln(E) + a2·ln(E)² + a3·ln(E)³) / E` — the standard log-polynomial HPGe efficiency curve |

`eff_table(energies, effs)` renders the points as a formatted table.

## In the GUI

The **Efficiency** tab in the navigation rail wraps this workflow: send fitted
peaks over from the Spectrum tab, enter each source's half-life, activity,
branching ratio and dates, and the tab computes the per-line efficiencies and
fits the curve.

For the full API, see {py:mod}`wara.efficiency` in the API reference.
