# Calibration

## Energy calibration

wara supports polynomial energy calibration by mapping known gamma-ray
peaks to their literature energies.

The energy calibration workflow is available both through the GUI
(**Calibration → Energy calibration**) and programmatically via
`wara.energy_calibration`.

## Efficiency calibration

Absolute and relative detector efficiency as a function of energy can be
fitted and stored using `wara.efficiency`.

## FWHM vs. energy

The detector resolution curve (FWHM as a function of energy) can be
characterised using `wara.resolution`.
