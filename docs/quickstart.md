# Quickstart

## Launching the GUI

After installation, launch the GUI from any directory:

```bash
wara -o
```

Then:

1. Select **File → Open → examples → data → test_data_cebr.csv**
2. Click **Find peaks**, select **LaBr/CeBr**, and hit **Apply**
3. Drag the mouse over one or more identified lines to perform a Gaussian fit

Info buttons throughout the GUI provide contextual guidance.

## Using wara in Python

```python
import wara

# Load a spectrum from a CSV file
sp = wara.Spectrum.from_file("my_spectrum.csv")

# Search for peaks
ps = wara.PeakSearch(sp, fwhm=10, snr=3)
ps.find_peaks()

# Fit the peaks
pf = wara.PeakFit(sp, ps.peaks)
pf.fit()
pf.plot()
```
