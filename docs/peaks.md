# Peak Finding and Fitting

## Peak search

`PeakSearch` locates peaks in a spectrum given the detector resolution (FWHM)
and a minimum signal-to-noise ratio.

```python
from wara import Spectrum, PeakSearch

sp = Spectrum.from_file("data.csv")
ps = PeakSearch(sp, fwhm=10, snr=3)
ps.find_peaks()
ps.plot()
```

## Peak fitting

Once peaks are found, `PeakFit` (and `AdvancedFit` for overlapping peaks)
decompose the spectrum into Gaussian components on top of a background model.

```python
from wara import PeakFit

pf = PeakFit(sp, ps.peaks)
pf.fit()
pf.plot()
print(pf.results)
```

For full API details see the API reference for `wara.peaksearch` and `wara.peakfit`.
