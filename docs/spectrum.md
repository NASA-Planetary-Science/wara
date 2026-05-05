# Spectrum

The `Spectrum` class is the core data container in wara. It holds counts,
energy bins, and metadata, and provides methods for smoothing, rebinning,
and plotting.

## Loading a spectrum

```python
from wara import Spectrum

sp = Spectrum.from_file("data.csv")
```

## Smoothing and rebinning

```python
sp_smooth = sp.smooth(window=5)
sp_rebin  = sp.rebin(factor=2)
```

## Plotting

```python
sp.plot()
```

For full API details see {py:class}`wara.spectrum.Spectrum` in the API reference.
