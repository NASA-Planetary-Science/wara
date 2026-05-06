# Installation

wara requires Python 3.8 or higher.

## From source

```bash
git clone https://github.com/NASA-Planetary-Science/wara.git
cd wara
pip install -e .
```

## Dependencies

wara installs the following dependencies automatically:

| Package | Purpose |
|---------|---------|
| `lmfit` | Peak fitting |
| `scipy` | Signal processing |
| `matplotlib` | Plotting |
| `pandas` | Data handling |
| `PyQt5` / `PyQtWebEngine` | GUI |
| `mplcursors` | Interactive plot cursors |
| `plotly` | 3D visualizations |
| `docopt` | CLI argument parsing |
| `dateparser` | Date parsing for LYNX data |
| `pyarrow` | Parquet file support |
| `natsort` | Natural sorting |
| `msgspec` / `bitstruct` | Binary data parsing |

## Data path configuration

If you use the API data-loading features, create a file called `data-path.txt`
in the repo root with one data folder path per line:

```
C:/Users/yourname/Documents/my-data
D:/external-drive/more-data
```

wara searches each path in order and uses the first one containing the requested run.
This file is excluded from version control so each user keeps their own local copy.
