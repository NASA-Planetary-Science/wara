# Installation

**wara** runs on Python 3.10 or higher.

## From source

wara is installed directly from its GitHub repository:

```bash
git clone https://github.com/NASA-Planetary-Science/wara.git
cd wara
pip install -e .
```

The `-e` (editable) install lets you pull updates with `git pull` without
reinstalling. Run the commands from the directory that contains
`pyproject.toml`.

```{tip}
Installing into a dedicated virtual environment keeps wara's dependencies
isolated from the rest of your system:

    python -m venv .venv
    # Windows:        .venv\Scripts\activate
    # macOS / Linux:  source .venv/bin/activate
    pip install -e .
```

## Verifying the installation

Once installed, the `wara` command is available on your `PATH`:

```bash
wara --help     # print the command-line usage
wara            # launch the GUI
```

If the GUI window opens, you are ready to go. Head to the
[Quickstart](quickstart.md) for a guided tour.

## Dependencies

wara installs the following packages automatically:

| Package | Purpose |
|---------|---------|
| `lmfit` | Peak fitting |
| `scipy` | Signal processing |
| `matplotlib` | Plotting |
| `pandas` | Data handling |
| `PyQtWebEngine` (pulls in `PyQt5`) | GUI |
| `mplcursors` | Interactive plot cursors |
| `plotly` | 3D visualizations |
| `docopt` | Command-line argument parsing |
| `dateparser` | Parsing run dates when loading API data |
| `pyarrow` | Parquet file support (API data) |
| `natsort` | Natural sorting of run files |
| `msgspec` / `bitstruct` | Binary list-mode data parsing |

```{note}
`PyQtWebEngine` provides the embedded browser used to render the interactive
Plotly 3D views, and it pulls in `PyQt5` as a requirement, so the full GUI
stack is installed for you.
```

## Data path configuration

If you use the API data-loading features, create a file called `data-path.txt`
in the repository root with one data folder path per line:

```
C:/Users/yourname/Documents/my-data
D:/external-drive/more-data
```

wara searches each path in order and uses the first one that contains the
requested run. This file is excluded from version control, so each user keeps
their own local copy.
