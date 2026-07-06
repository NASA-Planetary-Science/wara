"""LP-GRS Level-5 elemental abundance maps (the "calibrated" dataset).

The PDS volume ``lp-l-grs-5-elem-abundance-v1`` holds the derived elemental
composition of the lunar surface from the Lunar Prospector GRS (Lawrence et
al.), binned on an equal-area lat/lon grid at 2, 5, and 20 degree resolution:

    https://pds-geosciences.wustl.edu/lunar/lp-l-grs-5-elem-abundance-v1/lp_9001/data/

Each row of a ``.tab`` file is one map pixel: its lat/lon bounding box, the
mean atomic mass and neutron number density, the oxide weight fractions
(MgO, Al2O3, SiO2, CaO, TiO2, FeO), the K/Th/U abundances in ppm, and the
error-covariance terms (ignored here). Pixels are equal-area, so their
longitude width grows toward the poles (the polar rows span all longitudes).

This complements the raw Level-3 RDR spectra of :mod:`wara.planetary.lp`:
the GUI's dataset dropdown switches between the two.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from .lp import LP_DATA_DIR, _USER_AGENT

LP_ABUNDANCE_BASE_URL = (
    "https://pds-geosciences.wustl.edu/lunar/lp-l-grs-5-elem-abundance-v1/"
    "lp_9001/data/"
)

ABUNDANCE_RESOLUTIONS = (2, 5, 20)   # degrees (at the equator)

# Column order of the .tab files (per lpgrs_elem_abundance.fmt); the trailing
# error-covariance columns are not listed and not read.
_TAB_COLUMNS = ["pixel", "lat_s", "lat_n", "lon_w", "lon_e",
                "am", "neutron_den",
                "MgO", "Al2O3", "SiO2", "CaO", "TiO2", "FeO", "K", "Th", "U"]

# element key -> (pretty label, unit) for GUI display.
ABUNDANCE_ELEMENTS = {
    "Th": ("Th", "ppm"),
    "K": ("K", "ppm"),
    "U": ("U", "ppm"),
    "FeO": ("FeO", "wt. fraction"),
    "TiO2": ("TiO2", "wt. fraction"),
    "MgO": ("MgO", "wt. fraction"),
    "Al2O3": ("Al2O3", "wt. fraction"),
    "SiO2": ("SiO2", "wt. fraction"),
    "CaO": ("CaO", "wt. fraction"),
}


def abundance_filename(deg=2):
    if deg not in ABUNDANCE_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {ABUNDANCE_RESOLUTIONS}")
    return f"lpgrs_high1_elem_abundance_{deg}deg.tab"


def download_abundance(deg=2, data_dir=LP_DATA_DIR, timeout=60.0):
    """Fetch one abundance table into ``data_dir`` (skipped when cached).
    The files are small (the 2-degree map is ~4 MB). Returns the local path."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / abundance_filename(deg)
    if dest.exists():
        return dest
    req = Request(LP_ABUNDANCE_BASE_URL + dest.name,
                  headers={"User-Agent": _USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    tmp.replace(dest)
    return dest


def read_abundance(deg=2, data_dir=LP_DATA_DIR, path=None):
    """Read an abundance table into a dict of aligned arrays.

    Keys: ``lat_s``/``lat_n``/``lon_w``/``lon_e`` (pixel bounds, degrees) and
    one array per element in :data:`ABUNDANCE_ELEMENTS`. The file must have
    been downloaded first (see :func:`download_abundance`).
    """
    path = Path(path) if path is not None else Path(data_dir) / abundance_filename(deg)
    raw = np.loadtxt(path, usecols=range(len(_TAB_COLUMNS)))
    table = {name: raw[:, i] for i, name in enumerate(_TAB_COLUMNS)
             if name != "pixel"}
    return table


def abundance_grid(table, element, lon_axis, lat_axis):
    """Sample an abundance map onto a regular lon/lat grid.

    ``lon_axis`` (ascending, degrees east in [-180, 180]) and ``lat_axis``
    (ascending, [-90, 90]) are the 1-D axes of the target grid (e.g. the
    globe's :func:`~wara.planetary.moon.sphere_mesh` axes). Every pixel row of
    ``table`` paints its value into the grid cells inside its bounding box.
    Returns a ``(len(lat_axis), len(lon_axis))`` float array.
    """
    if element not in ABUNDANCE_ELEMENTS:
        raise ValueError(f"unknown element {element!r}; "
                         f"pick from {list(ABUNDANCE_ELEMENTS)}")
    lon_axis = np.asarray(lon_axis, dtype=float)
    lat_axis = np.asarray(lat_axis, dtype=float)
    grid = np.full((len(lat_axis), len(lon_axis)), np.nan)
    values = table[element]
    # Each pixel is an axis-aligned box: paint by slice. Upper edges are
    # inclusive so the grid's +90 lat / +180 lon boundary points get a value.
    for lat_s, lat_n, lon_w, lon_e, val in zip(
            table["lat_s"], table["lat_n"], table["lon_w"], table["lon_e"],
            values):
        i0 = np.searchsorted(lat_axis, lat_s, side="left")
        i1 = np.searchsorted(lat_axis, lat_n, side="right")
        j0 = np.searchsorted(lon_axis, lon_w, side="left")
        j1 = np.searchsorted(lon_axis, lon_e, side="right")
        grid[i0:i1, j0:j1] = val
    return grid
