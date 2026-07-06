"""LRO LOLA global topography (Gridded Data Record DEMs).

The Lunar Orbiter Laser Altimeter (LOLA, on Lunar Reconnaissance Orbiter)
global cylindrical DEMs from the PDS Geosciences node:

    https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/

``ldem_<ppd>.img`` is a raw little-endian int16 raster at ``ppd`` pixels per
degree (4 ppd = 1440x720 ~2 MB, 16 ppd = 5760x2880 ~33 MB, ...), row 0 at
+90 deg latitude, column 0 at 0 deg east longitude. Height in meters =
DN * 0.5 relative to the 1737.4 km reference sphere (per the .lbl label,
which is parsed rather than trusted blindly).

Used by the Planetary tab to (a) drape an elevation colormap over the Moon
and (b) displace the globe mesh radially — real 3D relief that the elemental
abundance maps of :mod:`wara.planetary.abundance` can be draped onto, so
topography and composition can be correlated visually.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from .lp import LP_DATA_DIR, _USER_AGENT

LOLA_BASE_URL = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/"
    "data/lola_gdr/cylindrical/img/"
)

# Global cylindrical DEMs offered (pixels per degree -> approx download size).
LOLA_RESOLUTIONS = (4, 16)


def lola_filename(ppd=4, ext="img"):
    if ppd not in LOLA_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {LOLA_RESOLUTIONS} "
                         f"pixels/degree")
    return f"ldem_{ppd}.{ext}"


def download_lola_dem(ppd=4, data_dir=LP_DATA_DIR, timeout=120.0):
    """Fetch one LOLA DEM (.img raster + .lbl label) into ``data_dir``.
    Cached: nothing is re-downloaded. Returns the local .img path."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("lbl", "img"):
        dest = data_dir / lola_filename(ppd, ext)
        if dest.exists():
            continue
        req = Request(LOLA_BASE_URL + dest.name,
                      headers={"User-Agent": _USER_AGENT})
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
    return data_dir / lola_filename(ppd, "img")


def _label_value(text, key):
    m = re.search(rf"^\s*{key}\s*=\s*([^\s<]+)", text, re.MULTILINE)
    if m is None:
        raise ValueError(f"key {key!r} not found in LOLA label")
    return m.group(1)


def read_lola_dem(ppd=4, data_dir=LP_DATA_DIR):
    """Read a downloaded LOLA DEM into ``{"elev_km": 2D array, "ppd": ppd}``.

    ``elev_km`` has shape ``(180*ppd, 360*ppd)`` with row 0 at +90 latitude
    and column 0 at 0 east longitude; values are elevation in km relative to
    the 1737.4 km reference sphere. Raster geometry and scaling are taken
    from the .lbl label.
    """
    data_dir = Path(data_dir)
    label = (data_dir / lola_filename(ppd, "lbl")).read_text(errors="replace")
    lines = int(_label_value(label, "LINES"))
    samples = int(_label_value(label, "LINE_SAMPLES"))
    bits = int(_label_value(label, "SAMPLE_BITS"))
    stype = _label_value(label, "SAMPLE_TYPE")
    scale = float(_label_value(label, "SCALING_FACTOR"))
    if bits != 16 or stype != "LSB_INTEGER":
        raise ValueError(f"unexpected LOLA sample format: {bits}-bit {stype}")
    raw = np.fromfile(data_dir / lola_filename(ppd, "img"), dtype="<i2")
    if raw.size != lines * samples:
        raise ValueError(f"LOLA raster size mismatch: {raw.size} values for "
                         f"{lines}x{samples}")
    elev_km = raw.reshape(lines, samples).astype(float) * scale / 1000.0
    return {"elev_km": elev_km, "ppd": ppd}


def elevation_grid(dem, lon_axis, lat_axis):
    """Sample a LOLA DEM onto a regular lon/lat grid.

    ``lon_axis`` (degrees east in [-180, 180]) and ``lat_axis`` ([-90, 90]),
    both ascending — e.g. the globe's :func:`~wara.planetary.moon.sphere_mesh`
    axes. Returns elevation in km, shape ``(len(lat_axis), len(lon_axis))``.
    """
    elev = dem["elev_km"]
    h, w = elev.shape
    lon = np.asarray(lon_axis, dtype=float) % 360.0        # raster starts at 0E
    lat = np.asarray(lat_axis, dtype=float)
    col = np.clip((lon / 360.0 * w).astype(int), 0, w - 1)
    row = np.clip(((90.0 - lat) / 180.0 * h).astype(int), 0, h - 1)
    return elev[np.ix_(row, col)]
