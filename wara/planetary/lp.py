"""Lunar Prospector GRS data retrieval and reading from the NASA PDS.

Step 2 of the WARA Planetary tab: search, download, and read the reduced
Lunar Prospector Gamma-Ray Spectrometer (LP-GRS) products hosted at the PDS
Geosciences node:

    https://pds-geosciences.wustl.edu/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/

The archive holds **one product per mission day**, named ``YYYY_DOY_grs`` with
three files each: a ``.dat`` binary table, its PDS4 ``.xml`` label, and a
legacy PDS3 ``.lbl`` label. Each product contains one row per ~32 s
accumulation with the 512-channel accepted/rejected spectra **and** the
ephemeris/housekeeping needed for regional selection:

    Spacecraft_Altitude (km), Subspacecraft_Latitude, Subspacecraft_Longitude,
    Deadtime, Overload, GRS_Tempature, Earth_Received_Time (fractional DOY)

Because the ephemeris lives *inside* the data files, searching works in two
stages: **by date** at the archive level (the filename carries year/DOY, and
the mission date determines the ~100 km vs ~30 km orbit phase), then **by
(latitude, longitude, altitude, ...)** after reading, via
:meth:`LPGrsDay.select` / :meth:`LPGrsDay.sum_spectrum`.

Longitude convention: the raw products store east-positive 0..360 deg;
:func:`read_grs_day` wraps it to ``[-180, 180)`` to match
:mod:`wara.planetary.moon`.

Only the directory listing and download need the network; everything else is
offline. Reading requires ``pds4_tools`` (imported lazily).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import numpy as np

from .moon import wrap_lon

LP_GRS_BASE_URL = (
    "https://pds-geosciences.wustl.edu/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/"
)

# Mission phases by orbit altitude. LP mapped from a ~100 km circular orbit
# from insertion (first GRS product: 1998 DOY 016) until 1998-12-19, when the
# orbit was lowered for the extended mission (~40 km, then ~30 km from
# 1999-01-29 until impact on 1999-07-31). The per-record Spacecraft_Altitude
# in the data is authoritative; this date split is only a coarse search key.
LOW_ALTITUDE_START = date(1998, 12, 19)

_STEM_RE = re.compile(r"(\d{4})_(\d{3})_grs$", re.IGNORECASE)
_USER_AGENT = "wara-planetary-lp/1.0"


# ── Archive listing ──────────────────────────────────────────────────────────
class _LinkParser(HTMLParser):
    """Collect every <a href=...> from an HTML directory index."""

    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def _http_get_text(url, timeout=60.0):
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def doy_to_date(year, doy):
    """Convert (year, day-of-year) to a :class:`datetime.date`."""
    return date(year, 1, 1) + timedelta(days=int(doy) - 1)


@dataclass(frozen=True)
class LPGrsProduct:
    """One day of LP-GRS data in the PDS archive (not yet downloaded)."""

    stem: str            # e.g. "1998_016_grs"
    day: date            # UTC date of the measurements
    url_xml: str         # PDS4 label (what pds4_tools reads)
    url_dat: str         # binary data table
    url_lbl: str = ""    # legacy PDS3 label (optional)

    @property
    def phase(self):
        """Orbit phase by mission date: ``"high"`` (~100 km) or ``"low"`` (~30-40 km)."""
        return "low" if self.day >= LOW_ALTITUDE_START else "high"


def list_grs_products(base_url=LP_GRS_BASE_URL, timeout=60.0, html=None):
    """List every LP-GRS product in the PDS archive, sorted by date.

    Fetches the archive's HTML directory index (or parses ``html`` if given,
    which keeps tests offline) and pairs each ``YYYY_DOY_grs.xml`` label with
    its ``.dat`` / ``.lbl`` companions.

    Returns a list of :class:`LPGrsProduct`.
    """
    if html is None:
        html = _http_get_text(base_url, timeout=timeout)
    parser = _LinkParser()
    parser.feed(html)

    # Group hrefs by product stem; hrefs may be absolute paths or bare names.
    by_stem = {}
    for href in parser.hrefs:
        name = href.rsplit("/", 1)[-1]
        if "." not in name:
            continue
        stem, ext = name.rsplit(".", 1)
        if not _STEM_RE.match(stem):
            continue
        by_stem.setdefault(stem, {})[ext.lower()] = urljoin(base_url, href)

    products = []
    for stem, urls in by_stem.items():
        if "xml" not in urls or "dat" not in urls:
            continue  # need the PDS4 label and the data to be usable
        year, doy = map(int, _STEM_RE.match(stem).groups())
        products.append(LPGrsProduct(
            stem=stem,
            day=doy_to_date(year, doy),
            url_xml=urls["xml"],
            url_dat=urls["dat"],
            url_lbl=urls.get("lbl", ""),
        ))
    products.sort(key=lambda p: p.day)
    return products


def _as_date(value):
    """Accept a date, datetime, or 'YYYY-MM-DD' string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def filter_products(products, start=None, end=None, phase=None):
    """Filter products by measurement date and/or orbit phase.

    ``start`` / ``end`` are inclusive bounds (date objects or ``"YYYY-MM-DD"``
    strings); ``phase`` is ``"high"`` (~100 km) or ``"low"`` (~30-40 km).
    """
    start, end = _as_date(start), _as_date(end)
    if phase is not None and phase not in ("high", "low"):
        raise ValueError(f"phase must be 'high' or 'low', got {phase!r}")
    out = []
    for p in products:
        if start is not None and p.day < start:
            continue
        if end is not None and p.day > end:
            continue
        if phase is not None and p.phase != phase:
            continue
        out.append(p)
    return out


# ── Download ─────────────────────────────────────────────────────────────────
def _download_file(url, dest, timeout=60.0, chunk_size=65536):
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def download_products(products, dest_dir, exts=("xml", "dat"),
                      skip_existing=True, timeout=60.0, progress=None):
    """Download products into ``dest_dir``; return the local ``.xml`` paths.

    Each product needs its ``.xml`` label and ``.dat`` table side by side for
    ``pds4_tools`` to read it — the default ``exts`` fetches exactly that pair.
    Files already present are kept when ``skip_existing`` (safe to re-run).
    ``progress``, if given, is called as ``progress(filename, i, n_total)``
    before each file transfer.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for p in products:
        for ext in exts:
            url = getattr(p, f"url_{ext}", "")
            if url:
                jobs.append((url, dest_dir / f"{p.stem}.{ext}"))

    for i, (url, dest) in enumerate(jobs):
        if skip_existing and dest.exists():
            continue
        if progress is not None:
            progress(dest.name, i + 1, len(jobs))
        _download_file(url, dest, timeout=timeout)

    return [dest_dir / f"{p.stem}.xml" for p in products]


# ── Reading and regional selection ───────────────────────────────────────────
@dataclass
class LPGrsDay:
    """One day of LP-GRS records with normalized field names.

    All per-record arrays share the first dimension ``n_records``; the spectra
    are ``(n_records, 512)`` raw counts per ~32 s accumulation.
    """

    stem: str
    day: date
    spectra: np.ndarray        # accepted 512-channel spectra
    rejected: np.ndarray       # coincidence-rejected spectra
    deadtime: np.ndarray       # fractional deadtime
    overload: np.ndarray       # overload counter
    temperature: np.ndarray    # GRS detector temperature (deg C)
    time_doy: np.ndarray       # Earth-received time as fractional day-of-year
    altitude_km: np.ndarray    # spacecraft altitude above the surface
    latitude: np.ndarray       # sub-spacecraft latitude, deg
    longitude: np.ndarray      # sub-spacecraft longitude, deg east in [-180, 180)

    @property
    def n_records(self):
        return len(self.spectra)

    def select(self, lat_range=None, lon_range=None, alt_range=None,
               time_range=None):
        """Boolean mask of records inside the given (inclusive) ranges.

        ``lon_range=(lo, hi)`` uses degrees east in ``[-180, 180)`` and may
        wrap the +/-180 seam: ``(170, -170)`` selects the far-side strip.
        ``time_range`` is in fractional day-of-year, matching ``time_doy``.
        """
        mask = np.ones(self.n_records, dtype=bool)
        if lat_range is not None:
            lo, hi = lat_range
            mask &= (self.latitude >= lo) & (self.latitude <= hi)
        if lon_range is not None:
            lo, hi = wrap_lon(lon_range[0]), wrap_lon(lon_range[1])
            if lo <= hi:
                mask &= (self.longitude >= lo) & (self.longitude <= hi)
            else:  # range crosses the +/-180 seam
                mask &= (self.longitude >= lo) | (self.longitude <= hi)
        if alt_range is not None:
            lo, hi = alt_range
            mask &= (self.altitude_km >= lo) & (self.altitude_km <= hi)
        if time_range is not None:
            lo, hi = time_range
            mask &= (self.time_doy >= lo) & (self.time_doy <= hi)
        return mask

    def sum_spectrum(self, mask=None, rejected=False):
        """Sum spectra over the selected records.

        Returns ``(spectrum, n_summed)`` where ``spectrum`` is the 512-channel
        total of the accepted (or ``rejected``) spectra and ``n_summed`` is how
        many records went in. ``mask=None`` sums the whole day.
        """
        data = self.rejected if rejected else self.spectra
        if mask is None:
            return data.sum(axis=0), self.n_records
        return data[mask].sum(axis=0), int(np.count_nonzero(mask))


# Raw PDS4 field name -> LPGrsDay attribute. The typo in "GRS_Tempature" is
# the archive's, not ours.
_FIELD_MAP = {
    "GROUP_0, Accepted Spectrum": "spectra",
    "GROUP_1, Rejected Spectrum": "rejected",
    "Deadtime": "deadtime",
    "Overload": "overload",
    "GRS_Tempature": "temperature",
    "Earth_Received_Time": "time_doy",
    "Spacecraft_Altitude": "altitude_km",
    "Subspacecraft_Latitude": "latitude",
    "Subspacecraft_Longitude": "longitude",
}


def read_grs_day(xml_path):
    """Read one downloaded LP-GRS product into an :class:`LPGrsDay`.

    ``xml_path`` is the PDS4 ``.xml`` label; the matching ``.dat`` must sit
    next to it (as :func:`download_products` arranges). Longitude is wrapped
    from the archive's 0..360 east convention to ``[-180, 180)``.
    """
    try:
        import pds4_tools
    except ImportError as err:
        raise ImportError(
            "Reading LP-GRS products requires the 'pds4_tools' package "
            "(pip install pds4_tools)."
        ) from err

    xml_path = Path(xml_path)
    struct = pds4_tools.read(str(xml_path), lazy_load=False, quiet=True)
    data = struct[0].data

    fields = {}
    for raw_name, attr in _FIELD_MAP.items():
        if raw_name not in data.dtype.names:
            raise ValueError(
                f"{xml_path.name}: expected field {raw_name!r} not found; "
                f"available: {data.dtype.names}"
            )
        fields[attr] = np.asarray(data[raw_name])
    fields["longitude"] = wrap_lon(fields["longitude"])

    m = _STEM_RE.match(xml_path.stem)
    day = doy_to_date(*map(int, m.groups())) if m else None
    return LPGrsDay(stem=xml_path.stem, day=day, **fields)
