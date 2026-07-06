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
import time
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

# Default home for downloaded LP-GRS products (gitignored) and for the bundled
# orbit-metadata CSV (tracked in git / shipped as package data).
LP_DATA_DIR = Path(__file__).resolve().parent / "data"
LP_METADATA_CSV = LP_DATA_DIR / "lp_grs_metadata.csv"

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


# ── Mission documentation ────────────────────────────────────────────────────
# The PDS archive ships extensive prose documentation (PDS3 catalog objects
# and summary documents). These render well as plain 80-column text, so the
# GUI's "Mission info" dialog shows them verbatim.
LP_MISSION_PAGE_URL = (
    "https://pds-geosciences.wustl.edu/missions/lunarp/reduced_grsns.html"
)

_LP_ARCHIVE_ROOT = (
    "https://pds-geosciences.wustl.edu/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/"
)
_LP_ABUNDANCE_ROOT = (
    "https://pds-geosciences.wustl.edu/lunar/lp-l-grs-5-elem-abundance-v1/"
    "lp_9001/"
)

# Display label -> document URL, in menu order.
LP_DOCUMENTS = {
    "Archive overview (AAREADME)": _LP_ARCHIVE_ROOT + "aareadme.txt",
    "Mission description": _LP_ARCHIVE_ROOT + "catalog/mission.cat",
    "Spacecraft description": _LP_ARCHIVE_ROOT + "catalog/lphost.cat",
    "GRS instrument": _LP_ARCHIVE_ROOT + "catalog/grsinst.cat",
    "NS instrument": _LP_ARCHIVE_ROOT + "catalog/nsinst.cat",
    "GRS data set": _LP_ARCHIVE_ROOT + "catalog/grsds.cat",
    "NS data set": _LP_ARCHIVE_ROOT + "catalog/nsds.cat",
    "Data products summary": _LP_ARCHIVE_ROOT + "document/lp_grns_summary.txt",
    "Abundance data set (Level 5)": _LP_ABUNDANCE_ROOT + "aareadme.txt",
    "References": _LP_ARCHIVE_ROOT + "catalog/ref.cat",
}


def fetch_document(label, data_dir=LP_DATA_DIR, timeout=60.0):
    """Return the text of one :data:`LP_DOCUMENTS` entry.

    Downloaded once into ``data_dir/docs/`` and read from there afterwards,
    so the Mission-info dialog works offline after the first open.
    """
    url = LP_DOCUMENTS[label]
    docs_dir = Path(data_dir) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    # Abundance aareadme shares its basename with the archive one — prefix
    # with a slug of the label to keep cache files unique.
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    dest = docs_dir / f"{slug}.txt"
    if not dest.exists():
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        # PDS STREAM documents end lines with CR CR LF; drop every CR so the
        # text doesn't render double-spaced.
        dest.write_text(text.replace("\r", ""), encoding="utf-8")
    return dest.read_text(encoding="utf-8").replace("\r", "")


# ── Orbit metadata (bundled CSV) ─────────────────────────────────────────────
# One-time scrape of the whole mission's ephemeris into a small CSV so the GUI
# can show what data exists — and draw the orbit path — without downloading or
# reading any product. Default stride 15 (~8 min cadence): the full mission
# (~1.35M records over 561 days) compresses to ~90k rows / a few MB.
METADATA_STRIDE = 15

_METADATA_HEADER = """\
# Lunar Prospector GRS orbit metadata, scraped once from the PDS archive at
# {base_url}
# One row per ~{stride} records (~{minutes:.0f} min): sub-spacecraft ephemeris of every
# daily product. Products before 1998-12-19 fly the ~100 km mapping orbit;
# later ones the ~30-40 km extended mission. Longitude is planetocentric
# east-positive in [-180, 180).
utc,product,lon_east_deg,lat_deg,alt_km
"""


def records_utc64(day_obj):
    """Per-record UTC timestamps of an :class:`LPGrsDay` (datetime64[s]).

    ``Earth_Received_Time`` is a fractional day-of-year — but the extended-
    mission products keep counting past 1998-12-31 (mission-continuous DOY:
    a 1999 product may carry values around 366+its real DOY). Re-anchor each
    product to its own date, so the year-scale offset cancels whichever
    convention the file uses.
    """
    t64 = (np.datetime64(f"{day_obj.day.year}-01-01")
           + np.round((day_obj.time_doy - 1.0) * 86400.0).astype("timedelta64[s]"))
    off_days = (np.median(t64 - np.datetime64(day_obj.day))
                / np.timedelta64(1, "D"))
    shift = int(round(off_days / 365.0)) * 365
    if shift:
        t64 = t64 - np.timedelta64(shift, "D")
    return t64


def _metadata_rows(day_obj, stride=METADATA_STRIDE):
    """CSV rows (no newline) for one product, subsampled by ``stride``."""
    t64 = records_utc64(day_obj)
    rows = []
    for i in range(0, day_obj.n_records, stride):
        ts = np.datetime_as_string(t64[i], unit="m")
        rows.append(f"{ts},{day_obj.stem},{day_obj.longitude[i]:.2f},"
                    f"{day_obj.latitude[i]:.2f},{day_obj.altitude_km[i]:.1f}")
    return rows


def products_in_metadata(csv_path=LP_METADATA_CSV):
    """Set of product stems already present in the metadata CSV (for resume)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    stems = set()
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("utc,"):
                continue
            parts = line.split(",")
            if len(parts) == 5:
                stems.add(parts[1])
    return stems


def build_orbit_metadata(csv_path=LP_METADATA_CSV, data_dir=LP_DATA_DIR,
                         stride=METADATA_STRIDE, keep_products=False,
                         products=None, progress=None):
    """Scrape the whole-mission orbit metadata into ``csv_path``.

    Downloads every LP-GRS product (unless already in ``data_dir``), extracts
    the subsampled ephemeris, and appends it to the CSV. Idempotent and
    resumable: products already in the CSV are skipped, so an interrupted run
    just continues where it left off. Products fetched only for scraping are
    deleted afterwards unless ``keep_products`` (pre-existing files in
    ``data_dir`` are always left alone).

    ``progress(msg)`` is called per product. Returns the number of products
    added in this run.
    """
    csv_path, data_dir = Path(csv_path), Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if products is None:
        products = list_grs_products()
    done = products_in_metadata(csv_path)

    if not csv_path.exists():
        csv_path.write_text(_METADATA_HEADER.format(
            base_url=LP_GRS_BASE_URL, stride=stride,
            minutes=stride * 32.0 / 60.0), encoding="utf-8")

    n_added = 0
    for i, p in enumerate(products):
        if p.stem in done:
            continue
        if progress is not None:
            progress(f"{p.stem} ({i + 1}/{len(products)})")
        xml = data_dir / f"{p.stem}.xml"
        dat = data_dir / f"{p.stem}.dat"
        fetched = not (xml.exists() and dat.exists())
        if fetched:
            download_products([p], data_dir)
        try:
            rows = _metadata_rows(read_grs_day(xml), stride=stride)
        finally:
            if fetched and not keep_products:
                for f in (xml, dat):
                    try:
                        f.unlink()
                    except OSError:
                        pass
        # Windows: a concurrent reader (tests, editors, AV scanners) can hold
        # the file and briefly deny the append — retry instead of losing a
        # multi-hundred-product run to a transient lock.
        for attempt in range(5):
            try:
                with open(csv_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(rows) + "\n")
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(1.0 + attempt)
        n_added += 1
    return n_added


def load_orbit_metadata(csv_path=LP_METADATA_CSV):
    """Read the bundled orbit-metadata CSV.

    Returns a dict of aligned arrays: ``utc`` (datetime64[s]), ``product``
    (str), ``lon`` , ``lat``, ``alt_km`` (float), sorted by time. Raises
    ``FileNotFoundError`` if the CSV has not been built/shipped.
    """
    csv_path = Path(csv_path)
    utc, product, lon, lat, alt = [], [], [], [], []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("utc,"):
                continue
            ts, stem, lo, la, al = line.split(",")
            utc.append(np.datetime64(ts))
            product.append(stem)
            lon.append(float(lo))
            lat.append(float(la))
            alt.append(float(al))
    order = np.argsort(np.asarray(utc))
    return {
        "utc": np.asarray(utc, dtype="datetime64[s]")[order],
        "product": np.asarray(product)[order],
        "lon": np.asarray(lon)[order],
        "lat": np.asarray(lat)[order],
        "alt_km": np.asarray(alt)[order],
    }
