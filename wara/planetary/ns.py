"""Lunar Prospector Neutron Spectrometer (LP-NS) time series from the NASA PDS.

The neutron data lives in the *same* PDS bundle as the LP-GRS spectra of
:mod:`wara.planetary.lp`, in the sibling ``ns/`` collection:

    https://pds-geosciences.wustl.edu/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/ns/

Structurally it is nothing like the GRS half of the bundle:

* **Whole-mission arrays, not one product per day.** One file holds every
  accumulation of one neutron type for one orbit phase and one sampling
  period (e.g. ``epitherm_neutron_high32sec.dat``, 542,686 samples).
* **The ephemeris is a separate file.** ``position_<phase><cadence>sec.dat``
  carries the lat/lon/altitude/time arrays that pair index-by-index with the
  counts. The 32 s position files hold four arrays; the **8 s ones hold only
  latitude and longitude** — no altitude, no time.
* **Moderated neutrons are self-contained**: that one file carries its own
  position arrays, with a different sample count than ``position_low32sec``,
  so it must never be joined to the shared position file.

The binary format is a bare little-endian dump — a ``uint32`` sample count
followed by back-to-back ``float32`` arrays — so unlike the GRS products this
needs no ``pds4_tools``, only numpy.

On top of the archived products, :data:`NS_RATIOS` defines derived ones -
currently the thermal/epithermal ratio, computed per accumulation from the two
component files (they share a position file, so the pairing is exact).

Counts are the archive's fully corrected accumulations (Lawrence et al. 2003,
Maurice et al. 2003); no deadtime or livetime correction is applied here.
Longitude is wrapped from the archive's 0..360 east convention to
``[-180, 180)`` to match :mod:`wara.planetary.moon`, and ``Earth_Received_Time``
is a *mission-continuous* fractional day-of-year counted from 1998-01-01 (the
extended-mission files run past 365), which :attr:`LPNsData.utc64` converts.

Only :func:`download_ns` needs the network; everything else is offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .lp import LP_DATA_DIR, _download_file
from .moon import wrap_lon

NS_BASE_URL = (
    "https://pds-geosciences.wustl.edu/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/ns/"
)

# Neutron type -> (file prefix, pretty label). The archive abbreviates
# epithermal to "epitherm"; the others spell out.
NS_KINDS = {
    "epithermal": ("epitherm", "Epithermal neutrons"),
    "thermal": ("thermal", "Thermal neutrons"),
    "fast": ("fast", "Fast neutrons"),
    "moderated": ("moderated", "Moderated neutrons"),
}

# Orbit phase -> (pretty label, nominal altitude). Matches lp.LOW_ALTITUDE_START:
# LP mapped from ~100 km until 1998-12-19, then dropped for the extended mission.
NS_PHASES = {
    "high": ("High orbit (~100 km)", date(1998, 1, 16), date(1999, 1, 16)),
    "low": ("Low orbit (~30-40 km)", date(1999, 1, 16), date(1999, 7, 31)),
}

NS_CADENCES = (8, 32)      # seconds per accumulation

# Derived products: a ratio of two archived count arrays. Thermal and
# epithermal counts of the same orbit phase and cadence are sampled on the
# *same* accumulations (they share one position file), so the ratio is exact
# per sample - no re-binning or interpolation. The thermal/epithermal ratio is
# the standard way to separate hydrogen (which moderates epithermal neutrons)
# from the neutron absorbers (Fe, Ti, Gd, Sm) that drive the thermal counts.
NS_RATIOS = {
    "thermal/epithermal": ("thermal", "epithermal",
                           "Thermal / epithermal ratio"),
}

# Display label -> catalog kind, in menu order (what the GUI lists).
NS_MENU = {
    "Epithermal": "epithermal",
    "Thermal": "thermal",
    "Fast": "fast",
    "Moderated": "moderated",
    "Thermal / epithermal": "thermal/epithermal",
}

# Sample counts declared by the PDS4 labels, used to validate what we read.
# Keyed the way the files are: (phase, cadence) for the shared arrays, and
# separately for the self-contained moderated product.
_N_SAMPLES = {
    ("high", 32): 542686,
    ("high", 8): 2170744,
    ("low", 32): 442572,
    ("low", 8): 1770288,
}
_N_MODERATED = 429608


@dataclass(frozen=True)
class NSProduct:
    """One LP-NS product: a neutron type at one orbit phase and cadence."""

    kind: str            # "epithermal" | "thermal" | "fast" | "moderated"
                         # or a derived kind such as "thermal/epithermal"
    phase: str           # "high" | "low"
    cadence: int         # 8 or 32 seconds
    n_samples: int
    self_positioned: bool = False   # carries its own lat/lon/alt/time arrays
    components: tuple = ()          # derived products: the kinds it divides

    @property
    def is_ratio(self):
        return bool(self.components)

    @property
    def counts_name(self):
        """The archived counts file. Derived products have none - they are
        computed from their :attr:`components`."""
        if self.is_ratio:
            raise ValueError(f"{self.kind} is derived from "
                             f"{' / '.join(self.components)}, not archived")
        prefix = NS_KINDS[self.kind][0]
        return f"{prefix}_neutron_{self.phase}{self.cadence}sec.dat"

    @property
    def position_name(self):
        """Companion ephemeris file, or ``None`` when self-positioned."""
        if self.self_positioned:
            return None
        return f"position_{self.phase}{self.cadence}sec.dat"

    @property
    def files(self):
        """Every file this product needs, counts first (both numerator and
        denominator for a ratio)."""
        if self.is_ratio:
            names = [ns_product(k, self.phase, self.cadence).counts_name
                     for k in self.components]
        else:
            names = [self.counts_name]
        if self.position_name:
            names.append(self.position_name)
        return names

    @property
    def pretty_kind(self):
        if self.is_ratio:
            return NS_RATIOS[self.kind][2]
        return NS_KINDS[self.kind][1]

    @property
    def label(self):
        return (f"{self.pretty_kind}, {NS_PHASES[self.phase][0]}, "
                f"{self.cadence} s")

    @property
    def value_label(self):
        """What one binned value means - the colorbar / y-axis label."""
        if self.is_ratio:
            return " / ".join(self.components) + " ratio"
        return f"counts / {self.cadence} s"

    @property
    def has_altitude(self):
        """The 8 s position files archive latitude and longitude only."""
        return self.cadence == 32

    has_time = has_altitude


def _build_catalog():
    """Every product the archive actually holds, keyed ``(kind, phase, cadence)``.

    Thermal and epithermal exist at both cadences and both orbit phases; fast
    neutrons only at 32 s; moderated neutrons only in the low orbit at 32 s.
    """
    out = {}
    for kind in ("epithermal", "thermal"):
        for phase in NS_PHASES:
            for cadence in NS_CADENCES:
                out[(kind, phase, cadence)] = NSProduct(
                    kind, phase, cadence, _N_SAMPLES[(phase, cadence)])
    for phase in NS_PHASES:
        out[("fast", phase, 32)] = NSProduct(
            "fast", phase, 32, _N_SAMPLES[(phase, 32)])
    out[("moderated", "low", 32)] = NSProduct(
        "moderated", "low", 32, _N_MODERATED, self_positioned=True)
    # Ratios exist wherever both of their components do.
    for kind, (num, den, _) in NS_RATIOS.items():
        for phase in NS_PHASES:
            for cadence in NS_CADENCES:
                if (num, phase, cadence) in out and (den, phase, cadence) in out:
                    out[(kind, phase, cadence)] = NSProduct(
                        kind, phase, cadence, _N_SAMPLES[(phase, cadence)],
                        components=(num, den))
    return out


NS_PRODUCTS = _build_catalog()


def ns_product(kind, phase, cadence=32):
    """Look up one :class:`NSProduct`, with a helpful error when it does not
    exist (fast neutrons at 8 s, moderated neutrons in the high orbit, ...)."""
    key = (kind, phase, int(cadence))
    if key in NS_PRODUCTS:
        return NS_PRODUCTS[key]
    if kind not in NS_KINDS and kind not in NS_RATIOS:
        raise ValueError(f"unknown neutron type {kind!r}; "
                         f"pick from {list(NS_KINDS) + list(NS_RATIOS)}")
    if phase not in NS_PHASES:
        raise ValueError(f"phase must be 'high' or 'low', got {phase!r}")
    have = sorted(f"{p.phase} {p.cadence} s" for p in NS_PRODUCTS.values()
                  if p.kind == kind)
    raise ValueError(f"the archive has no {kind} neutron data at {phase} "
                     f"orbit / {cadence} s; available: {', '.join(have)}")


# ── Download ─────────────────────────────────────────────────────────────────
def download_ns(kind, phase, cadence=32, data_dir=LP_DATA_DIR,
                skip_existing=True, timeout=60.0, progress=None):
    """Download one product's files into ``data_dir``; return their paths.

    Fetches the counts file and (unless the product carries its own ephemeris)
    the matching ``position_*`` file — 2-17 MB each. Files already present are
    kept when ``skip_existing``, so this is safe to re-run. ``progress``, if
    given, is called as ``progress(filename, i, n_total)`` before each
    transfer.
    """
    product = ns_product(kind, phase, cadence)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    names = product.files
    for i, name in enumerate(names):
        dest = data_dir / name
        paths.append(dest)
        if skip_existing and dest.exists():
            continue
        if progress is not None:
            progress(name, i + 1, len(names))
        _download_file(NS_BASE_URL + name, dest, timeout=timeout)
    return paths


# ── Reading ──────────────────────────────────────────────────────────────────
def _read_binary(path, n_arrays, expect_samples=None):
    """Read the archive's ``uint32`` count + ``n_arrays`` float32 arrays.

    Returns an ``(n_arrays, n_samples)`` array. The declared sample count is
    cross-checked against the file's actual length (and against
    ``expect_samples`` from the PDS4 label when given), so a truncated
    download fails loudly instead of silently mis-pairing counts with
    positions.
    """
    path = Path(path)
    header = np.fromfile(path, dtype="<u4", count=1)
    if not len(header):
        raise ValueError(f"{path.name}: empty file")
    n = int(header[0])
    values = np.fromfile(path, dtype="<f4", offset=4)
    if values.size != n * n_arrays:
        raise ValueError(
            f"{path.name}: header declares {n} samples x {n_arrays} array(s) "
            f"= {n * n_arrays} floats, file holds {values.size} "
            "(truncated or partially downloaded?)")
    if expect_samples is not None and n != expect_samples:
        raise ValueError(f"{path.name}: expected {expect_samples} samples "
                         f"per the PDS label, file declares {n}")
    return values.reshape(n_arrays, n)


@dataclass
class LPNsData:
    """One LP-NS product in memory, counts paired with their ephemeris.

    Every array shares the first dimension ``n_samples``. ``altitude_km`` and
    ``time_doy`` are ``None`` for the 8 s cadence, which the archive ships
    without them.
    """

    product: NSProduct
    counts: np.ndarray            # corrected counts per accumulation
    latitude: np.ndarray          # sub-spacecraft latitude, deg
    longitude: np.ndarray         # deg east in [-180, 180)
    altitude_km: np.ndarray | None = None
    time_doy: np.ndarray | None = None   # continuous DOY from 1998-01-01

    @property
    def n_samples(self):
        return len(self.counts)

    @property
    def kind(self):
        return self.product.kind

    @property
    def label(self):
        return self.product.label

    @property
    def utc64(self):
        """Per-sample UTC timestamps (``datetime64[s]``), or ``None`` at 8 s.

        ``Earth_Received_Time`` counts mission-continuous days from
        1998-01-01, so the extended-mission files carry values past 365 and a
        single anchor converts every phase.
        """
        if self.time_doy is None:
            return None
        return (np.datetime64("1998-01-01")
                + np.round((self.time_doy - 1.0) * 86400.0)
                .astype("timedelta64[s]"))

    def select(self, lat_range=None, lon_range=None, alt_range=None,
               time_range=None):
        """Boolean mask of samples inside the given (inclusive) ranges.

        Mirrors :meth:`wara.planetary.lp.LPGrsDay.select`: ``lon_range`` is in
        degrees east in ``[-180, 180)`` and may wrap the +/-180 seam, and
        ``time_range`` is in the same continuous day-of-year as
        :attr:`time_doy`. Asking for an altitude or time cut on an 8 s product
        raises, since the archive does not ship those arrays.
        """
        mask = np.ones(self.n_samples, dtype=bool)
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
            if self.altitude_km is None:
                raise ValueError(
                    f"{self.product.cadence} s products carry no altitude "
                    "array (the archive ships it only at 32 s)")
            lo, hi = alt_range
            mask &= (self.altitude_km >= lo) & (self.altitude_km <= hi)
        if time_range is not None:
            if self.time_doy is None:
                raise ValueError(
                    f"{self.product.cadence} s products carry no time array "
                    "(the archive ships it only at 32 s)")
            lo, hi = time_range
            mask &= (self.time_doy >= lo) & (self.time_doy <= hi)
        return mask


def read_ns(kind, phase, cadence=32, data_dir=LP_DATA_DIR,
            counts_path=None, position_path=None):
    """Read one downloaded LP-NS product into an :class:`LPNsData`.

    Files are looked up in ``data_dir`` by the archive's names (as
    :func:`download_ns` arranges); ``counts_path`` / ``position_path`` override
    that for tests or ad-hoc layouts.
    """
    product = ns_product(kind, phase, cadence)
    data_dir = Path(data_dir)
    if product.is_ratio:
        return _read_ratio(product, data_dir)
    counts_path = Path(counts_path or data_dir / product.counts_name)

    if product.self_positioned:
        # counts, latitude, longitude, altitude, ERT in one file.
        counts, lat, lon, alt, ert = _read_binary(
            counts_path, 5, product.n_samples)
    else:
        counts = _read_binary(counts_path, 1, product.n_samples)[0]
        position_path = Path(
            position_path or data_dir / product.position_name)
        n_arrays = 4 if product.has_altitude else 2
        arrays = _read_binary(position_path, n_arrays, product.n_samples)
        lat, lon = arrays[0], arrays[1]
        alt, ert = (arrays[2], arrays[3]) if n_arrays == 4 else (None, None)

    return LPNsData(
        product=product,
        counts=np.asarray(counts, dtype=float),
        latitude=np.asarray(lat, dtype=float),
        longitude=wrap_lon(np.asarray(lon, dtype=float)),
        altitude_km=None if alt is None else np.asarray(alt, dtype=float),
        time_doy=None if ert is None else np.asarray(ert, dtype=float),
    )


def _read_ratio(product, data_dir):
    """Read a derived ratio product: both components, divided per sample.

    The components share one position file and one set of accumulations, so
    the division is index-by-index. Samples where the denominator is zero come
    back ``NaN`` rather than infinite.
    """
    num, den = (read_ns(k, product.phase, product.cadence, data_dir=data_dir)
                for k in product.components)
    if num.n_samples != den.n_samples:
        raise ValueError(
            f"{product.components[0]} has {num.n_samples} samples but "
            f"{product.components[1]} has {den.n_samples}: they cannot be "
            "paired sample by sample")
    counts = np.divide(num.counts, den.counts,
                       out=np.full_like(num.counts, np.nan),
                       where=den.counts != 0)
    return LPNsData(
        product=product, counts=counts, latitude=num.latitude,
        longitude=num.longitude, altitude_km=num.altitude_km,
        time_doy=num.time_doy)


# ── Binning ──────────────────────────────────────────────────────────────────
NS_STATISTICS = {
    "mean": "Mean counts per accumulation",
    "samples": "Samples per bin (coverage)",
}


def _bin_edges(bin_deg):
    """Regular lon/lat cell edges spanning the whole Moon."""
    n_lon = int(round(360.0 / bin_deg))
    n_lat = int(round(180.0 / bin_deg))
    if n_lon < 1 or n_lat < 1:
        raise ValueError(f"bin_deg={bin_deg} is larger than the Moon")
    return (np.linspace(-180.0, 180.0, n_lon + 1),
            np.linspace(-90.0, 90.0, n_lat + 1))


def neutron_bins(data, bin_deg=2.0, statistic="mean", mask=None):
    """Bin the counts onto a regular ``bin_deg`` lat/lon grid.

    Returns ``(grid, lon_edges, lat_edges)`` where ``grid`` is
    ``(n_lat, n_lon)``: the mean counts of the samples falling in each cell
    (``statistic="mean"``, empty cells ``NaN``) or how many samples landed
    there (``statistic="samples"``).
    """
    if statistic not in NS_STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}; "
                         f"pick from {list(NS_STATISTICS)}")
    lon_edges, lat_edges = _bin_edges(bin_deg)
    lat, lon, counts = data.latitude, data.longitude, data.counts
    if mask is not None:
        lat, lon, counts = lat[mask], lon[mask], counts[mask]
    if statistic == "samples":
        n_all, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
        return n_all, lon_edges, lat_edges
    # A ratio carries NaN wherever its denominator was zero: drop those
    # samples instead of poisoning their whole cell.
    good = np.isfinite(counts)
    lat, lon, counts = lat[good], lon[good], counts[good]
    n_per_cell, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
    total, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges],
                                 weights=counts)
    grid = np.divide(total, n_per_cell, out=np.full_like(total, np.nan),
                     where=n_per_cell > 0)
    return grid, lon_edges, lat_edges


def neutron_map(data, lon_axis, lat_axis, bin_deg=2.0, statistic="mean",
                mask=None):
    """Sample a binned neutron map onto a regular lon/lat grid.

    Same contract as :func:`wara.planetary.abundance.abundance_grid` — the
    ``(len(lat_axis), len(lon_axis))`` array it returns drapes straight onto
    the globe mesh — but the values come from binning the time series at
    ``bin_deg`` first, so the map stays readable when the display mesh is
    finer than the along-track sampling.
    """
    grid, lon_edges, lat_edges = neutron_bins(
        data, bin_deg=bin_deg, statistic=statistic, mask=mask)
    # Nearest enclosing cell for every mesh point (blocky pixels, like the
    # abundance drape); the upper edge belongs to its own last cell.
    j = np.clip(np.searchsorted(lon_edges, np.asarray(lon_axis, float),
                                side="right") - 1, 0, grid.shape[1] - 1)
    i = np.clip(np.searchsorted(lat_edges, np.asarray(lat_axis, float),
                                side="right") - 1, 0, grid.shape[0] - 1)
    return grid[np.ix_(i, j)]


def zonal_profile(data, bin_deg=2.0, mask=None):
    """Mean counts per latitude band — the classic LP-NS view.

    Returns ``(lat_centers, mean, sem, n)``: bands with no samples come back
    ``NaN`` (``n = 0``). The polar dip in epithermal counts (neutrons
    moderated by hydrogen) is the feature this is meant to show.
    """
    lat_edges = _bin_edges(bin_deg)[1]
    lat, counts = data.latitude, data.counts
    if mask is not None:
        lat, counts = lat[mask], counts[mask]
    good = np.isfinite(counts)
    lat, counts = lat[good], counts[good]
    n, _ = np.histogram(lat, bins=lat_edges)
    total, _ = np.histogram(lat, bins=lat_edges, weights=counts)
    sq, _ = np.histogram(lat, bins=lat_edges, weights=counts ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, total / n, np.nan)
        var = np.where(n > 1, sq / n - mean ** 2, np.nan)
        sem = np.sqrt(np.maximum(var, 0.0) / np.maximum(n, 1))
        sem = np.where(n > 1, sem, np.nan)
    centers = 0.5 * (lat_edges[1:] + lat_edges[:-1])
    return centers, mean, sem, n


def region_stats(data, mask):
    """Summary of the samples inside a region mask.

    Returns ``{"n", "mean", "sd", "sem"}``; an empty region gives ``n = 0``
    and ``NaN`` statistics.
    """
    counts = data.counts[mask]
    counts = counts[np.isfinite(counts)]
    n = int(counts.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "sem": np.nan}
    sd = float(counts.std(ddof=1)) if n > 1 else np.nan
    return {"n": n, "mean": float(counts.mean()), "sd": sd,
            "sem": sd / np.sqrt(n) if n > 1 else np.nan}
