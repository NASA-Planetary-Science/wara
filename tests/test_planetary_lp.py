"""Tests for LP-GRS PDS search/download/read (wara.planetary.lp).

Everything runs offline: archive listing is exercised against a canned HTML
directory index, downloads against a monkeypatched urlopen, and reading
against the sample product bundled in the repo (skipped if absent).
"""
import io
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from wara.planetary import lp
from wara.planetary import (
    LPGrsDay,
    LPGrsProduct,
    doy_to_date,
    download_products,
    filter_products,
    list_grs_products,
    read_grs_day,
)

# Trimmed copy of the real IIS directory index at
# pds-geosciences.wustl.edu/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/
FAKE_INDEX = """
<html><head><title>listing</title></head><body><pre>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/">[To Parent Directory]</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/1998_016_grs.dat">1998_016_grs.dat</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/1998_016_grs.lbl">1998_016_grs.lbl</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/1998_016_grs.xml">1998_016_grs.xml</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/1998_360_grs.dat">1998_360_grs.dat</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/1998_360_grs.xml">1998_360_grs.xml</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/1999_045_grs.dat">1999_045_grs.dat</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/1999_045_grs.lbl">1999_045_grs.lbl</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/1999_045_grs.xml">1999_045_grs.xml</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/1999_050_grs.dat">orphan dat, no xml</A>
<A HREF="/lunar/lp-l-grs-3-rdr-v1/lp_2xxx/grs/readme.txt">readme.txt</A>
</pre></body></html>
"""

REPO_SAMPLE = (Path(__file__).parents[1] / "wara" / "planetary-nuclear-spect"
               / "LP" / "LP_data" / "1998_016_grs.xml")


def _fake_products():
    return list_grs_products(html=FAKE_INDEX)


# ── Archive listing ──────────────────────────────────────────────────────────
def test_list_parses_products_and_pairs_files():
    products = _fake_products()
    # The orphan .dat (no .xml) and readme.txt must be dropped.
    assert [p.stem for p in products] == [
        "1998_016_grs", "1998_360_grs", "1999_045_grs"]
    p0 = products[0]
    assert p0.url_xml.endswith("/lp_2xxx/grs/1998_016_grs.xml")
    assert p0.url_xml.startswith("https://pds-geosciences.wustl.edu/")
    assert p0.url_dat.endswith("1998_016_grs.dat")
    assert p0.url_lbl.endswith("1998_016_grs.lbl")
    # 1998_360 has no .lbl in the listing — optional, so still usable.
    assert products[1].url_lbl == ""


def test_dates_come_from_year_doy_in_filename():
    products = _fake_products()
    assert products[0].day == date(1998, 1, 16)
    assert products[1].day == date(1998, 12, 26)
    assert products[2].day == date(1999, 2, 14)
    assert doy_to_date(1998, 1) == date(1998, 1, 1)
    assert doy_to_date(1999, 365) == date(1999, 12, 31)


def test_phase_splits_at_orbit_lowering_date():
    products = _fake_products()
    assert products[0].phase == "high"   # Jan 1998, 100 km mapping orbit
    assert products[1].phase == "low"    # Dec 26 1998, after 1998-12-19
    assert products[2].phase == "low"
    assert lp.LOW_ALTITUDE_START == date(1998, 12, 19)


# ── Filtering ────────────────────────────────────────────────────────────────
def test_filter_by_date_range_inclusive_and_string_dates():
    products = _fake_products()
    sel = filter_products(products, start="1998-01-16", end="1998-12-26")
    assert [p.stem for p in sel] == ["1998_016_grs", "1998_360_grs"]
    # date objects work too, and bounds are inclusive
    sel = filter_products(products, start=date(1999, 2, 14))
    assert [p.stem for p in sel] == ["1999_045_grs"]


def test_filter_by_phase():
    products = _fake_products()
    assert [p.stem for p in filter_products(products, phase="high")] == \
        ["1998_016_grs"]
    assert len(filter_products(products, phase="low")) == 2
    with pytest.raises(ValueError):
        filter_products(products, phase="medium")


# ── Download ─────────────────────────────────────────────────────────────────
def test_download_writes_pairs_and_skips_existing(tmp_path, monkeypatch):
    fetched = []

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        fetched.append(req.full_url)
        return FakeResponse(b"payload:" + req.full_url.encode())

    monkeypatch.setattr(lp, "urlopen", fake_urlopen)
    products = _fake_products()[:2]

    calls = []
    xmls = download_products(products, tmp_path,
                             progress=lambda name, i, n: calls.append(name))
    # Returns local xml label paths, one per product, and wrote xml+dat pairs.
    assert xmls == [tmp_path / "1998_016_grs.xml", tmp_path / "1998_360_grs.xml"]
    assert sorted(f.name for f in tmp_path.iterdir()) == [
        "1998_016_grs.dat", "1998_016_grs.xml",
        "1998_360_grs.dat", "1998_360_grs.xml"]
    assert len(fetched) == 4 and len(calls) == 4
    # File contents come from the right URLs.
    body = (tmp_path / "1998_016_grs.dat").read_bytes().decode()
    assert body.endswith("1998_016_grs.dat")

    # Second run: everything exists, nothing re-fetched.
    fetched.clear()
    download_products(products, tmp_path)
    assert fetched == []


# ── Regional selection / summing on synthetic data ───────────────────────────
def _synthetic_day():
    n = 6
    spectra = np.arange(n * 4, dtype=float).reshape(n, 4)
    return LPGrsDay(
        stem="1998_016_grs", day=date(1998, 1, 16),
        spectra=spectra, rejected=spectra * 0.1,
        deadtime=np.full(n, 0.04), overload=np.zeros(n),
        temperature=np.full(n, -28.0),
        time_doy=np.linspace(16.0, 16.5, n),
        altitude_km=np.array([100, 100, 100, 30, 30, 30], dtype=float),
        latitude=np.array([0.0, 40.0, 80.0, -40.0, 0.0, 40.0]),
        longitude=np.array([0.0, -30.0, 175.0, -175.0, 90.0, -30.0]),
    )


def test_select_lat_lon_alt_ranges():
    day = _synthetic_day()
    assert day.n_records == 6
    mask = day.select(lat_range=(20, 60), lon_range=(-40, -20))
    assert list(mask) == [False, True, False, False, False, True]
    # Altitude cut mimics the 100 km vs 30 km phase separation.
    mask = day.select(lat_range=(20, 60), lon_range=(-40, -20),
                      alt_range=(90, 110))
    assert list(mask) == [False, True, False, False, False, False]
    mask = day.select(time_range=(16.0, 16.1))
    assert list(mask) == [True, True, False, False, False, False]


def test_select_lon_range_wraps_the_seam():
    day = _synthetic_day()
    mask = day.select(lon_range=(170, -170))  # far-side strip across +/-180
    assert list(mask) == [False, False, True, True, False, False]
    # 0..360-style input is wrapped too: (170, 190) == (170, -170).
    assert list(day.select(lon_range=(170, 190))) == list(mask)


def test_sum_spectrum_matches_manual_sum():
    day = _synthetic_day()
    total, n = day.sum_spectrum()
    assert n == 6
    assert np.allclose(total, day.spectra.sum(axis=0))
    mask = day.select(lat_range=(20, 60))
    region, n_region = day.sum_spectrum(mask)
    assert n_region == 2
    assert np.allclose(region, day.spectra[mask].sum(axis=0))
    rej, _ = day.sum_spectrum(mask, rejected=True)
    assert np.allclose(rej, region * 0.1)


# ── Orbit metadata (bundled CSV) ─────────────────────────────────────────────
def _synthetic_day_for_meta(stem="1998_016_grs", n=45):
    doy = int(stem.split("_")[1])
    return LPGrsDay(
        stem=stem, day=date(1998, 1, 1) + (date(1998, 1, 2) - date(1998, 1, 1)) * (doy - 1),
        spectra=np.zeros((n, 4)), rejected=np.zeros((n, 4)),
        deadtime=np.zeros(n), overload=np.zeros(n),
        temperature=np.zeros(n),
        time_doy=doy + np.linspace(0.0, 0.9, n),
        altitude_km=np.linspace(99.0, 101.0, n),
        latitude=np.linspace(-88.0, 88.0, n),
        longitude=np.linspace(-170.0, 170.0, n),
    )


def test_build_and_load_orbit_metadata_roundtrip(tmp_path, monkeypatch):
    csv = tmp_path / "meta.csv"
    days = {s: _synthetic_day_for_meta(s) for s in ("1998_016_grs", "1998_017_grs")}
    prods = [LPGrsProduct(stem=s, day=d.day, url_xml=f"http://x/{s}.xml",
                          url_dat=f"http://x/{s}.dat") for s, d in days.items()]
    fetched = []
    monkeypatch.setattr(lp, "download_products",
                        lambda products, dest, **kw: fetched.extend(
                            p.stem for p in products))
    monkeypatch.setattr(lp, "read_grs_day",
                        lambda xml: days[Path(xml).stem])

    n = lp.build_orbit_metadata(csv_path=csv, data_dir=tmp_path, stride=10,
                                products=prods)
    assert n == 2
    assert fetched == ["1998_016_grs", "1998_017_grs"]
    # Products fetched only for scraping are not kept (they were never written
    # here since download is mocked — the point is the CSV content).
    md = lp.load_orbit_metadata(csv)
    assert len(md["utc"]) == 2 * 5              # ceil(45/10) per product
    assert set(md["product"]) == set(days)
    assert md["utc"].dtype == np.dtype("datetime64[s]")
    assert np.all(np.diff(md["utc"]).astype(float) >= 0)   # time-sorted
    assert md["lat"].min() >= -90 and md["lat"].max() <= 90
    assert 99.0 <= md["alt_km"].min() <= md["alt_km"].max() <= 101.0

    # Resumable: a second run adds nothing and refetches nothing.
    fetched.clear()
    assert lp.build_orbit_metadata(csv_path=csv, data_dir=tmp_path, stride=10,
                                   products=prods) == 0
    assert fetched == []
    assert lp.products_in_metadata(csv) == set(days)


def test_build_metadata_skips_download_for_cached_products(tmp_path, monkeypatch):
    csv = tmp_path / "meta.csv"
    d = _synthetic_day_for_meta("1998_020_grs")
    (tmp_path / "1998_020_grs.xml").write_text("x")
    (tmp_path / "1998_020_grs.dat").write_text("x")
    monkeypatch.setattr(lp, "download_products",
                        lambda *a, **k: pytest.fail("should not download"))
    monkeypatch.setattr(lp, "read_grs_day", lambda xml: d)
    p = LPGrsProduct(stem=d.stem, day=d.day, url_xml="u", url_dat="u")
    assert lp.build_orbit_metadata(csv_path=csv, data_dir=tmp_path,
                                   products=[p]) == 1
    # Pre-existing files are never deleted.
    assert (tmp_path / "1998_020_grs.dat").exists()


def test_records_utc64_reanchors_mission_continuous_doy():
    # Extended-mission products carry Earth_Received_Time as DOY counted
    # continuously from 1998 (a 1999 product holds ~366+its real DOY). The
    # helper must re-anchor them to the product's own date.
    d = _synthetic_day_for_meta("1998_016_grs")
    t = lp.records_utc64(d)
    assert np.datetime_as_string(t[0], unit="D") == "1998-01-16"
    d99 = _synthetic_day_for_meta("1999_100_grs", n=10)
    d99.day = date(1999, 4, 10)                      # 1999 DOY 100
    d99.time_doy = 365 + 100 + np.linspace(0.0, 0.9, 10)   # continuous DOY
    t = lp.records_utc64(d99)
    assert np.datetime_as_string(t[0], unit="D") == "1999-04-10"
    # A product that resets normally is untouched.
    d99.time_doy = 100 + np.linspace(0.0, 0.9, 10)
    t = lp.records_utc64(d99)
    assert np.datetime_as_string(t[0], unit="D") == "1999-04-10"


def test_bundled_metadata_csv_loads_if_present():
    if not lp.LP_METADATA_CSV.exists():
        pytest.skip("bundled metadata not built yet")
    md = lp.load_orbit_metadata()
    assert len(md["utc"]) > 50000
    assert len(set(md["product"])) == 512            # every daily product
    # The mission window: no timestamps outside 1998-01-16 .. 1999-07-31.
    assert md["utc"].min() >= np.datetime64("1998-01-16")
    assert md["utc"].max() <= np.datetime64("1999-07-31T23:59:59")
    assert md["lat"].min() >= -90 and md["lat"].max() <= 90
    assert np.all(md["lon"] >= -180) and np.all(md["lon"] <= 180)
    assert md["alt_km"].min() > 5 and md["alt_km"].max() < 250


# ── Mission documentation ────────────────────────────────────────────────────
def test_fetch_document_downloads_once_then_reads_cache(tmp_path, monkeypatch):
    import io
    fetched = []

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        fetched.append(req.full_url)
        return FakeResponse(b"Mission Overview\n================\ntext")

    monkeypatch.setattr(lp, "urlopen", fake_urlopen)
    text = lp.fetch_document("Mission description", data_dir=tmp_path)
    assert "Mission Overview" in text
    assert fetched == [lp.LP_DOCUMENTS["Mission description"]]
    # Cached under docs/ with a label-derived name; second call is offline.
    assert (tmp_path / "docs" / "mission_description.txt").exists()
    fetched.clear()
    assert "Mission Overview" in lp.fetch_document("Mission description",
                                                   data_dir=tmp_path)
    assert fetched == []


def test_lp_documents_registry_is_sane():
    assert len(lp.LP_DOCUMENTS) >= 8
    labels = list(lp.LP_DOCUMENTS)
    assert "Mission description" in labels and "GRS instrument" in labels
    for url in lp.LP_DOCUMENTS.values():
        assert url.startswith("https://pds-geosciences.wustl.edu/")
    # Distinct cache slugs even where basenames collide (two aareadme.txt).
    import re
    slugs = {re.sub(r"[^a-z0-9]+", "_", l.lower()).strip("_") for l in labels}
    assert len(slugs) == len(labels)


# ── Elemental abundance maps (Level 5) ───────────────────────────────────────
def _abundance_tab_text():
    # Three equal-area-style pixels: a polar cap spanning all longitudes and
    # two half-shell cells. Columns: pixel, lat_s, lat_n, lon_w, lon_e, AM,
    # neutron_den, MgO, Al2O3, SiO2, CaO, TiO2, FeO, K, Th, U (+2 ignored
    # covariance columns to mimic the real file's trailing block).
    return (
        "0 -90.0 -30.0 -180.0 180.0 21.7 1.5e-6 "
        "0.07 0.27 0.45 0.16 0.001 0.05 400.0 1.0 0.2 1e-6 0.0\n"
        "1 -30.0 90.0 -180.0 0.0 21.9 1.2e-6 "
        "0.08 0.22 0.45 0.15 0.004 0.08 800.0 4.0 0.5 1e-6 0.0\n"
        "2 -30.0 90.0 0.0 180.0 21.8 1.4e-6 "
        "0.07 0.25 0.44 0.15 0.001 0.10 440.0 2.0 0.3 1e-6 0.0\n"
    )


def test_read_abundance_and_grid_sampling(tmp_path):
    from wara.planetary import abundance as ab

    path = tmp_path / "lpgrs_high1_elem_abundance_5deg.tab"
    path.write_text(_abundance_tab_text())
    table = ab.read_abundance(5, data_dir=tmp_path)
    assert len(table["lat_s"]) == 3
    assert np.allclose(table["Th"], [1.0, 4.0, 2.0])
    assert np.allclose(table["FeO"], [0.05, 0.08, 0.10])

    lon = np.linspace(-180.0, 180.0, 37)
    lat = np.linspace(-90.0, 90.0, 19)
    grid = ab.abundance_grid(table, "Th", lon, lat)
    assert grid.shape == (19, 37)
    assert not np.isnan(grid).any()          # pixels tile the whole sphere
    # Polar cap (all longitudes below -30 lat) = 1.0.
    assert np.all(grid[lat < -30.0, :] == 1.0)
    # Northern half-shells split at lon 0: west 4.0, east 2.0.
    assert grid[np.searchsorted(lat, 45.0), np.searchsorted(lon, -90.0)] == 4.0
    assert grid[np.searchsorted(lat, 45.0), np.searchsorted(lon, 90.0)] == 2.0
    # Upper edges are inclusive: the +90 lat / +180 lon corner has a value.
    assert grid[-1, -1] == 2.0

    with pytest.raises(ValueError):
        ab.abundance_grid(table, "Xx", lon, lat)


def test_abundance_filename_validates_resolution():
    from wara.planetary import abundance as ab
    assert ab.abundance_filename(2) == "lpgrs_high1_elem_abundance_2deg.tab"
    with pytest.raises(ValueError):
        ab.abundance_filename(3)


# ── LOLA topography ──────────────────────────────────────────────────────────
def _write_fake_lola(tmp_path, ppd=4):
    # Tiny "global" DEM honoring the real layout: row 0 = +90 lat, col 0 =
    # 0 E, int16 LSB, height = DN * 0.5 m. Use 4 rows x 8 cols.
    from wara.planetary import lola
    lines, samples = 4, 8
    lbl = (f"LINES = {lines}\nLINE_SAMPLES = {samples}\n"
           "SAMPLE_TYPE = LSB_INTEGER\nSAMPLE_BITS = 16\n"
           "SCALING_FACTOR = 0.5\nOFFSET = 1737400.\n")
    (tmp_path / lola.lola_filename(ppd, "lbl")).write_text(lbl)
    dn = np.arange(lines * samples, dtype="<i2").reshape(lines, samples)
    dn[0, 0] = -4000        # +90 lat, 0 E: -2.0 km
    dn[3, 4] = 8000         # southern row, 180 E: +4.0 km
    dn.tofile(tmp_path / lola.lola_filename(ppd, "img"))
    return dn


def test_read_lola_dem_parses_label_and_scales(tmp_path):
    from wara.planetary import lola
    _write_fake_lola(tmp_path)
    dem = lola.read_lola_dem(4, data_dir=tmp_path)
    assert dem["elev_km"].shape == (4, 8)
    assert dem["elev_km"][0, 0] == -2.0     # DN -4000 * 0.5 m = -2 km
    assert dem["elev_km"][3, 4] == 4.0
    with pytest.raises(ValueError):
        lola.lola_filename(7)


def test_elevation_grid_samples_with_lon_wrap(tmp_path):
    from wara.planetary import lola
    _write_fake_lola(tmp_path)
    dem = lola.read_lola_dem(4, data_dir=tmp_path)
    lon = np.array([-180.0, -90.0, 0.0, 90.0, 179.9])
    lat = np.array([-89.9, 0.1, 89.9])
    g = lola.elevation_grid(dem, lon, lat)
    assert g.shape == (3, 5)
    # +90 lat / 0 E (row 0, col 0 of the raster) = -2 km.
    assert g[2, 2] == -2.0
    # Southern row at lon -180 == raster 180 E (col 4) = +4 km.
    assert g[0, 0] == 4.0


def test_bundled_lola_dem_loads_if_present():
    from wara.planetary import lola
    from wara.planetary.lp import LP_DATA_DIR
    if not (LP_DATA_DIR / "ldem_4.img").exists():
        pytest.skip("LOLA DEM not downloaded")
    dem = lola.read_lola_dem(4)
    assert dem["elev_km"].shape == (720, 1440)
    # Known LOLA global extremes: about -9.1 to +10.8 km.
    assert -10.0 < dem["elev_km"].min() < -7.0
    assert 9.0 < dem["elev_km"].max() < 12.0


# ── Reading the bundled real product ─────────────────────────────────────────
@pytest.mark.skipif(not REPO_SAMPLE.exists(), reason="repo sample not present")
def test_read_real_product():
    pytest.importorskip("pds4_tools")
    day = read_grs_day(REPO_SAMPLE)
    assert day.stem == "1998_016_grs"
    assert day.day == date(1998, 1, 16)
    assert day.spectra.shape == (day.n_records, 512)
    assert day.rejected.shape == day.spectra.shape
    for attr in ("deadtime", "overload", "temperature", "time_doy",
                 "altitude_km", "latitude", "longitude"):
        assert getattr(day, attr).shape == (day.n_records,)
    # Longitude was wrapped from 0..360 east to [-180, 180).
    assert day.longitude.min() >= -180.0 and day.longitude.max() < 180.0
    # First science day: 100 km mapping orbit, pole-to-pole coverage.
    assert 90.0 < day.altitude_km.mean() < 110.0
    assert day.latitude.min() < -85.0 and day.latitude.max() > 85.0
    # A regional selection returns a plausible, non-empty spectrum.
    mask = day.select(lat_range=(-30, 30), lon_range=(-60, 60))
    spectrum, n_sel = day.sum_spectrum(mask)
    assert 0 < n_sel < day.n_records
    assert spectrum.shape == (512,)
    assert spectrum.sum() > 0
