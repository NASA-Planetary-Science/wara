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
