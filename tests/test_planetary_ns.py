"""Tests for the LP Neutron Spectrometer reader/binning (wara.planetary.ns).

The archive's binary layout is a bare little-endian dump — a uint32 sample
count followed by back-to-back float32 arrays — so every fixture here writes
real files in that exact format instead of mocking the reader. Products are
monkeypatched down to a handful of samples: the module validates what it reads
against the sample counts the PDS4 labels declare, and the real ones are in
the hundreds of thousands.
"""
import numpy as np
import pytest

from wara.planetary import ns


def write_dat(path, arrays):
    """Write the archive's format: uint32 count, then float32 arrays."""
    arrays = [np.asarray(a, dtype="<f4") for a in arrays]
    n = len(arrays[0])
    with open(path, "wb") as f:
        f.write(np.array([n], dtype="<u4").tobytes())
        for a in arrays:
            f.write(a.tobytes())
    return path


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A 6-sample stand-in for epithermal / high orbit / 32 s."""
    key = ("epithermal", "high", 32)
    product = ns.NSProduct(*key, n_samples=6)
    monkeypatch.setitem(ns.NS_PRODUCTS, key, product)
    write_dat(tmp_path / product.counts_name, [[10, 20, 30, 40, 50, 60]])
    write_dat(tmp_path / product.position_name, [
        [-80.0, -80.0, 0.0, 0.0, 80.0, 80.0],          # latitude
        [10.0, 350.0, 10.0, 200.0, 10.0, 10.0],        # longitude, 0..360 east
        [100.0, 101.0, 99.0, 98.0, 100.0, 102.0],      # altitude
        [17.5, 17.6, 200.0, 300.0, 380.0, 400.0],      # continuous DOY
    ])
    return tmp_path


# ── Catalog ──────────────────────────────────────────────────────────────────
def test_catalog_matches_the_archive():
    """11 archived products (thermal/epithermal at both cadences and both
    orbits, fast at 32 s only, one low-orbit moderated) plus the 4 derived
    thermal/epithermal ratios."""
    archived = {k: p for k, p in ns.NS_PRODUCTS.items() if not p.is_ratio}
    assert len(archived) == 11
    assert len(ns.NS_PRODUCTS) == 15
    assert ("fast", "high", 32) in ns.NS_PRODUCTS
    assert ("fast", "high", 8) not in ns.NS_PRODUCTS
    assert ("moderated", "low", 32) in ns.NS_PRODUCTS
    assert ("moderated", "high", 32) not in ns.NS_PRODUCTS


def test_ratio_products_exist_wherever_both_components_do():
    for phase in ("high", "low"):
        for cadence in (8, 32):
            assert ("thermal/epithermal", phase, cadence) in ns.NS_PRODUCTS
    p = ns.ns_product("thermal/epithermal", "high", 32)
    assert p.is_ratio and p.components == ("thermal", "epithermal")
    # Both component count files plus the shared ephemeris.
    assert p.files == ["thermal_neutron_high32sec.dat",
                       "epitherm_neutron_high32sec.dat",
                       "position_high32sec.dat"]
    assert p.value_label == "thermal / epithermal ratio"
    assert "Thermal / epithermal ratio" in p.label
    with pytest.raises(ValueError, match="not archived"):
        p.counts_name


def test_ratio_divides_per_accumulation(tmp_path, monkeypatch):
    """Thermal and epithermal share one position file and one set of
    accumulations, so the ratio is exact sample by sample."""
    for kind, counts in (("thermal", [30.0, 40.0, 0.0]),
                         ("epithermal", [10.0, 8.0, 5.0])):
        key = (kind, "high", 32)
        product = ns.NSProduct(*key, n_samples=3)
        monkeypatch.setitem(ns.NS_PRODUCTS, key, product)
        write_dat(tmp_path / product.counts_name, [counts])
    key = ("thermal/epithermal", "high", 32)
    monkeypatch.setitem(ns.NS_PRODUCTS, key, ns.NSProduct(
        *key, n_samples=3, components=("thermal", "epithermal")))
    write_dat(tmp_path / "position_high32sec.dat",
              [[-80.0, 0.0, 80.0], [0.0, 10.0, 20.0],
               [100.0, 100.0, 100.0], [17.0, 18.0, 19.0]])
    data = ns.read_ns(*key, data_dir=tmp_path)
    assert list(data.counts) == [3.0, 5.0, 0.0]
    assert data.product.is_ratio
    # A zero *denominator* would be NaN, not infinity.
    write_dat(tmp_path / "epitherm_neutron_high32sec.dat", [[10.0, 0.0, 5.0]])
    data = ns.read_ns(*key, data_dir=tmp_path)
    assert np.isnan(data.counts[1])
    # NaN samples are dropped from the binning, not spread over their cell.
    grid, _, _ = ns.neutron_bins(data, bin_deg=90.0)
    assert np.isfinite(grid[np.isfinite(grid)]).all()
    _, mean, _, n = ns.zonal_profile(data, bin_deg=90.0)
    assert n.sum() == 2
    stats = ns.region_stats(data, np.ones(3, dtype=bool))
    assert stats["n"] == 2


def test_product_file_names():
    p = ns.ns_product("epithermal", "high", 32)
    assert p.counts_name == "epitherm_neutron_high32sec.dat"
    assert p.position_name == "position_high32sec.dat"
    assert p.files == [p.counts_name, p.position_name]
    # Moderated neutrons carry their own ephemeris — no position companion.
    mod = ns.ns_product("moderated", "low", 32)
    assert mod.self_positioned and mod.position_name is None
    assert mod.files == ["moderated_neutron_low32sec.dat"]


def test_eight_second_products_have_no_altitude_or_time():
    p = ns.ns_product("thermal", "low", 8)
    assert not p.has_altitude and not p.has_time


def test_unavailable_combination_explains_itself():
    with pytest.raises(ValueError, match="no fast neutron data"):
        ns.ns_product("fast", "high", 8)
    with pytest.raises(ValueError, match="unknown neutron type"):
        ns.ns_product("cold", "high", 32)
    with pytest.raises(ValueError, match="high.*low"):
        ns.ns_product("thermal", "middle", 32)


# ── Reading ──────────────────────────────────────────────────────────────────
def test_read_pairs_counts_with_position(tiny):
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    assert data.n_samples == 6
    assert list(data.counts) == [10, 20, 30, 40, 50, 60]
    assert list(data.altitude_km[:2]) == [100.0, 101.0]
    # Longitude is wrapped from the archive's 0..360 east to [-180, 180).
    assert data.longitude[1] == pytest.approx(-10.0)
    assert data.longitude[3] == pytest.approx(-160.0)
    assert data.kind == "epithermal" and "Epithermal" in data.label


def test_utc_anchors_on_1998_and_survives_the_year_boundary(tiny):
    """Earth_Received_Time is a mission-continuous day-of-year from
    1998-01-01, so extended-mission samples carry values past 365."""
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    utc = data.utc64
    assert str(utc[0]).startswith("1998-01-17")
    assert str(utc[-1]).startswith("1999-02-04")   # DOY 400


def test_moderated_product_reads_its_own_position(tmp_path, monkeypatch):
    key = ("moderated", "low", 32)
    product = ns.NSProduct(*key, n_samples=3, self_positioned=True)
    monkeypatch.setitem(ns.NS_PRODUCTS, key, product)
    write_dat(tmp_path / product.counts_name, [
        [1.0, 2.0, 3.0],          # counts
        [-10.0, 0.0, 10.0],       # latitude
        [0.0, 90.0, 350.0],       # longitude
        [30.0, 31.0, 32.0],       # altitude
        [360.0, 361.0, 362.0],    # continuous DOY
    ])
    data = ns.read_ns(*key, data_dir=tmp_path)
    assert list(data.counts) == [1.0, 2.0, 3.0]
    assert data.longitude[2] == pytest.approx(-10.0)
    assert data.altitude_km[0] == pytest.approx(30.0)


def test_eight_second_read_leaves_altitude_and_time_none(tmp_path, monkeypatch):
    key = ("thermal", "high", 8)
    product = ns.NSProduct(*key, n_samples=4)
    monkeypatch.setitem(ns.NS_PRODUCTS, key, product)
    write_dat(tmp_path / product.counts_name, [[1, 2, 3, 4]])
    write_dat(tmp_path / product.position_name,
              [[0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 20.0, 30.0]])
    data = ns.read_ns(*key, data_dir=tmp_path)
    assert data.altitude_km is None and data.time_doy is None
    assert data.utc64 is None


def test_truncated_download_fails_loudly(tiny):
    """A short file would otherwise mis-pair counts with positions."""
    product = ns.ns_product("epithermal", "high", 32)
    path = tiny / product.counts_name
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError, match="truncated"):
        ns.read_ns("epithermal", "high", 32, data_dir=tiny)


def test_sample_count_must_match_the_label(tmp_path, monkeypatch):
    key = ("epithermal", "low", 32)
    monkeypatch.setitem(ns.NS_PRODUCTS, key, ns.NSProduct(*key, n_samples=5))
    product = ns.NS_PRODUCTS[key]
    write_dat(tmp_path / product.counts_name, [[1, 2, 3]])       # 3, not 5
    write_dat(tmp_path / product.position_name, [[0], [0], [0], [0]])
    with pytest.raises(ValueError, match="expected 5 samples"):
        ns.read_ns(*key, data_dir=tmp_path)


# ── Selection ────────────────────────────────────────────────────────────────
def test_select_box_and_seam(tiny):
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    mask = data.select(lat_range=(-85, -75))
    assert list(mask) == [True, True, False, False, False, False]
    # A box across the +/-180 seam catches lon -160 but not lon 10.
    mask = data.select(lon_range=(170, -150))
    assert list(mask) == [False, False, False, True, False, False]
    # Altitude and time cut together: only the second sample is at 101 km
    # *and* early in the mission.
    mask = data.select(alt_range=(100.5, 101.5), time_range=(0, 100))
    assert list(mask) == [False, True, False, False, False, False]


def test_select_rejects_altitude_and_time_at_8s(tmp_path, monkeypatch):
    key = ("thermal", "low", 8)
    product = ns.NSProduct(*key, n_samples=2)
    monkeypatch.setitem(ns.NS_PRODUCTS, key, product)
    write_dat(tmp_path / product.counts_name, [[1, 2]])
    write_dat(tmp_path / product.position_name, [[0.0, 1.0], [0.0, 1.0]])
    data = ns.read_ns(*key, data_dir=tmp_path)
    with pytest.raises(ValueError, match="no altitude array"):
        data.select(alt_range=(0, 50))
    with pytest.raises(ValueError, match="no time array"):
        data.select(time_range=(0, 400))


# ── Binning ──────────────────────────────────────────────────────────────────
def test_neutron_bins_mean_and_coverage(tiny):
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    grid, lon_edges, lat_edges = ns.neutron_bins(data, bin_deg=90.0)
    assert grid.shape == (2, 4)                     # 180/90 x 360/90
    assert len(lon_edges) == 5 and len(lat_edges) == 3
    counts, _, _ = ns.neutron_bins(data, bin_deg=90.0, statistic="samples")
    assert counts.sum() == data.n_samples
    # Cells nothing flew over are NaN in the mean map, 0 in the coverage map.
    assert np.isnan(grid).any() and (counts == 0).any()
    assert np.nanmin(grid) >= 10.0 and np.nanmax(grid) <= 60.0


def test_neutron_bins_honors_a_mask(tiny):
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    mask = data.select(lat_range=(-90, -10))        # the two southern samples
    counts, _, _ = ns.neutron_bins(data, bin_deg=90.0, statistic="samples",
                                   mask=mask)
    assert counts.sum() == 2


def test_neutron_bins_rejects_unknown_statistic(tiny):
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    with pytest.raises(ValueError, match="unknown statistic"):
        ns.neutron_bins(data, statistic="median")


def test_neutron_map_matches_the_mesh_and_stays_blocky(tiny):
    """Same contract as abundance_grid: one value per mesh point, sampled
    from the coarse bins (so a fine globe mesh shows map pixels)."""
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    lon_axis = np.linspace(-180, 180, 37)
    lat_axis = np.linspace(-90, 90, 19)
    grid = ns.neutron_map(data, lon_axis, lat_axis, bin_deg=90.0)
    assert grid.shape == (19, 37)
    # The 4x2 bin grid has at most 8 distinct values across 703 mesh points.
    assert len(set(np.unique(grid[~np.isnan(grid)]).tolist())) <= 8


def test_zonal_profile_reproduces_the_band_means(tiny):
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    lat, mean, sem, n = ns.zonal_profile(data, bin_deg=90.0)
    assert list(lat) == [-45.0, 45.0]
    # Band edges are [-90, 0) and [0, 90]: the two equatorial samples land in
    # the northern band.
    assert list(n) == [2, 4]
    assert mean[0] == pytest.approx(np.mean([10, 20]))
    assert mean[1] == pytest.approx(np.mean([30, 40, 50, 60]))
    assert np.all(sem > 0)


def test_zonal_profile_empty_bands_are_nan(tiny):
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    lat, mean, sem, n = ns.zonal_profile(data, bin_deg=10.0)
    assert np.isnan(mean[n == 0]).all()
    assert n.sum() == data.n_samples


def test_region_stats(tiny):
    data = ns.read_ns("epithermal", "high", 32, data_dir=tiny)
    stats = ns.region_stats(data, data.select(lat_range=(-85, -75)))
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(15.0)
    assert stats["sd"] == pytest.approx(np.std([10, 20], ddof=1))
    empty = ns.region_stats(data, np.zeros(data.n_samples, dtype=bool))
    assert empty["n"] == 0 and np.isnan(empty["mean"])
