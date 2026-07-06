"""Tests for the Planetary tab's 3D Moon coordinate system and figure.

The important invariant is that (longitude, latitude) maps to the physically
correct point on the sphere and that the mapping inverts exactly. The figure
tests just confirm the Plotly object is well-formed and self-consistent.
"""
import numpy as np
import pytest

from wara.planetary import moon
from wara.planetary import (
    R_MOON_KM,
    lonlat_to_xyz,
    xyz_to_lonlat,
    sphere_mesh,
    default_texture_path,
    load_texture_intensity,
    moon_figure,
)


# ── Coordinate conversions ───────────────────────────────────────────────────
def test_cardinal_points_map_to_expected_axes():
    R = R_MOON_KM
    # (lon, lat) -> expected (x, y, z)
    cases = [
        ((0.0, 0.0), (R, 0.0, 0.0)),      # sub-Earth point on +x
        ((90.0, 0.0), (0.0, R, 0.0)),     # +90E on +y
        ((-90.0, 0.0), (0.0, -R, 0.0)),   # 90W on -y
        ((180.0, 0.0), (-R, 0.0, 0.0)),   # far side on -x
        ((0.0, 90.0), (0.0, 0.0, R)),     # north pole on +z
        ((0.0, -90.0), (0.0, 0.0, -R)),   # south pole on -z
    ]
    for (lon, lat), expected in cases:
        xyz = lonlat_to_xyz(lon, lat)
        assert np.allclose(xyz, expected, atol=1e-6), (lon, lat, xyz, expected)


def test_radius_scales_and_is_default_lunar():
    # Every surface point sits at exactly R_MOON_KM from the center.
    lon = np.linspace(-180, 180, 37)
    lat = np.linspace(-90, 90, 19)
    lon_g, lat_g = np.meshgrid(lon, lat)
    x, y, z = lonlat_to_xyz(lon_g, lat_g)
    assert np.allclose(np.sqrt(x**2 + y**2 + z**2), R_MOON_KM)
    # Custom radius (e.g. a lifted marker) scales linearly.
    x2, y2, z2 = lonlat_to_xyz(lon_g, lat_g, radius=R_MOON_KM * 1.02)
    assert np.allclose(np.sqrt(x2**2 + y2**2 + z2**2), R_MOON_KM * 1.02)


def test_roundtrip_lonlat_xyz_lonlat():
    rng = np.random.default_rng(0)
    lon = rng.uniform(-179.9, 179.9, 500)
    lat = rng.uniform(-89.9, 89.9, 500)
    x, y, z = lonlat_to_xyz(lon, lat)
    lon2, lat2 = xyz_to_lonlat(x, y, z)
    assert np.allclose(lon, lon2, atol=1e-6)
    assert np.allclose(lat, lat2, atol=1e-6)


def test_roundtrip_survives_radius_change():
    # A marker lifted off the surface must still report its true lon/lat.
    lon, lat = 42.0, -17.0
    x, y, z = lonlat_to_xyz(lon, lat, radius=R_MOON_KM * 1.05)
    lon2, lat2 = xyz_to_lonlat(x, y, z)
    assert np.isclose(lon, float(lon2), atol=1e-6)
    assert np.isclose(lat, float(lat2), atol=1e-6)


def test_wrap_lon_normalizes_range():
    assert np.isclose(moon.wrap_lon(190.0), -170.0)
    assert np.isclose(moon.wrap_lon(-190.0), 170.0)
    assert np.isclose(moon.wrap_lon(360.0), 0.0)
    assert np.all(np.abs(moon.wrap_lon(np.linspace(-720, 720, 101))) <= 180.0)


def test_east_longitude_is_positive_y():
    # Sanity on handedness: increasing longitude from 0 moves toward +y.
    _, y_small, _ = lonlat_to_xyz(10.0, 0.0)
    assert y_small > 0


# ── Mesh ─────────────────────────────────────────────────────────────────────
def test_sphere_mesh_shape_and_bounds():
    x, y, z, lon_g, lat_g = sphere_mesh(n_lon=72, n_lat=36)
    assert x.shape == y.shape == z.shape == (36, 72)
    assert np.isclose(lon_g.min(), -180) and np.isclose(lon_g.max(), 180)
    assert np.isclose(lat_g.min(), -90) and np.isclose(lat_g.max(), 90)
    assert np.allclose(np.sqrt(x**2 + y**2 + z**2), R_MOON_KM)


# ── Texture ──────────────────────────────────────────────────────────────────
def test_default_texture_exists():
    import os
    assert os.path.exists(default_texture_path())


def test_texture_intensity_matches_mesh_and_is_normalized():
    pytest.importorskip("PIL")
    _, _, _, lon_g, lat_g = sphere_mesh(n_lon=72, n_lat=36)
    intensity = load_texture_intensity(lon_g, lat_g)
    assert intensity.shape == lon_g.shape
    assert intensity.min() >= 0.0 and intensity.max() <= 1.0


# ── Figure ───────────────────────────────────────────────────────────────────
def test_moon_figure_builds_with_markers():
    pytest.importorskip("plotly")
    markers = [(0.0, 0.0, "origin"), (90.0, 45.0, "NE")]
    fig = moon_figure(n_lon=48, n_lat=24, markers=markers)
    # One Surface + graticule lines + one Scatter3d marker trace.
    types = [type(t).__name__ for t in fig.data]
    assert "Surface" in types
    assert "Scatter3d" in types
    # The marker trace carries our two labels.
    marker_traces = [t for t in fig.data
                     if type(t).__name__ == "Scatter3d" and t.mode == "markers+text"]
    assert len(marker_traces) == 1
    assert list(marker_traces[0].text) == ["origin", "NE"]


def test_moon_figure_without_graticule_or_markers():
    pytest.importorskip("plotly")
    fig = moon_figure(n_lon=48, n_lat=24, graticule=False, markers=None)
    types = [type(t).__name__ for t in fig.data]
    assert types == ["Surface"]


def test_surface_hovertext_matches_geometry():
    pytest.importorskip("plotly")
    fig = moon_figure(n_lon=48, n_lat=24, graticule=False)
    surf = fig.data[0]
    # Surface shows raw per-vertex text via hoverinfo="text" (templates do not
    # resolve reliably on go.Surface). The per-vertex label must match the
    # lon/lat recovered from the surface geometry itself.
    assert surf.hoverinfo == "text"
    # Must stay a 2D (nlat, nlon) array. A list-of-lists gets collapsed to one
    # string per row, which is exactly the hover-misalignment bug we fixed.
    text = np.asarray(surf.text)
    assert text.shape == surf.z.shape
    lon_r, lat_r = xyz_to_lonlat(surf.x, surf.y, surf.z)
    # Check a few interior vertices (away from the +/-180 seam) parse back.
    for i, j in [(5, 5), (12, 20), (18, 40)]:
        label = text[i, j]
        parts = label.replace("°", "").split()
        lon_lbl, lat_lbl = float(parts[1]), float(parts[3])
        if abs(lon_lbl) < 179.9:
            assert np.isclose(lon_lbl, lon_r[i, j], atol=0.05)
        assert np.isclose(lat_lbl, lat_r[i, j], atol=0.05)
