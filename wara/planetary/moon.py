"""A textured, correctly-georeferenced 3D Moon for the Planetary tab.

The one thing that has to be right here is the **coordinate system**: a point
given as ``(longitude, latitude)`` in degrees must land on the physically
correct spot of the sphere, and clicking the sphere must recover the same
``(longitude, latitude)``. Everything else (texture, graticule, styling) is
cosmetic and layered on top of that mapping.

Conventions
-----------
* **Planetocentric, east-positive.** Longitude increases eastward; latitude is
  measured from the equator. This matches the Lunar Prospector PDS ephemeris
  and the convention used by the reduced LP-GRS products.
* Longitude is carried internally in the ``[-180, 180]`` range (0 deg = the
  sub-Earth / near-side prime meridian). :func:`wrap_lon` normalizes any input.
* The Cartesian frame is the usual right-handed one::

      x = R cos(lat) cos(lon)
      y = R cos(lat) sin(lon)
      z = R sin(lat)

  so ``+x`` points at (lon, lat) = (0, 0), ``+y`` at (90, 0), and ``+z`` at the
  north pole.
* Radius is in kilometers (mean lunar radius ``R_MOON_KM``); pass ``radius`` to
  override (e.g. to draw a marker slightly above the surface).

The bundled equirectangular albedo map (``data/moon_albedo_1024.png``, a
downsampled NASA/USGS lunar mosaic) is centered on 0 deg: its left edge is
lon = -180, right edge lon = +180, top row lat = +90, bottom row lat = -90.
"""
from importlib.resources import files

import numpy as np

# Mean radius of the Moon in km (IAU).
R_MOON_KM = 1737.4

# Important lunar reference points as (lon_east, lat, name), planetocentric
# east-positive degrees in [-180, 180). Center coordinates from the IAU/USGS
# Gazetteer of Planetary Nomenclature. Chosen for orientation and for LP-GRS
# science: the major near-side maria (KREEP/Th-rich Procellarum-Imbrium
# terrane), landmark craters, the far-side basins, the poles (LP's hydrogen
# result), and the Apollo 11 site.
LUNAR_LANDMARKS = [
    # Near-side maria
    (-15.6, 32.8, "Mare Imbrium"),
    (17.5, 28.0, "Mare Serenitatis"),
    (31.4, 8.5, "Mare Tranquillitatis"),
    (59.1, 17.0, "Mare Crisium"),
    (51.3, -7.8, "Mare Fecunditatis"),
    (35.5, -15.2, "Mare Nectaris"),
    (-16.6, -21.3, "Mare Nubium"),
    (-38.6, -24.4, "Mare Humorum"),
    (-57.4, 18.4, "Oceanus Procellarum"),
    (1.4, 56.0, "Mare Frigoris"),
    # Landmark craters
    (-11.4, -43.3, "Tycho"),
    (-20.1, 9.6, "Copernicus"),
    (-47.4, 23.7, "Aristarchus"),
    (-9.4, 51.6, "Plato"),
    # Far side
    (-169.0, -53.0, "South Pole–Aitken basin"),
    (-92.8, -19.4, "Mare Orientale"),
    (147.9, 27.3, "Mare Moscoviense"),
    (129.1, -20.4, "Tsiolkovskiy"),
    # Poles and historic site
    (0.0, 90.0, "North pole"),
    (0.0, -90.0, "South pole"),
    (23.5, 0.7, "Apollo 11"),
]

# Dark theme background matching the WARA GUI (see wara.gui.theme).
_BG = "#0a0a0f"


# ── Coordinate conversions ───────────────────────────────────────────────────
def wrap_lon(lon):
    """Wrap longitude(s) into the half-open range ``[-180, 180)`` degrees."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def lonlat_to_xyz(lon, lat, radius=R_MOON_KM):
    """Convert longitude/latitude (degrees) to Cartesian ``(x, y, z)``.

    Accepts scalars or array-likes of matching shape. ``radius`` may be a scalar
    or broadcastable array (handy for lifting markers just off the surface).
    """
    lon_r = np.radians(np.asarray(lon, dtype=float))
    lat_r = np.radians(np.asarray(lat, dtype=float))
    cos_lat = np.cos(lat_r)
    x = radius * cos_lat * np.cos(lon_r)
    y = radius * cos_lat * np.sin(lon_r)
    z = radius * np.sin(lat_r)
    return x, y, z


def xyz_to_lonlat(x, y, z):
    """Convert Cartesian ``(x, y, z)`` back to ``(longitude, latitude)`` degrees.

    The radius is taken from the point itself, so this inverts
    :func:`lonlat_to_xyz` for any ``radius`` (including lifted markers). Longitude
    is returned in ``[-180, 180)``; latitude in ``[-90, 90]``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.sqrt(x * x + y * y + z * z)
    # Guard the poles / origin so arcsin stays in range.
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.degrees(np.arcsin(np.clip(np.where(r == 0, 0.0, z / r), -1.0, 1.0)))
    lon = wrap_lon(np.degrees(np.arctan2(y, x)))
    return lon, lat


# ── Geometry ─────────────────────────────────────────────────────────────────
def sphere_mesh(n_lon=360, n_lat=180, radius=R_MOON_KM):
    """Build a lon/lat sphere mesh for the Moon surface.

    Returns ``(x, y, z, lon_grid, lat_grid)`` where each array has shape
    ``(n_lat, n_lon)``. ``lon`` spans ``[-180, 180]`` and ``lat`` spans
    ``[-90, 90]`` inclusive, so the seam and both poles are represented.
    """
    lon = np.linspace(-180.0, 180.0, n_lon)
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y, z = lonlat_to_xyz(lon_grid, lat_grid, radius)
    return x, y, z, lon_grid, lat_grid


def graticule_traces(step=30, radius=R_MOON_KM, n=181, color="rgba(255,255,255,0.18)"):
    """Latitude/longitude grid lines, lifted just above the surface.

    Returns a list of Plotly ``Scatter3d`` line traces (one per meridian and
    parallel), spaced every ``step`` degrees. Requires ``plotly``.
    """
    import plotly.graph_objects as go

    lift = radius * 1.002
    traces = []
    # Meridians (constant longitude, latitude sweeps pole to pole).
    lat = np.linspace(-90, 90, n)
    for lon0 in np.arange(-180, 180, step):
        x, y, z = lonlat_to_xyz(np.full_like(lat, lon0), lat, lift)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", line=dict(color=color, width=1),
            hoverinfo="skip", showlegend=False))
    # Parallels (constant latitude, longitude sweeps the full circle).
    lon = np.linspace(-180, 180, n)
    for lat0 in np.arange(-60, 90, step):
        x, y, z = lonlat_to_xyz(lon, np.full_like(lon, lat0), lift)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", line=dict(color=color, width=1),
            hoverinfo="skip", showlegend=False))
    return traces


# ── Texture ──────────────────────────────────────────────────────────────────
def default_texture_path(size=1024):
    """Path to a bundled equirectangular lunar albedo map (grayscale).

    Two downsamples of the NASA/USGS mosaic ship with wara: ``size=1024``
    (1024x512, the default) and ``size=2048`` (2048x1024, for high-resolution
    globes such as the GUI's Planetary tab).
    """
    if size not in (1024, 2048):
        raise ValueError(f"no bundled texture of size {size} (use 1024 or 2048)")
    return str(files("wara.planetary").joinpath(f"data/moon_albedo_{size}.png"))


def load_texture_intensity(lon_grid, lat_grid, texture=None):
    """Sample a grayscale albedo image onto the mesh vertices.

    ``lon_grid`` / ``lat_grid`` are the degree grids from :func:`sphere_mesh`.
    ``texture`` is a path to an equirectangular image (default: the bundled map)
    centered on 0 deg, north-up. Returns a float array (same shape as the grids)
    of 0..1 intensities to feed ``go.Surface(surfacecolor=...)``, or ``None`` if
    Pillow is unavailable.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    Image.MAX_IMAGE_PIXELS = None
    path = texture or default_texture_path()
    img = np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0
    h, w = img.shape
    # Map lon in [-180,180] -> column [0, w-1]; lat in [-90,90] -> row [0, h-1]
    # with the top row (row 0) at the north pole (+90).
    col = np.clip(((wrap_lon(lon_grid) + 180.0) / 360.0 * (w - 1)).astype(int), 0, w - 1)
    row = np.clip(((90.0 - lat_grid) / 180.0 * (h - 1)).astype(int), 0, h - 1)
    return img[row, col]


# ── Figure ───────────────────────────────────────────────────────────────────
def moon_figure(texture=None, n_lon=360, n_lat=180, radius=R_MOON_KM,
                graticule=True, markers=None, title="Lunar Prospector — Moon",
                hover=True, compact=False):
    """Build a Plotly 3D figure of the Moon with the correct lon/lat mapping.

    Parameters
    ----------
    texture : str or None
        Path to an equirectangular grayscale albedo map. ``None`` uses the
        bundled downsampled NASA/USGS mosaic. Pass the full-resolution image for
        a crisper globe.
    n_lon, n_lat : int
        Surface mesh resolution (longitude x latitude samples).
    radius : float
        Sphere radius in km.
    graticule : bool
        Overlay 30-degree lat/lon grid lines.
    markers : sequence of (lon, lat, label) or None
        Optional reference points drawn as labeled dots just above the surface —
        useful for visually confirming the coordinate system.
    title : str
        Figure title.
    hover : bool
        Attach the per-vertex "lon/lat" hover label to the surface. At high mesh
        resolutions this string array dominates the serialized figure size, so
        embedders that do their own hover readout (e.g. the GUI's JavaScript
        overlay) pass ``hover=False``.
    compact : bool
        Round coordinates to 0.1 km and intensities to 3 decimals. Visually
        lossless but shrinks the JSON/HTML payload substantially — intended for
        high-resolution meshes embedded in a QWebEngineView.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    x, y, z, lon_grid, lat_grid = sphere_mesh(n_lon, n_lat, radius)
    intensity = load_texture_intensity(lon_grid, lat_grid, texture)
    if intensity is None:
        # No Pillow: fall back to a flat mid-gray sphere so the figure still
        # renders (coordinates are unaffected).
        intensity = np.full_like(lon_grid, 0.6)
    if compact:
        x, y, z = np.round(x, 1), np.round(y, 1), np.round(z, 1)
        intensity = np.round(intensity, 3)

    surface_kw = {}
    if hover:
        # Pre-format the per-vertex coordinate label. go.Surface does not resolve
        # hovertemplate tokens (``%{customdata[0]}``/``%{text}`` leak the raw
        # template), and a *list of lists* is collapsed to one string per row (so
        # hover only aligns on a fraction of the mesh). A 2D **numpy** array is
        # kept as (nlat, nlon); ``hoverinfo="text"`` shows it directly.
        wlon = wrap_lon(lon_grid)
        hovertext = np.array([
            [f"lon {float(lon_v):.1f}°  lat {float(lat_v):.1f}°"
             for lon_v, lat_v in zip(lon_row, lat_row)]
            for lon_row, lat_row in zip(wlon, lat_grid)
        ])
        surface_kw.update(text=hovertext, hoverinfo="text")
    else:
        surface_kw.update(hoverinfo="none")
    surface = go.Surface(
        x=x, y=y, z=z, surfacecolor=intensity,
        colorscale="gray", cmin=0.0, cmax=1.0, showscale=False,
        lighting=dict(ambient=0.85, diffuse=0.25, specular=0.05),
        # Plotly highlights the hovered mesh row/column with dark contour
        # curves by default — on a globe they read as bogus orbit tracks.
        contours=dict(x=dict(highlight=False), y=dict(highlight=False),
                      z=dict(highlight=False)),
        **surface_kw,
    )
    traces = [surface]
    if graticule:
        traces.extend(graticule_traces(radius=radius))

    if markers:
        mlon, mlat, mtext = [], [], []
        for lon0, lat0, label in markers:
            mlon.append(lon0); mlat.append(lat0); mtext.append(label)
        mx, my, mz = lonlat_to_xyz(mlon, mlat, radius * 1.02)
        traces.append(go.Scatter3d(
            x=mx, y=my, z=mz, mode="markers+text",
            marker=dict(size=5, color="#00e5ff"),
            text=mtext, textposition="top center",
            textfont=dict(color="#f4f6ff", size=11),
            hoverinfo="text", showlegend=False))

    fig = go.Figure(data=traces)
    axis = dict(visible=False, showgrid=False, zeroline=False, showbackground=False)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#f4f6ff")),
        paper_bgcolor=_BG,
        scene=dict(
            xaxis=axis, yaxis=axis, zaxis=axis,
            aspectmode="data",
            bgcolor=_BG,
            camera=dict(eye=dict(x=1.8, y=0.0, z=0.6), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
