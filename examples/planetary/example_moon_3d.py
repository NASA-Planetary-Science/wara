"""Minimum working example: a 3D Moon with the correct lat/lon coordinates.

This is step 1 of the WARA Planetary tab (Lunar Prospector). It builds an
interactive Plotly globe of the Moon from the bundled equirectangular albedo
map and drops labeled reference markers at well-known locations so you can
visually confirm that longitude/latitude map to the right spots.

Run it with::

    python example_moon_3d.py

A browser tab (or your default Plotly renderer) opens with the interactive
globe. Orbit it: 0 deg lon / 0 deg lat faces +x (the sub-Earth point), the
north pole is +z. Hover anywhere on the surface to read the lon/lat there.
"""
import numpy as np

from wara.planetary import moon_figure, lonlat_to_xyz, xyz_to_lonlat

# Reference points (lon, lat, label). Values from the LP-GRS / lunar literature.
# The near-side maria sit around the prime meridian; the poles and the +90 lon
# point (eastern limb) frame the coordinate system.
MARKERS = [
    (0.0, 0.0, "0°, 0° (sub-Earth)"),
    (0.0, 90.0, "North pole"),
    (0.0, -90.0, "South pole"),
    (90.0, 0.0, "90°E equator"),
    (-90.0, 0.0, "90°W equator"),
    (23.4, 0.7, "Mare Tranquillitatis"),   # near Apollo 11
    (-23.0, -3.0, "Mare Nubium"),
]

# ---- Sanity check: the coordinate mapping round-trips --------------------
print("Coordinate round-trip check (lon, lat -> xyz -> lon, lat):")
for lon, lat, label in MARKERS:
    x, y, z = lonlat_to_xyz(lon, lat)
    lon2, lat2 = xyz_to_lonlat(x, y, z)
    ok = np.allclose([lon, lat], [float(lon2), float(lat2)], atol=1e-6)
    print(f"  {label:24s} ({lon:6.1f}, {lat:6.1f}) -> "
          f"({float(lon2):6.1f}, {float(lat2):6.1f})  {'OK' if ok else 'MISMATCH'}")

# ---- Build and show the interactive globe -------------------------------
fig = moon_figure(markers=MARKERS)
fig.show()
