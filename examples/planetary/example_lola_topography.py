"""Minimum working example: LRO LOLA topography + composition correlation.

Downloads the LOLA global cylindrical DEM (4 pixels/degree, ~2 MB, cached in
``wara/planetary/data/``) and:

1. maps the global elevation (the South Pole-Aitken basin, the far-side
   highlands, and the flat near-side maria are unmistakable);
2. correlates topography with composition: FeO abundance (LP-GRS Level 5)
   vs. LOLA elevation per abundance pixel. The two lunar terranes separate
   cleanly — low-lying maria are iron-rich basalt, high-standing highlands
   are iron-poor anorthosite.

Run it with::

    python example_lola_topography.py
"""
import matplotlib.pyplot as plt
import numpy as np

from wara.planetary import (download_abundance, download_lola_dem,
                            elevation_grid, read_abundance, read_lola_dem)

download_lola_dem(4)
dem = read_lola_dem(4)
elev = dem["elev_km"]
print(f"LOLA DEM: {elev.shape}, {elev.min():.1f} .. {elev.max():.1f} km")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                               gridspec_kw={"width_ratios": [1.6, 1]})

im = ax1.imshow(elev, origin="upper", extent=(0, 360, -90, 90),
                cmap="terrain", aspect="auto")
fig.colorbar(im, ax=ax1, label="Elevation (km)")
ax1.set_xlabel("Longitude (deg east)")
ax1.set_ylabel("Latitude (deg)")
ax1.set_title("LRO LOLA global topography (4 px/deg)")

# ---- Composition vs topography --------------------------------------------
download_abundance(5)
table = read_abundance(5)
# Mean elevation of each abundance pixel (sampled at the pixel centers).
lon_c = (table["lon_w"] + table["lon_e"]) / 2.0
lat_c = (table["lat_s"] + table["lat_n"]) / 2.0
elev_c = np.array([
    elevation_grid(dem, np.array([lo]), np.array([la]))[0, 0]
    for lo, la in zip(lon_c, lat_c)])

sc = ax2.scatter(elev_c, table["FeO"] * 100.0, s=6, c=table["Th"],
                 cmap="inferno")
fig.colorbar(sc, ax=ax2, label="Th (ppm)")
ax2.set_xlabel("LOLA elevation (km)")
ax2.set_ylabel("FeO (wt %)")
ax2.set_title("Composition vs topography (5° pixels)")
fig.tight_layout()

r = np.corrcoef(elev_c, table["FeO"])[0, 1]
print(f"FeO-elevation correlation: r = {r:.2f} "
      "(maria are low and iron-rich, highlands high and iron-poor)")
plt.show()
