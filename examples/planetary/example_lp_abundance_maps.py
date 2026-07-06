"""Minimum working example: LP-GRS Level-5 elemental abundance maps.

Besides the raw daily spectra (Level 3 RDR), the PDS hosts the *derived*
("calibrated") Lunar Prospector GRS product: elemental abundances of the
lunar surface on an equal-area grid (lp-l-grs-5-elem-abundance-v1), at 2, 5,
and 20 degree resolution — oxide weight fractions (MgO, Al2O3, SiO2, CaO,
TiO2, FeO) and K/Th/U in ppm.

This example downloads the 5-degree table (~1.6 MB, cached in
``wara/planetary/data/``) and maps thorium and iron: the Th map shows the
famous KREEP-rich Procellarum-Imbrium terrane on the near side.

Run it with::

    python example_lp_abundance_maps.py
"""
import matplotlib.pyplot as plt
import numpy as np

from wara.planetary import download_abundance, read_abundance, abundance_grid

DEG = 5
download_abundance(DEG)
table = read_abundance(DEG)
print(f"{len(table['lat_s'])} equal-area pixels at {DEG} deg resolution")
print(f"Th:  {table['Th'].min():.3f} .. {table['Th'].max():.2f} ppm")
print(f"FeO: {table['FeO'].min():.3f} .. {table['FeO'].max():.3f} wt. fraction")

# Sample both maps onto a fine regular grid for plotting.
lon = np.linspace(-180.0, 180.0, 721)
lat = np.linspace(-90.0, 90.0, 361)
fig, axes = plt.subplots(2, 1, figsize=(11, 9))
for ax, (elem, unit, cmap) in zip(axes, [("Th", "ppm", "inferno"),
                                         ("FeO", "wt. fraction", "viridis")]):
    grid = abundance_grid(table, elem, lon, lat)
    im = ax.imshow(grid, origin="lower", extent=(-180, 180, -90, 90),
                   cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, label=f"{elem} ({unit})")
    ax.set_xlabel("Longitude (deg east)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Lunar Prospector GRS — {elem} abundance ({DEG}° pixels)")
fig.tight_layout()
plt.show()
