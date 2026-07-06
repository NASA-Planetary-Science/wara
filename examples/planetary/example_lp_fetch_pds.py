"""Minimum working example: pull Lunar Prospector GRS data from the NASA PDS.

This is step 2 of the WARA Planetary tab. It queries the live LP-GRS archive
at the PDS Geosciences node, shows what is available, filters the products by
measurement date (which also fixes the orbit phase: ~100 km before 1998-12-19,
~30-40 km after), downloads one day's worth of data, and plots the summed
gamma spectrum together with that day's ground track on the Moon.

Run it with::

    python example_lp_fetch_pds.py

Requires internet access on the first run; the downloaded files (~1 MB/day)
are cached in ``LP_data/`` next to this script and re-used afterwards.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from wara.planetary import list_grs_products, filter_products, download_products
from wara.planetary import read_grs_day

DATA_DIR = Path(__file__).parent / "LP_data"

# ---- 1) Search the archive ------------------------------------------------
print("Listing the LP-GRS archive on the PDS (one product per mission day)...")
products = list_grs_products()
print(f"  {len(products)} daily products available, "
      f"{products[0].day} .. {products[-1].day}")
n_high = sum(p.phase == "high" for p in products)
print(f"  {n_high} days in the ~100 km mapping orbit, "
      f"{len(products) - n_high} days in the low (~30-40 km) orbit")

# ---- 2) Filter by date ------------------------------------------------------
# First three days of GRS science data, all in the 100 km phase.
selected = filter_products(products, start="1998-01-16", end="1998-01-18")
print(f"\nSelected {len(selected)} product(s):")
for p in selected:
    print(f"  {p.stem}  ({p.day}, {p.phase}-altitude phase)")

# The same call can search the low-altitude phase instead, e.g.:
#   filter_products(products, phase="low", start="1999-03-01", end="1999-03-07")

# ---- 3) Download (label + data pairs, skipped if already cached) -----------
xml_paths = download_products(
    selected[:1], DATA_DIR,
    progress=lambda name, i, n: print(f"  downloading {name} ({i}/{n})"),
)

# ---- 4) Read and plot -------------------------------------------------------
day = read_grs_day(xml_paths[0])
print(f"\n{day.stem}: {day.n_records} records of ~32 s each")
print(f"  altitude   {day.altitude_km.min():.1f} .. {day.altitude_km.max():.1f} km")
print(f"  latitude   {day.latitude.min():.1f} .. {day.latitude.max():.1f} deg")
print(f"  longitude  {day.longitude.min():.1f} .. {day.longitude.max():.1f} deg east")

spectrum, n = day.sum_spectrum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(spectrum, drawstyle="steps-mid", color="C0")
ax1.set_yscale("log")
ax1.set_xlabel("Channel")
ax1.set_ylabel("Counts")
ax1.set_title(f"LP-GRS {day.day} — sum of {n} accepted spectra")

sc = ax2.scatter(day.longitude, day.latitude, c=day.altitude_km, s=3, cmap="viridis")
fig.colorbar(sc, ax=ax2, label="Altitude (km)")
ax2.set_xlim(-180, 180)
ax2.set_ylim(-90, 90)
ax2.set_xlabel("Longitude (deg east)")
ax2.set_ylabel("Latitude (deg)")
ax2.set_title("Sub-spacecraft ground track")
fig.tight_layout()
plt.show()
