"""Minimum working example: the bundled LP-GRS orbit metadata.

wara ships a small CSV (``wara/planetary/data/lp_grs_metadata.csv``) with the
sub-spacecraft ephemeris (UTC, product, lon, lat, altitude) of *every* daily
Lunar Prospector GRS product, scraped once from the PDS. It answers "what data
exists, where, and when" without downloading a single product:

1. the whole-mission ground track (lat/lon coverage);
2. altitude vs. time — the ~100 km mapping orbit and the two orbit lowerings
   of the extended mission (~40 km, then ~30 km) are obvious;
3. per-day product availability.

Run it with::

    python example_lp_orbit_metadata.py

(The CSV is rebuilt with ``wara.planetary.build_orbit_metadata()`` — a one-time
~5 GB scrape of the PDS archive that keeps only the ephemeris.)
"""
import matplotlib.pyplot as plt
import numpy as np

from wara.planetary import load_orbit_metadata

md = load_orbit_metadata()
n_products = len(set(md["product"]))
d0 = np.datetime_as_string(md["utc"].min(), unit="D")
d1 = np.datetime_as_string(md["utc"].max(), unit="D")
print(f"{len(md['utc'])} metadata points from {n_products} daily products, "
      f"{d0} .. {d1}")
print(f"altitude {md['alt_km'].min():.1f} .. {md['alt_km'].max():.1f} km")

t_days = (md["utc"] - md["utc"].min()) / np.timedelta64(1, "D")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8),
                               gridspec_kw={"height_ratios": [2, 1]})

sc = ax1.scatter(md["lon"], md["lat"], c=t_days, s=1, cmap="viridis")
fig.colorbar(sc, ax=ax1, label=f"days since {d0}")
ax1.set_xlim(-180, 180)
ax1.set_ylim(-90, 90)
ax1.set_xlabel("Longitude (deg east)")
ax1.set_ylabel("Latitude (deg)")
ax1.set_title(f"Lunar Prospector GRS — whole-mission ground track "
              f"({n_products} days)")

ax2.plot(t_days, md["alt_km"], ".", ms=1, color="C0")
ax2.set_xlabel(f"Days since {d0}")
ax2.set_ylabel("Altitude (km)")
ax2.set_title("Orbit altitude: 100 km mapping orbit, then the "
              "low-altitude extended mission")
fig.tight_layout()
plt.show()
