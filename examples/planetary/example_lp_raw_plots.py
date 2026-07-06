"""Minimum working example: understand the raw LP-GRS data (no network).

Step 3 of the WARA Planetary tab. Reads one day of Lunar Prospector GRS data
from a local file and makes the basic plots that explain what is in a product:

1. the summed 512-channel gamma spectrum, accepted vs. coincidence-rejected;
2. the day's ground track (sub-spacecraft lon/lat) colored by count rate;
3. a regional selection — the spectrum from a (lat, lon) box compared against
   the full-day average, which is the seed of the Planetary tab's
   "click a region on the Moon, see its spectrum" workflow.

It uses the sample product bundled with the repo; point ``XML_FILE`` at any
file downloaded by ``example_lp_fetch_pds.py`` to look at a different day.

Run it with::

    python example_lp_raw_plots.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wara.planetary import read_grs_day

# Bundled sample: first day of GRS science data (1998, DOY 016), 100 km orbit.
# Falls back to anything already downloaded by example_lp_fetch_pds.py.
_REPO_SAMPLE = (Path(__file__).parents[2] / "wara" / "planetary-nuclear-spect"
                / "LP" / "LP_data" / "1998_016_grs.xml")
_DOWNLOADED = sorted((Path(__file__).parent / "LP_data").glob("*_grs.xml"))
XML_FILE = _REPO_SAMPLE if _REPO_SAMPLE.exists() else _DOWNLOADED[0]

day = read_grs_day(XML_FILE)
print(f"{day.stem}: {day.n_records} records (~32 s each) on {day.day}")
print(f"  spectra shape {day.spectra.shape} (records x channels)")
print(f"  altitude {day.altitude_km.min():.1f}..{day.altitude_km.max():.1f} km, "
      f"deadtime {day.deadtime.mean() * 100:.2f}% mean, "
      f"detector {day.temperature.mean():.1f} degC mean")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# ---- 1) Accepted vs rejected summed spectra -------------------------------
ax = axes[0, 0]
accepted, n = day.sum_spectrum()
rejected, _ = day.sum_spectrum(rejected=True)
ax.plot(accepted, drawstyle="steps-mid", label="Accepted", color="C0")
ax.plot(rejected, drawstyle="steps-mid", label="Rejected (anticoincidence)",
        color="C3", alpha=0.7)
ax.set_yscale("log")
ax.set_xlabel("Channel")
ax.set_ylabel("Counts")
ax.set_title(f"Full day, {n} records summed")
ax.legend()

# ---- 2) Ground track colored by total count rate ---------------------------
ax = axes[0, 1]
counts_per_record = day.spectra.sum(axis=1)
sc = ax.scatter(day.longitude, day.latitude, c=counts_per_record, s=3,
                cmap="inferno")
fig.colorbar(sc, ax=ax, label="Counts / record")
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude (deg east)")
ax.set_ylabel("Latitude (deg)")
ax.set_title("Ground track, total counts per ~32 s record")

# ---- 3) Regional selection: a lat/lon box vs the whole day -----------------
# LP flies a polar orbit and the Moon rotates only ~13 deg/day beneath it, so
# a single day samples every latitude but just two narrow longitude bands
# (ascending and descending passes). Center the demo box on the day's
# most-sampled longitude so the selection is guaranteed to contain records;
# selecting an arbitrary region (e.g. Mare Imbrium) needs several weeks of
# products summed together.
hist, edges = np.histogram(day.longitude, bins=72, range=(-180, 180))
lon0 = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
LAT_RANGE, LON_RANGE = (25.0, 45.0), (lon0 - 15.0, lon0 + 15.0)
mask = day.select(lat_range=LAT_RANGE, lon_range=LON_RANGE)
region, n_region = day.sum_spectrum(mask)
print(f"  region lat {LAT_RANGE}, lon {LON_RANGE}: {n_region} records")

ax = axes[1, 0]
# Per-record averages so the two curves are comparable despite different N.
ax.plot(accepted / n, drawstyle="steps-mid", label=f"Whole day ({n} rec)",
        color="C0", alpha=0.8)
if n_region:
    ax.plot(region / n_region, drawstyle="steps-mid",
            label=f"Region ({n_region} rec)", color="C1")
ax.set_yscale("log")
ax.set_xlabel("Channel")
ax.set_ylabel("Counts / record")
ax.set_title("Regional spectrum vs full-day average")
ax.legend()

ax = axes[1, 1]
ax.scatter(day.longitude, day.latitude, s=2, color="0.7", label="All records")
ax.scatter(day.longitude[mask], day.latitude[mask], s=4, color="C1",
           label="Selected region")
ax.add_patch(plt.Rectangle((LON_RANGE[0], LAT_RANGE[0]),
                           LON_RANGE[1] - LON_RANGE[0],
                           LAT_RANGE[1] - LAT_RANGE[0],
                           fill=False, edgecolor="C1", lw=1.5))
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel("Longitude (deg east)")
ax.set_ylabel("Latitude (deg)")
ax.set_title("Records inside the selection box")
ax.legend(loc="lower left")

fig.suptitle(f"Lunar Prospector GRS — raw data tour, {day.stem}")
fig.tight_layout()
plt.show()
