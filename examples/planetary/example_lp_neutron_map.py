"""Minimum working example: Lunar Prospector neutron counts (LP-NS).

Besides the gamma-ray spectra, Lunar Prospector carried a Neutron
Spectrometer. Its reduced time series lives in the same PDS bundle as the
LP-GRS products, in the ``ns/`` collection — but as **whole-mission arrays**
rather than one product per day, with the ephemeris in a separate
``position_*`` file that pairs index by index with the counts.

This example downloads the epithermal / high-orbit / 32 s pair (~11 MB, cached
in ``wara/planetary/data/``), bins it into a map, and plots it next to the
zonal (mean-vs-latitude) profile. Epithermal counts dip toward both poles:
that is neutrons moderated by hydrogen in the permanently shadowed polar
regions — the LP-NS result that put water ice on the map.

Run it with::

    python example_lp_neutron_map.py
"""
import matplotlib.pyplot as plt
import numpy as np

from wara.planetary import (
    download_ns,
    neutron_bins,
    read_ns,
    region_stats,
    zonal_profile,
)

KIND, PHASE, CADENCE = "epithermal", "high", 32
BIN_DEG = 2.0

download_ns(KIND, PHASE, CADENCE, progress=lambda name, i, n:
            print(f"  downloading {name} ({i}/{n})"))
data = read_ns(KIND, PHASE, CADENCE)

print(f"{data.label}: {data.n_samples} accumulations")
print(f"  counts   {data.counts.min():.1f} .. {data.counts.max():.1f} "
      f"(mean {data.counts.mean():.1f})")
print(f"  altitude {data.altitude_km.min():.1f} .. "
      f"{data.altitude_km.max():.1f} km")
print(f"  covering {data.utc64[0]} .. {data.utc64[-1]}")

# Both poles vs the equator: the hydrogen signature, in one number each.
for lo, hi, name in ((-90, -80, "south pole"), (-10, 10, "equator"),
                     (80, 90, "north pole")):
    stats = region_stats(data, data.select(lat_range=(lo, hi)))
    print(f"  {name:11s} {stats['mean']:7.2f} +/- {stats['sem']:.2f} counts "
          f"({stats['n']} accumulations)")

grid, lon_edges, lat_edges = neutron_bins(data, bin_deg=BIN_DEG)
lat, mean, sem, n = zonal_profile(data, bin_deg=BIN_DEG)

# The derived thermal/epithermal ratio is a product like any other: both
# components are sampled on the same accumulations (they share one position
# file), so it divides exactly, sample by sample. It downloads the thermal
# counts alongside what is already cached.
download_ns("thermal/epithermal", PHASE, CADENCE)
ratio = read_ns("thermal/epithermal", PHASE, CADENCE)
r_lat, r_mean, r_sem, _ = zonal_profile(ratio, bin_deg=BIN_DEG)
print(f"{ratio.label}: {np.nanmean(ratio.counts):.3f} on average, "
      f"{np.nanmean(r_mean[:2]):.3f} over the south pole and "
      f"{np.nanmean(r_mean[-2:]):.3f} over the north")

fig, (ax_map, ax_prof) = plt.subplots(
    2, 1, figsize=(11, 8), gridspec_kw=dict(height_ratios=[2, 1]))

im = ax_map.imshow(grid, origin="lower", extent=(-180, 180, -90, 90),
                   cmap="viridis", aspect="auto",
                   vmin=np.nanpercentile(grid, 1),
                   vmax=np.nanpercentile(grid, 99))
fig.colorbar(im, ax=ax_map, label=f"counts / {CADENCE} s")
ax_map.set_title(f"{data.label} — {BIN_DEG:g}° bins")
ax_map.set_xlabel("Longitude (deg east)")
ax_map.set_ylabel("Latitude (deg)")

ax_prof.plot(lat, mean, color="tab:blue", lw=1.5, label=KIND)
ax_prof.fill_between(lat, mean - sem, mean + sem, color="tab:blue", alpha=0.3,
                     linewidth=0)
ax_prof.set_xlim(-90, 90)
ax_prof.set_xticks(range(-90, 91, 30))
ax_prof.set_xlabel("Latitude (deg)")
ax_prof.set_ylabel(f"Mean counts / {CADENCE} s")
ax_prof.grid(alpha=0.3)

# The ratio rises toward both poles for the same reason the epithermal counts
# fall there: hydrogen. Plot it on a twin axis, in its own units.
ax_ratio = ax_prof.twinx()
ax_ratio.plot(r_lat, r_mean, color="tab:orange", lw=1.5,
              label="thermal / epithermal")
ax_ratio.set_ylabel("Thermal / epithermal ratio")
lines = ax_prof.get_lines() + ax_ratio.get_lines()
ax_prof.legend(lines, [ln.get_label() for ln in lines], loc="lower center",
               ncol=2, fontsize=9)

fig.tight_layout()
plt.show()
