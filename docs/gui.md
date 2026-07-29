# The GUI

Everything in the User Guide is also available through wara's graphical
interface. As of **v2.0** the redesigned GUI is the default — a three-column
layout with a **navigation rail**, a contextual **options panel**, and the
**plot area**.

## Launching

```bash
wara              # the GUI
```

You can also open
a file and preconfigure the peak search from the command line — see the
[Quickstart](quickstart.md#command-line-options).

## The Spectrum tab

The **Spectrum** tab is the hub: open a spectrum, find and fit peaks, and
identify isotopes. Click **Open Spectrum**, expand **Auto-Find Peaks** and pick a
detector preset, then use **Drag and fit** to fit lines by dragging across them.

![Spectrum tab](../figs/GUI-spectrum.png)

The same workflow on an HPGe spectrum, where the high resolution and Hypermet
fitting matter:

![Spectrum tab — HPGe](../figs/GUI-spectrum-HPGe.png)

Dragging across peaks opens the fit, with per-peak components, residuals, and the
chosen background model:

![Peak fitting](../figs/GUI-peakfit.png)

These map to [Peak Finding](peak_finding.md) and [Peak Fitting](peak_fitting.md).

## The Nuclear Database & isotope identification

A core feature of the Spectrum tab is identifying *which isotope* produced each
peak. wara ships several gamma-line **databases** and ranks the most likely
source isotope for every fitted energy.

![Nuclear database](../figs/GUI-database.png)

The bundled databases (the same list the GUI shows) are:

| Database | Contents |
|----------|----------|
| **Common lab sources** | Calibration/check sources (e.g. ¹³⁷Cs, ⁶⁰Co, ²²Na) |
| **Delayed activation (IAEA)** | Activation-product decay lines |
| **Neutron capture (CapGam)** | (n,γ) capture gamma lines |
| **Neutron capture (IAEA)** | (n,γ) capture gamma lines (IAEA compilation) |
| **Neutron capture (ENDF/B-VII.1)** | (n,γ) capture lines from ENDF/B-VII.1 |
| **Inelastic 2.45 MeV (ENDF/B-VII.1)** | (n,n′γ) lines at 2.45 MeV (D-D neutrons) |
| **Inelastic (Baghdad)** | (n,n′γ) inelastic-scattering lines |
| **TALYS 14 MeV** | Computed 14.1 MeV reaction lines |
| **Inelastic 14 MeV (ENDF/B-VII.1)** | (n,n′γ) lines at 14 MeV (D-T neutrons) |
| **(n,2n) 14 MeV (ENDF/B-VII.1)** | (n,2n) reaction lines at 14 MeV |
| **(n,p) 14 MeV (ENDF/B-VII.1)** | (n,p) reaction lines at 14 MeV |
| **(n,a) 14 MeV (ENDF/B-VII.1)** | (n,α) reaction lines at 14 MeV |
| **Natural radiation** | Naturally occurring lines (⁴⁰K, U/Th series) |

### How identification works

For each peak energy, candidate isotopes are ranked using each library's own line
strengths (cross section or intensity), then refined with:

- **A resolution-matched energy window with proximity weighting** — a database
  line matches an observed peak only within an energy window sized to the
  detector resolution (about twice the FWHM). That width is derived from the
  peak-finder's resolution model and the spectrum's **energy calibration** (the
  keV-per-channel dispersion), so a coarse LaBr resolution opens the window wider
  than a sharp HPGe one. Within the window, matches are weighted by proximity — a
  Gaussian in the energy difference — so a line 0.1 keV away vastly outranks one
  several keV away even though both are "inside". A good energy calibration
  therefore tightens the window and sharpens the ranking. (Without a calibrated
  resolution model, a generous fallback window is used.)
- **Natural isotopic abundance** — for reaction-on-natural-target libraries
  (capture, inelastic, TALYS), the line yield scales with the target isotope's
  abundance. This is why a 846.8 keV line in natural iron is identified as ⁵⁶Fe
  (≈92 % abundant), not ⁵⁷Fe, even though ⁵⁷Fe's bare cross section is larger.
- **A terrestrial element-naturalness prior** — common rock/air/biological
  elements (O, Si, Fe, H, C, N…) are favored over rare ones.
- **A half-life procurability prior (check sources)** — for the lab-source
  library, candidates are weighted toward nuclides long-lived enough to exist as
  a real check source (¹³⁷Cs, ⁶⁰Co, ²⁴¹Am…) over short-lived activation/fission
  products that couldn't be bought and kept.
- **Corroborating lines and escape peaks** — other lines of the same isotope, and
  the single/double-escape peaks (E − 511, E − 1022 keV) of high-energy gammas,
  boost a candidate. For example ¹⁶O's 6129 keV line together with its escapes
  and its 6917 / 3685 keV lines pin the assignment.

Source/decay libraries (lab sources, natural radiation, delayed activation) carry
specific radionuclides rather than sample elements, so the abundance and element
priors are not applied to them; the half-life prior applies only to the
lab-source library.

**How the percentage is assigned.** For each peak, the factors above are combined
into a single score per candidate, and the scores of all candidates competing at
that energy are normalised to sum to 100 % **within each database**. The
percentage shown next to a candidate is therefore its share of the evidence among
the alternatives *in that library* — a relative likelihood, not an absolute
confidence that the assignment is correct. Each database reports its own
independently-normalised percentages (so the same peak can read "98 %" in one
library and "60 %" in another), and candidates that round to 0 % are dropped. The
scriptable `rank_candidates` (see below) instead pools the libraries and
normalises across them, giving one set of probabilities comparable between
databases.

### The Isotope ID panel

Expand **Isotope ID** (a drop-down, like **Auto-Find Peaks**) to turn on
hover identification and tailor it to your experiment:

- **Enable Isotope ID** — with a calibrated spectrum and found peaks, hover any
  peak to see the most likely isotope from each selected database, with a
  probability.
- **Databases** — tick only the libraries relevant to your measurement (use
  **All** / **None** to select quickly). Deselected libraries are excluded from
  both the hover ranking and the export.
- **Atomic number (Z)** — restrict candidates to a `Z from … to …` range. The
  full range (1–118) considers every element; narrow it when you *know* certain
  elements can't be present (e.g. exclude Gd, Z = 64) to remove false matches.
- **Export table…** — save a CSV of the identified isotopes: the top candidate
  from each selected database for every found peak. Each row carries the energy,
  database, isotope (parent) and element, the line strength (cross section or
  intensity, with its unit), and — for decay libraries — the decay mode and the
  daughter isotope that actually emits the gamma (e.g. ¹³⁷Cs → ¹³⁷Ba), plus the
  probability, matched line, and number of corroborating lines seen. Honours the
  current database and Z-range selection.

### From Python

The same engine is available programmatically in
{py:mod}`wara.nuclide_identificator`:

```python
from wara.nuclide_identificator import identify_best, rank_candidates

energies = [511.0, 846.8, 1460.8]   # keV, e.g. fitted peak centroids

identify_best(energies)        # single most likely isotope per energy
rank_candidates(energies)      # ranked candidates across all databases
```

`rank_candidates` returns comparable probabilities *across* libraries;
`identify` (not shown) returns the top candidates *within* each library
separately. Restrict the search with `databases=[...]`, limit candidates to an
atomic-number span with `z_range=(z_min, z_max)` (the same control as the panel's
Z-range), and set the matching window with `tol` (a constant in keV or a callable
`tol(energy)`).

## Calibration, Efficiency, and Resolution tabs

The next three rail entries wrap the calibration workflows. Send fitted peaks
over from the Spectrum tab; for energy calibration you can look up reference
energies from the nuclear database described above.

![Calibration tab](../figs/GUI-calibration.png)

See [Energy Calibration](calibration.md), [Efficiency
Calibration](efficiency.md), and [Resolution](resolution.md).

## The API tab

The **API** tab is the interactive front end for Associated Particle Imaging:
load a run's parquet data into linked energy/time/X–Y panels, apply live filters,
send a selected spectrum back to the Spectrum tab, and explore the reconstructed
3D hit cloud.

![API tab](../figs/GUI-API.png)

See [Associated Particle Imaging](api.md).

## The Neutrons tab

The **Neutrons** tab is an interactive pulse-shape-discrimination (PSD) explorer
for digitized PMT traces (a PicoScope `.npz`/`.npy`/`.txt`/`.csv` file, or a
PIXIE-16 run loaded straight from disk). It shows three linked panels:

- **Traces** — a random sample of baseline-corrected pulses with draggable
  markers for the voltage threshold and the gate-start / prompt-end / tail-end
  boundaries. Dragging a marker recomputes the energy and PSD of every pulse.
- **MCA** — the pulse-integral (energy) spectrum; drag a horizontal span to
  restrict the other panels.
- **PSD** — a 2-D energy-vs-PSD histogram; drag a box to restrict the other
  panels. Arming **PSD selections** switches to multi-box mode, where each box
  gets its own colour in the Traces and MCA panels.

### Averaging the traces

The **Average trace only** checkbox (DISPLAY section, off by default) replaces
the individual pulses with a single averaged trace:

- normally, the mean of the random sample currently drawn (the *Traces* count
  set just above the checkbox — 150 by default);
- with **PSD selections** ON, one averaged trace per coloured box, drawn in that
  box's colour and averaged over *every* pulse inside it, not just the sampled
  ones. This is the quickest way to compare the mean pulse shape of a
  neutron-like against a gamma-like region.

The panel title reports how many pulses went into the average. A runnable
version of both modes is in
`examples/other/example_neutron_average_trace.py`; the underlying numerics live
in {py:mod}`wara.neutron_psd` (see `examples/other/example_neutron_psd.py`).

### Figure of merit

The **Figure of merit** button (SELECTION section) re-arms the PSD rubber band:
instead of cross-filtering, the box you drag selects the slice to characterise,
and the result opens in its own window. The main canvas is untouched.

Drag a box on the PSD panel that **spans both the gamma and the neutron band**
over the energy slice you want to characterise (discrimination varies strongly
with light output, so the FOM is always quoted for a slice). The pulses inside
are projected onto the PSD axis, histogrammed into **FOM bins** (80 by default)
and fitted with a double Gaussian

```
y(PSD) = A_γ·exp(−(PSD − μ_γ)² / 2σ_γ²) + A_n·exp(−(PSD − μ_n)² / 2σ_n²)
```

whose lower-mean component is the gamma band and whose higher-mean component is
the neutron band. Each width is converted with `FWHM = 2√(2 ln 2)·σ ≈ 2.3548·σ`
and reduced to

```
FOM = S / (FWHM_γ + FWHM_n) = |μ_n − μ_γ| / (FWHM_γ + FWHM_n)
```

The pop-out window (with its own matplotlib toolbar, so the fit can be zoomed
and saved) shows the projection, both components, their sum, and a read-out of
the FOM, the separation `S`, the summed FWHM, the pulse count and the fit R².
The read-out is green when `FOM ≥ 1.27` — the usual threshold for clean
neutron/gamma separation — and red below it; the options panel repeats the value
next to **FOM**.

Notes:

- The FOM box only sets the projected slice; unlike the normal selection box it
  does not cross-filter the Traces and MCA panels. It stays outlined in green
  after the drag, and re-dragging replaces it and refreshes the window.
- The window is non-modal and stays on top, so the selection can be adjusted
  while it is open. Closing it does not disarm the mode — the next box reopens
  it — while disarming **Figure of merit** closes it.
- **Figure of merit** and **PSD selections** are mutually exclusive — arming one
  disarms the other, since both own the PSD rubber band.
- Moving a gate marker re-fits the standing slice with the new PSD values, as
  does changing **FOM bins**. Loading a new file or run disarms the mode.
- Fit failures (an empty or single-band slice) are reported in the window
  instead of a plot.

The numerics are {py:func}`wara.neutron_psd.figure_of_merit` (and
{py:meth}`wara.neutron_psd.NeutronTraces.fom` for a ready-made dataset), which
return a `FOMResult` carrying the fitted parameters, the histogram and the
`curve()` / `component()` helpers. A runnable version is in
`examples/other/example_neutron_fom.py`.

## The Planetary tab

The **Planetary** tab visualizes and analyzes data from NASA planetary
gamma-ray/neutron spectrometer missions, starting with the **Lunar Prospector
GRS**. The canvas is split vertically: on top, a fully interactive
high-resolution 3D Moon with the correct latitude/longitude coordinates
(hover anywhere to read lon/lat); below it, the gamma spectrum for the
selected region.

The left menu drives the workflow:

1. **Search PDS** — query the NASA PDS archive for daily LP-GRS products by
   measurement date and orbit phase (~100 km mapping orbit vs the ~30–40 km
   extended mission).
2. **Download** — fetch the matching products (~11 MB per day) into
   `wara/planetary/data/` (configurable; kept between sessions and ignored by
   git), so datasets you decide to keep are never re-scraped from the web.
   Already-downloaded files are skipped.
3. **Load into memory** — read the cached products; every ~32 s record carries
   its own sub-spacecraft coordinates and altitude.
4. **Arm "Select region", then click the Moon** — a lat/lon box (configurable
   half-width) is drawn at the clicked point and the spectra of all records
   inside it are summed and plotted, optionally against the scaled all-data
   average. Each selection disarms the button again, so orbiting the globe
   never selects by accident.
5. While **Keep spectra** is checked, selecting a new region keeps the
   previous spectrum visible so several regions can be compared: each kept
   spectrum and its box on the Moon share a color. Untick it to drop the kept
   spectra. **Send to spectrum** hands the regional spectrum to the Spectrum
   tab for peak finding and fitting.
6. **Show orbit path** overlays the spacecraft ground track on the globe,
   colored by measurement time (hover a point for the exact UTC date,
   position, and altitude); untick to hide it. It uses the loaded records
   when data is in memory — otherwise it falls back to the bundled
   whole-mission orbit metadata (`wara/planetary/data/lp_grs_metadata.csv`,
   the sub-spacecraft lat/lon/altitude of every daily product, scraped once
   from the PDS), so LP coverage is visible without downloading anything.
   In metadata mode the path follows the date-range and orbit controls and
   redraws when they change (leave the dates empty to see the whole mission).
   The same metadata drives the "products available / downloaded" summary at
   the top of the options panel.
7. **Landmarks** labels important reference points on the
   Moon — the major maria, landmark craters, far-side basins, the poles, and
   the Apollo 11 site; tick to show them.
8. The **Dataset** dropdown switches between the raw Level-3 spectra and the
   **calibrated Level-5 elemental abundance maps**
   (`lp-l-grs-5-elem-abundance-v1`): pick an element (Th, K, U, FeO, TiO₂,
   MgO, Al₂O₃, SiO₂, CaO), a pixel size (2°/5°/20°), and a colormap, and the
   map is draped over the Moon with a colorbar. An **Opacity** control makes
   the composition map semi-transparent, so the base layer shows through —
   the grey albedo Moon, or the LOLA elevation colors if that was the
   previously selected dataset (both colorbars stay visible) — the Th map shows the KREEP-rich
   Procellarum–Imbrium terrane. Switching back to Raw restores the albedo
   Moon. Tables download once (~1–4 MB) into the planetary data folder.

9. **Topography (LRO LOLA)** — the Dataset dropdown's **Elevation (LOLA)**
   entry drapes the LOLA global elevation model over the Moon as a color
   map, and the **3D topography** checkbox displaces the globe surface by
   the real terrain (adjustable vertical exaggeration; the DEM downloads
   once, ~2 MB). The relief works with every drape, so **Calibrated +
   3D topography** shows composition over terrain — e.g. the iron-rich,
   low-lying maria vs. the iron-poor highlands.
10. **2D map (equirectangular)** switches to a flat longitude/latitude map
    that shows the Moon at the **native resolution of the texture** (an
    image, not a mesh — much sharper than the globe). Drapes, landmarks,
    orbit paths, selection boxes, and click-to-select all keep working, and
    the axes zoom/pan like any 2D plot; untick to return to the globe.
    (3D relief applies only to the globe.)
11. **Mission info…** opens a browser for the mission's PDS documentation —
   the archive overview, mission/spacecraft/instrument descriptions, the
   GRS/NS data-set documents, the data-products summary, and the reference
   list — fetched once from the PDS Geosciences node and cached locally, with
   a link to the [LP reduced GRS/NS mission
   page](https://pds-geosciences.wustl.edu/missions/lunarp/reduced_grsns.html).

The same backend is scriptable from Python via `wara.planetary`
(`list_grs_products`, `filter_products`, `download_products`, `read_grs_day`,
`download_abundance`, `read_abundance`, `abundance_grid`).

## Tips

- **Tooltips** are placed throughout the GUI — hover over any control for
  contextual guidance.
- Most tabs **hand data to one another** — fitted peaks flow from Spectrum to
  Calibration, Efficiency, and Resolution; spectra flow from the API tab back to
  Spectrum.
