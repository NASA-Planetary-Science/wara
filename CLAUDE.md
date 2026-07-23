# wara

Python package for **gamma-ray and neutron spectroscopy** analysis and
visualization: spectrum smoothing/rebinning, peak search, multi-peak Gaussian
fitting, energy/efficiency/resolution calibration, nuclide identification
(neutron-induced gamma emphasis), PIXIE/Associated Particle Imaging (API) data,
and planetary nuclear spectroscopy. Ships a PyQt5 GUI (entry point: `wara`).

## Commands

```bash
pip install -e .            # editable install (run from repo root)
wara                        # launch the v2.0 GUI (default)
wara --legacy               # launch the pre-2.0 GUI (frozen, pending removal)
pytest tests -q             # full test suite
pytest tests/test_spectrum.py   # a single test file
ruff check .                # lint
```

Requires Python >= 3.10 (hard floor: PEP 604 `str | None` unions are evaluated
at import time in `matplotlib_theme.py` and `apicalc.py`).

## Layout

- `wara/` — package root. `__init__.py` re-exports the public API
  (`Spectrum`, `peakfit`, `PeakSearch`). CLI dispatch lives in `cli.py`.
- `wara/gui/` — the current PyQt5 GUI. `app.py` wires the nav rail; each tab is
  its own module (`spectrum.py`, `calibration.py`, `efficiency.py`,
  `resolution.py`, `api*.py`, `neutrons.py`, `planetary.py`, `fitting.py`,
  `isotope_id.py`, ...). Shared widgets/theme in `widgets.py`, `theme.py`.
- `wara/gui_legacy/` — **frozen**, pending removal. Excluded from ruff. Don't
  add features here; only touch if explicitly asked.
- `wara/nuclear-data/` — bundled `.txt`/`.csv` nuclide databases (package data).
- `wara/planetary/`, `wara/planetary-nuclear-spect/` — planetary spectroscopy.
  `planetary-nuclear-spect` holds ad-hoc mission-data scripts (not importable
  package code); excluded from ruff.
- `tests/` — pytest suite; `test_gui_*.py` cover the GUI tabs.
- `examples/` — runnable examples with sample data under `examples/data/`.
- `scratch/` — **exploratory only**. Contains `test_*`-named scripts that must
  never be collected by pytest (`testpaths = ["tests"]` enforces this).
  Excluded from ruff.
- `docs/` — Sphinx docs (published to wara.readthedocs.io).

## Conventions

- **A feature is not "done" until it has all four:** implementation, a test in
  `tests/`, a minimal runnable example in `examples/`, and Read the Docs
  documentation in `docs/`. See the `add-feature` skill.
- **When editing anything under `wara/gui/`, follow the `gui-conventions`
  skill.** Two recurring bugs: (1) the options panel is a fixed 270px
  (`OPT_W`) — wrap labels and give full-width buttons
  `QSizePolicy.Ignored` so content never overflows/clips; (2) data-loading
  buttons must flash a solid "busy" color (with `repaint()`) while working.
- **GUI tests run headless.** Set `QT_QPA_PLATFORM=offscreen` and
  `MPLBACKEND=Agg` (the test files also set these themselves; CI sets them too).
- Matplotlib backend must stay non-interactive in tests (`Agg`).
- Ruff lint ignores `E702` project-wide: GUI modules deliberately pair related
  Qt layout calls on one line (e.g. `lay.setContentsMargins(...); lay.setSpacing(8)`).
- `wara/__init__.py` intentionally uses wildcard re-exports (`F401`/`F403`
  ignored there).
- Version is derived from git tags via `setuptools_scm` and written to
  `wara/version.py` — never edit that file by hand.
- CI matrix: ubuntu/windows/macos x Python 3.10 & 3.12. Keep changes portable.
- Use tooltips in the GUI generously

## API data

API/PIXIE data loading reads folder paths from `data-path.txt` in the repo root
(one path per line, first match wins). This file is gitignored — each user keeps
their own local copy.
