# wara

Python package for **gamma-ray and neutron spectroscopy** analysis and
visualization: spectrum smoothing/rebinning, peak search, multi-peak Gaussian
fitting, energy/efficiency/resolution calibration, nuclide identification
(neutron-induced gamma emphasis), PIXIE/Associated Particle Imaging (API) data,
and planetary nuclear spectroscopy. Ships a PyQt5 GUI (entry point: `wara`).

## Commands

```bash
pip install -e .            # editable install (run from repo root)
wara                        # launch the GUI
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
  The pre-2.0 GUI (`wara/gui_legacy/`) was retired in v2.1 — it is gone from
  the tree, and `wara --legacy` now prints a notice and launches the current GUI.
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
- **Every new test must run in under 1 second.** Do not add a slower test
  without explicit permission from the user — ask first, don't just add it and
  flag it afterwards. Check with
  `pytest tests -q --durations=15` (which reports `setup`/`call` separately;
  both count). Common causes and the fixes that worked:
  - `ax.hist` builds one matplotlib patch per bin, so production bin defaults
    (512 dt bins) dominate. Shrink the bin count on the controller in the
    fixture, not in the option line-edits that other tests assert on.
  - Iteration/sampling budgets: seeded solvers (`random_state=0`) converge in
    far fewer iterations than the production cap. Use a modest budget with a
    documented margin.
  - Process-wide one-time costs (nuclide databases, the NIST abundance table,
    `dateparser`'s locale data) are warmed in `tests/conftest.py` during
    collection. Add new ones there rather than letting them land on whichever
    test touches them first — and never hide a genuinely slow test behind a
    fixture, since `setup` time counts too.
  - If a test is slow because production code is slow, fix the production
    code — that is where most of the wins in this suite came from.
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
