---
name: add-feature
description: Use when adding or extending a feature in the wara package (analysis code or GUI). Enforces wara's definition of "done" — every feature ships with a test, a minimal runnable example, the implementation, and Read the Docs documentation. Trigger on "add a feature", "implement", "add support for", "new capability".
---

# Adding a feature to wara

A feature is not done until **all four** of these exist. Do them together, not
"code now, docs later." When you finish, report which of the four you touched.

## 1. Implementation

- Analysis code goes in a top-level module under `wara/` (e.g.
  `wara/spectrum.py`, `wara/peakfit.py`). GUI code goes under `wara/gui/`
  (one module per tab — see the `gui-conventions` skill before editing GUI).
- If it's part of the public API, re-export it from `wara/__init__.py`
  (currently `Spectrum`, `peakfit`, `PeakSearch`).
- Match the surrounding style. Do **not** add features to `wara/gui_legacy/`
  (frozen) or `wara/planetary-nuclear-spect/` (ad-hoc scripts).

## 2. Test

- Add to `tests/`, mirroring the module name: `wara/foo.py` → `tests/test_foo.py`;
  a GUI tab → `tests/test_gui_<tab>.py`.
- Never put tests in `scratch/` — pytest only collects `tests/`
  (`testpaths = ["tests"]`), so a test elsewhere silently never runs.
- GUI tests run **headless**: ensure `QT_QPA_PLATFORM=offscreen` and
  `MPLBACKEND=Agg` (existing `test_gui_*.py` set these at the top — copy that).
- Run it: `pytest tests/test_foo.py -q`. It must pass on Python 3.10 (the floor).

## 3. Minimal working example

- Add a small, self-contained script under the matching `examples/` subfolder
  (`examples/spectrum/`, `examples/peakfit/`, `examples/pixie/`, ...; create a
  new subfolder if none fits).
- Use sample data from `examples/data/` (e.g. `test_data_cebr.csv`) — do not
  depend on the user's private API data or `data-path.txt`.
- The example should run top-to-bottom with `python examples/.../your_example.py`
  and demonstrate the feature in the fewest lines that are still realistic.

## 4. Read the Docs documentation

- Docs are Sphinx under `docs/`, published to wara.readthedocs.io.
- Add or extend the relevant `.rst` page for the feature; if you added a public
  symbol, make sure it appears in the API reference.
- Reference the example you wrote in step 3 where it helps the reader.
- Build/check locally if possible before considering it done.

## Final check

Run the suite and lint before declaring done:

```bash
pytest tests -q
ruff check .
```
