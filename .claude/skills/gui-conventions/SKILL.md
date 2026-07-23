---
name: gui-conventions
description: Use whenever editing or adding to the wara PyQt5 GUI (any file under wara/gui/) — adding buttons, labels, options panels, or data-loading actions. Enforces the options-panel width discipline and the data-loading visual-feedback pattern, two recurring sources of GUI bugs. Trigger on GUI/tab/panel/button/widget work in wara/gui/.
---

# wara GUI conventions

The GUI is a three-column layout in `wara/gui/app.py`: nav rail → **options
panel** (column 2) → plot stack. Each tab's options live in a `*Options` class
(a `QScrollArea`) under its own module (`spectrum.py`, `calibration.py`,
`neutrons.py`, ...). Shared helpers are in `wara/gui/widgets.py`
(`header`, `labeled_row`, `stat_row`, `check_row`, `hsep`, `SpinBox`,
`DoubleSpinBox`, `ComboBox`) and `wara/gui/theme.py` (colors, stylesheet).

## Rule 1 — the options panel has a FIXED width; never let content overflow it

The options panel is a fixed width for every tab: `MainWindow.OPT_W = 270`
(`app.py:159`). Adding a wide widget silently forces the scroll area wider than
the viewport and **clips the right edge** of controls. Whenever you add to a
`*Options` panel, verify:

- **Labels/help text wrap:** call `label.setWordWrap(True)` on any label that
  could exceed the panel width (see the many `setWordWrap(True)` calls in
  `api_combine.py`, `api_diagnostics.py`).
- **Full-width buttons shrink, not push:** give buttons meant to fill the panel
  width `b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)` so they shrink
  with the panel instead of forcing horizontal scroll. Canonical example and the
  reason are in `neutrons.py:600-605`:

  ```python
  # Let full-width buttons shrink with the fixed-width panel instead of
  # forcing the scroll content wider than the viewport (which would clip
  # their right edge).
  for b in (self.btn_load, self.btn_reset, self.btn_send_spec, ...):
      b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
  ```

- **Narrow inputs get an explicit width:** small numeric fields use
  `ed.setFixedWidth(70)` (or 80/104) rather than stretching — see `api.py`,
  `api_combine.py`.

After adding widgets, mentally (or by running `wara`) check the panel at its
270px width: nothing should be cut off or require horizontal scrolling.

## Rule 2 — buttons that load/process data must show a "working" state

Any button that triggers a blocking load or computation must change color while
it works, so the user knows to wait. Use the flash pattern from
`api_shifts.py`. Set up the button with a stored base style and accent color:

```python
btn._color = T.ACCENT_CYAN                    # or ACCENT_GREEN, etc.
btn._base_css = self._apply_css(btn._color, solid=False)   # outline style
btn.setStyleSheet(btn._base_css)
```

Then in the click handler, fill it solid before the work and revert after:

```python
def _on_load(self):
    b = self.btn_load
    b.setStyleSheet(self._apply_css(b._color, solid=True))  # solid = "busy"
    b.repaint()                     # force the color to paint before blocking
    # ... do the blocking load/compute ...
    b.setStyleSheet(b._base_css)    # revert to outline when done
```

- `repaint()` (or `QApplication.processEvents()`) before the blocking call is
  required, or the color change won't appear until after the work finishes.
- For a brief action, `api_shifts.py:339` reverts via
  `QTimer.singleShot(220, lambda: b.setStyleSheet(b._base_css))`; for a real
  load, revert in a `finally` after the work completes.
- Disable the button while loading if a double-click would re-enter
  (`b.setEnabled(False)` / `True`), and consider `Qt.WaitCursor` for long ops.

## Reminders

- Buttons should set `setCursor(Qt.PointingHandCursor)` (used everywhere).
- Use the `widgets.py` helpers and `theme.py` colors — don't hardcode styles.
- GUI changes need a headless test in `tests/test_gui_<tab>.py`
  (`QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`). See the `add-feature` skill.
