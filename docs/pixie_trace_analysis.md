# Pixie-16 fast trigger / CFD reconstruction

The Pixie-16 firmware time-stamps every event with an internal fast filter
(for triggering) and a CFD (for sub-sample timing), but it only ever writes
out the *final* CFD fractional time -- not the fast filter trace, the fast
trigger point, or the CFD trace it computed to get there. Comparing the
recorded timing against the pulse shape, or debugging an unusual `FastThresh`
/ `CFDThresh` setting, needs those intermediate traces reconstructed offline.

{py:mod}`wara.pixie_trace_analysis` reimplements that pipeline directly from a
recorded ADC trace and the run's DSP settings (Pixie-16 User Manual v3.06,
secs 3.3.7-3.3.8):

1. **fast filter** ({py:func}`wara.pixie_trace_analysis.fast_filter`) -- the
   trapezoidal leading-minus-trailing running sum (Eq 3-1) used to detect that
   a pulse has arrived,
2. **fast trigger** ({py:func}`wara.pixie_trace_analysis.find_fast_trigger`)
   -- the first sample where that filter crosses the `FastThresh` DSP
   register,
3. **CFD trace** ({py:func}`wara.pixie_trace_analysis.cfd_trace`) -- the
   bipolar response (Eq 3-5, the 500 MHz Pixie-16 variant) whose zero crossing
   marks the pulse's sub-sample arrival time, and
4. **CFD trigger** ({py:func}`wara.pixie_trace_analysis.find_cfd_crossing`) --
   the sub-sample zero-crossing position, gated by the `CFDThresh` DSP
   register to ignore noise-caused crossings ahead of the real pulse.

```{note}
The 500 MHz CFD's `w`, `B`, `D`, `L` parameters (manual Table 3-4) are fixed
in hardware at `w=1, B=5, D=5, L=1` -- they are *not* the `CFDScale` /
`CFDDelay` DSP registers, which apply to the 100/250 MHz CFD variant (Eq 3-2)
instead.
```

```{note}
Custom firmware builds may run the CFD at a fractional weight instead of the
stock `w=1`. Pass `cfd_trace(..., w=...)` to reconstruct that variant; the
weight our local custom firmware uses is exposed as
{py:data}`wara.pixie_trace_analysis.CFD_W_CUSTOM` (`0.3125`). In the GUI, the
Diagnostics → Binary → Random traces view exposes this as the **Custom firmware
(w=0.3125)** checkbox.
```

## Reading the DSP geometry

`FastLength` and `FastGap` are stored in `fast_filter_cycles` units and need
converting to ADC-sample (2 ns) units the same way
{py:func}`wara.helper_api.read_slow_filter_geometry` does for the energy
filter:

```python
from wara import pixie_trace_analysis as pta

FL, FG, fast_thresh = pta.read_fast_trigger_geometry(date, runnr, ch)
cfd_thresh = pta.read_cfd_threshold(date, runnr, ch)
```

## Reconstructing a trace

```python
from wara import helper_api

df = helper_api.read_trace_data(date, runnr)
trace = df.trace[0]

result = pta.compute_cfd_timing(trace, FL, FG, fast_thresh, cfd_thresh)
result["fast_filter"]         # trapezoidal fast filter response
result["fast_trigger_index"]  # sample where it crosses FastThresh
result["cfd_trace"]           # bipolar CFD response
result["cfd_position"]        # sub-sample zero-crossing (samples), NaN if none
result["cfd_valid"]           # False wherever cfd_position is NaN
```

Every function accepts either a single 1D trace or a 2D array of traces
(one per row), so the same call reconstructs a whole batch of events at once.

```{seealso}
`examples/pixie/example_pixie_trace_analysis.py` runs this end to end on real
acquisitions -- one read with {py:func}`wara.helper_api.read_trace_data`, one
with {py:func}`wara.helper_api.read_binary_data` -- and plots the trace with
the reconstructed fast trigger and CFD zero crossing overlaid.
```

## In the GUI

The same reconstruction is available without writing any code, on the API
tab's diagnostics dialog: **API → Diagnostics… → Binary → Random traces**.
Load a run, hit **Sample**, and tick either of

- **Fast Filter** — draws the fast filter of every sampled trace on its own
  y-axis, with a dashed line at that channel's `FastThresh`, and
- **CFD** — draws the 500 MHz CFD trace on a second y-axis, with a dashed line
  at `CFDThresh` and a marker line at zero.

Both can be on at once (the CFD axis then moves outward so the two right-hand
spines stay readable), and both re-draw on every new **Sample**. Where more
than one channel is sampled, one dashed threshold line is drawn per distinct
register value and the legend lists them all.

The checkboxes stay greyed out until a run whose settings file supplies
`FastLength` / `FastGap` / `FastThresh` / `CFDThresh` is loaded — that covers
runs read as **Trace data** and as **Binary data**, but the overlays are turned
off automatically if a later run has no settings to read. They also have no
effect while the **FFT** toggle is on, since the filters live in the time
domain.

```{seealso}
`examples/pixie/example_gui_random_trace_filters.py` drives that dialog from a
script — load a run, pick a channel, sample, and switch both overlays on.
```

For the full API, see {py:mod}`wara.pixie_trace_analysis` in the API
reference.
