"""
Pulse-shape discrimination on an API run's per-event traces with
``wara.neutron_psd.TracePSD`` (the backend of the API tab's PSD panel).

Each event's trace is timed by the PIXIE-style fast filter (30 ADC threshold);
the gates sit relative to that trigger: open 30 ns before it, prompt window
closes 20 ns after it, tail runs to the end of the trace.

Needs the run under a folder listed in ``data-path.txt``.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from wara import helper_api, read_parquet_api
from wara.neutron_psd import TracePSD

DATE, RUNNR, CH = "2026-09-24", 1, 1          # EJ250 neutron detector

df = read_parquet_api.read_parquet_file(DATE, RUNNR, ch=CH)
dt_ns = helper_api.read_sample_interval_ns(DATE, RUNNR)
psd = TracePSD.from_column(df["Trace"], dt_ns).compute()
v = psd.valid
print(f"{v.sum():,} of {psd.n_events:,} events with a valid PSD; "
      f"gates (ns, aligned axis): {psd.gate_times()}")

fig, (ax_psd, ax_tr) = plt.subplots(1, 2, figsize=(12, 4.5))
lo, hi = np.percentile(psd.psd[v], [0.5, 99.5])
ax_psd.hist2d(df["energy"][v], psd.psd[v], bins=[300, np.linspace(lo, hi, 300)],
              norm=LogNorm(), cmap="jet")
ax_psd.set_xlabel("Energy [ADC]")
ax_psd.set_ylabel("PSD = 1 - Q_prompt / Q_total")
ax_psd.set_title(f"{DATE} RUN {RUNNR} ch {CH}")

# A random sample of traces aligned on their trigger, with the gates.
sample = np.random.default_rng(0).choice(np.flatnonzero(v), 100, replace=False)
ax_tr.plot(psd.time_ns, psd.aligned(sample).T, lw=0.5, alpha=0.5)
for t in psd.gate_times():
    ax_tr.axvline(t, color="k", ls="--")
ax_tr.set_xlim(psd.trigger_ns - 60, psd.time_ns[-1])
ax_tr.set_xlabel("Time [ns]")
ax_tr.set_ylabel("Amplitude [ADC]")
fig.tight_layout()

# The API tab's "Apply to data" saves these values in a "PSD" column of the
# new run's parquet (NaN for other channels / invalid events), e.g.:
#     full["PSD"] = np.where(psd.valid, psd.psd, np.nan)
plt.show()
