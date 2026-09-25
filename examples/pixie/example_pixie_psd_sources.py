"""
Read a PIXIE run for pulse-shape discrimination from each available source
(trace data, binary data, parquet data) and compare the PSD of one channel.

Needs the run under a folder listed in ``data-path.txt``.
"""
import matplotlib.pyplot as plt

from wara import helper_api
from wara.neutron_psd import (
    NeutronTraces,
    available_pixie_sources,
    pixie_trace_channels,
    read_pixie_run,
)

DATE = "2026-09-24"
RUNNR = 1

sources = available_pixie_sources(DATE, RUNNR)
print("Available sources:", sources)
dt_ns = helper_api.read_sample_interval_ns(DATE, RUNNR)

fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 4),
                         squeeze=False)
for ax, source in zip(axes[0], sources):
    df = read_pixie_run(DATE, RUNNR, source=source, cfd="on", align="fast")
    channel = pixie_trace_channels(df)[0]
    nt = NeutronTraces.from_pixie(channel=channel, df=df, dt_ns=dt_ns,
                                  source=source).compute()
    v = nt.valid
    ax.hist2d(nt.energy[v], nt.psd[v], bins=150, cmap="jet", cmin=1)
    ax.set_title(f"{source} data · ch {channel} · {v.sum():,} pulses")
    # Parquet runs use the recorded PIXIE energy; the others the gate integral.
    ax.set_xlabel("Energy [ADC]" if source == "parquet" else "Energy [ADC·ns]")
    ax.set_ylabel("PSD = 1 - Q_prompt / Q_total")
fig.tight_layout()
plt.show()
