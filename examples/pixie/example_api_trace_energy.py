"""
Energy from the traces instead of the recorded PIXIE energy (API runs).

PIXIE stores energy in 16 bits, so a pulse above 65535 wraps around to a low
value. When every event carries a trace, TracePSD's per-event trace integral
Q_total is an energy axis that cannot wrap -- the API tab's "Energy from
traces" option. Pulses cut off by the trace window are rejected
(``n_misaligned``) rather than given a truncated energy.

Synthetic uint16 traces stand in for a parquet ``Trace`` column, so no data
is needed.
"""
import matplotlib.pyplot as plt
import numpy as np

from wara.neutron_psd import TracePSD

rng = np.random.default_rng(0)
n, n_samples = 3000, 500
x = np.arange(n_samples)
amp = rng.uniform(200, 14000, n)                         # pulse heights (ADC)
start = rng.normal(216, 2, n)
start[:5] = 440                                          # a few cut-off pulses
body = np.where(x >= start[:, None],
                amp[:, None] * np.exp(-(x - start[:, None]) / 8.0), 0.0)
traces = np.round(1660 + body + rng.normal(0, 3, body.shape)).astype(np.uint16)
pixie_energy = (7.0 * amp) % 65536                       # the 16-bit wrap

p = TracePSD(traces, dt_ns=2.0).compute()
print(f"{p.valid.sum()} valid, {p.n_misaligned} rejected (pulse cut off)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(pixie_energy, amp, s=2)
axes[0].set_xlabel("Recorded PIXIE energy [ADC]")
axes[0].set_title("PIXIE energy: wrapped pulses fall at low E")
axes[1].scatter(p.q_total[p.valid], amp[p.valid], s=2)
axes[1].set_xlabel("Energy from traces, Q_total [ADC·ns]")
axes[1].set_title("Energy from traces: monotonic")
for ax in axes:
    ax.set_ylabel("Pulse amplitude [ADC]")
fig.tight_layout()
plt.show()
