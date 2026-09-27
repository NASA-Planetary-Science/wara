"""
Bypass the recorded PIXIE energy and use each trace's gate integral instead.

PIXIE stores energy in 16 bits, so a pulse whose energy exceeds 65535 wraps
around to a small value: large-amplitude traces then show up at low energy.
Parquet runs use the recorded energy by default; ``recorded_energy=False``
(the Neutrons tab's "Energy from traces" option) takes Q_total from the traces.

Synthetic pulses stand in for a PIXIE parquet run, so no data is needed.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wara.neutron_psd import NeutronTraces

rng = np.random.default_rng(0)
n = 2000
t = np.arange(200)
amp = rng.uniform(100, 12000, n)                        # pulse heights (ADC)
shape = np.where(t >= 40, np.exp(-(t - 40) / 8.0), 0.0)
traces = [1600 + a * shape + rng.normal(0, 5, t.size) for a in amp]
true_energy = 8.0 * amp                                  # up to ~96k ADC
pixie_energy = true_energy % 65536                       # the 16-bit wrap
df = pd.DataFrame({"channel": 1, "trace": traces, "energy": pixie_energy})

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, recorded in zip(axes, (True, False)):
    nt = NeutronTraces.from_pixie(channel=1, df=df, dt_ns=2.0, align=None,
                                  source="parquet",
                                  recorded_energy=recorded).compute()
    ax.scatter(nt.energy, amp, s=2)
    ax.set_xlabel("Recorded PIXIE energy [ADC]" if recorded
                  else "Energy from traces, Q_total [ADC·ns]")
    ax.set_ylabel("Pulse amplitude [ADC]")
axes[0].set_title("PIXIE energy: wrapped pulses fall at low E")
axes[1].set_title("Energy from traces: monotonic")
fig.tight_layout()
plt.show()
