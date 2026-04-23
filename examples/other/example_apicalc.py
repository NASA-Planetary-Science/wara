"""
apicalc example using synthetic API-detector event data.

Demonstrates event filtering (energy, time, position), position
reconstruction with calc_own_pos, FFT of a time-difference signal,
and the alpha-fraction helper approximate_fa.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wara import apicalc

RNG = np.random.default_rng(42)
N = 5000

# ---- Synthetic event DataFrame ------------------------------------------
# Simulate detector hits: energy in ADC channels, dt in ns, XY position
df_raw = pd.DataFrame({
    "energy": RNG.normal(loc=1000, scale=80, size=N),   # main peak ~1000 ch
    "dt":     RNG.normal(loc=20,   scale=15,  size=N),  # prompt peak at 20 ns
    "X":      RNG.uniform(-0.9, 0.9, N),
    "Y":      RNG.uniform(-0.9, 0.9, N),
    # ABCD anode signals for position reconstruction
    "A":      RNG.uniform(0.1, 1.0, N),
    "B":      RNG.uniform(0.1, 1.0, N),
    "C":      RNG.uniform(0.1, 1.0, N),
    "D":      RNG.uniform(0.1, 1.0, N),
})

# ---- Position reconstruction --------------------------------------------
df = apicalc.calc_own_pos(df_raw)
print(f"X2 range: [{df['X2'].min():.3f}, {df['X2'].max():.3f}]")
print(f"Y2 range: [{df['Y2'].min():.3f}, {df['Y2'].max():.3f}]")

# ---- Event filtering ----------------------------------------------------
erange  = [900, 1100]     # energy window around peak
trange  = [0, 50]         # prompt neutron window (ns)
xrange  = [-0.5, 0.5]
yrange  = [-0.5, 0.5]

df_e   = apicalc.dfe(df, erange=erange)
df_t   = apicalc.dft(df, trange=trange)
df_et  = apicalc.dfet(df, erange=erange, trange=trange)
df_xye = apicalc.dfxye(df, xrange=xrange, yrange=yrange, erange=erange)
df_all = apicalc.dftxye(df, xrange=xrange, yrange=yrange, erange=erange, trange=trange)

print(f"\nTotal events       : {len(df)}")
print(f"After energy cut   : {len(df_e)}")
print(f"After time cut     : {len(df_t)}")
print(f"After energy+time  : {len(df_et)}")
print(f"After XY+energy    : {len(df_xye)}")
print(f"After all cuts     : {len(df_all)}")

# ---- Plotting -----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Energy spectrum before/after cut
axes[0].hist(df["energy"], bins=80, alpha=0.5, label="all")
axes[0].hist(df_e["energy"], bins=80, alpha=0.7, label=f"E in {erange}")
axes[0].set_xlabel("Energy (ch)")
axes[0].set_title("Energy spectrum")
axes[0].legend()

# Time spectrum before/after cut
axes[1].hist(df["dt"], bins=80, range=(-50, 100), alpha=0.5, label="all")
axes[1].hist(df_t["dt"], bins=80, range=(-50, 100), alpha=0.7, label=f"dt in {trange} ns")
axes[1].set_xlabel("dt (ns)")
axes[1].set_title("Time spectrum")
axes[1].legend()

# 2D position map of selected events
axes[2].hist2d(df_all["X2"], df_all["Y2"], bins=40, cmap="plasma")
axes[2].set_xlabel("X2")
axes[2].set_ylabel("Y2")
axes[2].set_title("2D position after all cuts")

plt.tight_layout()
plt.show()

# ---- FFT of time-difference signal --------------------------------------
# Compute FFT on the dt values of selected events to look for oscillations
dt_signal = df_et["dt"].values
sampling_rate = 1.0  # 1 sample per event (arbitrary units)
freqs, magnitude = apicalc.compute_fft(dt_signal, sampling_rate)

fig2, ax = plt.subplots(figsize=(8, 4))
ax.plot(freqs[1:], magnitude[1:])  # skip DC
ax.set_xlabel("Frequency")
ax.set_ylabel("Magnitude")
ax.set_title("FFT of dt signal")
plt.tight_layout()
plt.show()

# ---- Alpha-fraction geometry helper ------------------------------------
for dist in [10, 20, 30, 50, 100]:
    fa = apicalc.approximate_fa(L=dist, S=10)
    print(f"  L={dist:3d} cm, S=10 cm -> fa = {fa:.4f}")
