"""
Exponential decay fitting example using synthetic data.

Demonstrates single- and double-exponential fits with the Decay_exp class.
"""

import numpy as np
import matplotlib.pyplot as plt
from wara import decay_exponential as decay

RNG = np.random.default_rng(42)

# ---- Synthetic single-decay data ----------------------------------------
# True parameters: A=1000, k=0.05, C=50
t = np.linspace(1, 100, 60)
noise = 15.0
y_single = decay.single_decay_plus_constant(t, A=1000, k=0.05, C=50)
y_single += RNG.normal(0, noise, size=len(t))
yerr_single = np.full_like(t, noise)

# ---- Fit single decay ---------------------------------------------------
exp1 = decay.Decay_exp(x=t, y=y_single, yerr=yerr_single)
exp1.fit_single_decay()
print("=== Single decay fit ===")
print(exp1.fit_result.fit_report())

ax_fit, ax_res = exp1.plot(show_components=True)
ax_fit.set_title("Single exponential decay")
plt.show()

# ---- Synthetic double-decay data ----------------------------------------
# True parameters: A1=700, k1=0.04, A2=300, k2=0.3, C=30
y_double = decay.double_decay_plus_constant(t, A1=700, A2=300, k1=0.04, k2=0.3, C=30)
y_double += RNG.normal(0, noise, size=len(t))
yerr_double = np.full_like(t, noise)

# ---- Fit double decay ---------------------------------------------------
exp2 = decay.Decay_exp(x=t, y=y_double, yerr=yerr_double)
exp2.fit_double_decay()
print("\n=== Double decay fit ===")
print(exp2.fit_result.fit_report())

ax_fit, ax_res = exp2.plot(show_components=True)
ax_fit.set_title("Double exponential decay")
plt.show()
