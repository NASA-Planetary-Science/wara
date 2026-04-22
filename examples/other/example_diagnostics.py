"""
Example file for 'Diagnostics' tab
"""
from wara import diagnostics
from wara import file_reader
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("../data/test_folder_diag")

spe = file_reader.ReadSPE(folder / "RUN000.Spe")
diag = diagnostics.Diagnostics(folder)
diag.calculate_integral(xmid=2478, width=70)

plt.figure()
plt.plot(diag.absolute_time, diag.areas, "o-", label="Peak integrals")

diag.fit_peaks(xmid=2478, width=70)
plt.plot(diag.absolute_time, diag.areas, "o-", label="Peak fits")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Counts")
