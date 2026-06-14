"""
Likely-isotope identification from a list of peak energies.

Two features shown:
  * identify()  -> the TOP candidates for each database, per energy (stage 1),
                   with natural isotopic abundance folded into the ranking.
  * the 846.8 keV peak: by bare cross section TALYS prefers 57Fe, but 56Fe is
    ~92 % abundant vs 57Fe's ~2 %, so abundance weighting correctly picks 56Fe.

Scenario: a 14 MeV neutron-inelastic spectrum with 16O (6129 keV line + its
escape peaks and other 16O lines) plus an iron 846.8 keV line.
"""
import pandas as pd

from wara import nuclide_identificator as nid

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 20)

energies = [
    6129.9,    # 16O full-energy peak
    5618.9,    # 16O 6129 single-escape peak  (6129.9 - 511)
    5107.9,    # 16O 6129 double-escape peak  (6129.9 - 1022)
    6917.1,    # 16O
    3684.5,    # 16O
    2742.0,    # 16O
    846.8,     # iron inelastic line
    1779,
    2223,
]


def tol(energy):
    """Energy-dependent window: ~0.1 % of E, 1.5 keV floor."""
    return max(1.5, 0.001 * energy)


print("Input energies (keV):", energies, "\n")

# ── Top candidates for each database, per energy ──────────────────────────────
per_db = nid.identify(energies, tol=tol, top_n=3)
for name in ("TALYS 14 MeV", "Inelastic (Baghdad)"):
    print(f"=== {name} — top 3 candidates per energy ===")
    print(per_db[name].to_string(index=False))
    print()

# ── The 846.8 keV peak: abundance turns 57Fe into 56Fe ────────────────────────
std_talys = nid.standardize(nid.load_database("TALYS 14 MeV"))
print("=== 846.8 keV in TALYS — effect of natural isotopic abundance ===")
without = nid.score_database([846.8], std_talys, tol=2.0, top_n=3, weight_abundance=False)
withab = nid.score_database([846.8], std_talys, tol=2.0, top_n=3, weight_abundance=True)
print("without abundance:", list(zip(without["Isotope"], without["Probability"])))
print("with abundance:   ", list(zip(withab["Isotope"], withab["Probability"])))
