"""Nuclear Database tab: element/isotope filtering of the line libraries."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd  # noqa: E402

from wara.gui.nuclear import filter_lines, load_database  # noqa: E402

ISOTOPES = ["1H", "2H", "232Th", "103Rh", "16O", "60Co", "165Ho", "98Mo",
            "35Cl", "28Si", "56Fe", "156Gd", "00C"]
DF = pd.DataFrame({"Energy (keV)": [100.0 + i for i in range(len(ISOTOPES))],
                   "Isotope": ISOTOPES})


def _isos(text):
    return set(filter_lines(DF, text, "", "")["Isotope"])


def test_element_list_returns_only_requested_elements():
    # Regression: "H" used to match Th/Rh and "O" matched Co/Ho/Mo (suffix match).
    assert _isos("H, Cl, Si, O") == {"1H", "2H", "35Cl", "28Si", "16O"}


def test_isotope_tokens_either_order_and_exact_mass():
    assert _isos("Co60, 56Fe, Co-60") == {"60Co", "56Fe"}
    assert _isos("56") == {"56Fe"}          # not 156Gd
    assert _isos("C") == {"00C"}            # not 60Co
    assert _isos("") == set(ISOTOPES)


def test_real_database_element_filter():
    out = filter_lines(load_database("Neutron capture (IAEA)"), "H, Cl, Si, O", "", "")
    syms = set(out["Isotope"].str.replace(r"\d", "", regex=True))
    assert syms == {"H", "Cl", "Si", "O"}
