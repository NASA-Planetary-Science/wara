"""
Parse NIST data file
"""
from collections import defaultdict
from functools import lru_cache
import re
import pandas as pd
from importlib.resources import files


# The two shipped data files never change during a session, but
# isotopic_abundance() is called once per element (~100 times) by the nuclide
# identifier -- so re-reading and re-parsing them each time dominated the
# lookup. Cache the parsed contents; the callers only read them.
@lru_cache(maxsize=1)
def _symbols():
    data_dir = files("wara").joinpath("nuclear-data")
    path = str(data_dir.joinpath("elements_symbols.csv"))
    return frozenset(pd.read_csv(path)["Sym"])


@lru_cache(maxsize=1)
def _nist_lines():
    data_dir = files("wara").joinpath("nuclear-data")
    path = str(data_dir.joinpath("Isotopes-NIST.txt"))
    with open(path, "r") as f:
        return tuple(f.readlines())


@lru_cache(maxsize=1)
def _abundance_index():
    """``{symbol: {'ZSym-A': fraction}}`` for every element, built in one pass.

    The nuclide identifier asks for ~100 elements per lookup, and scanning the
    ~27k-line NIST file per element made that the single most expensive part of
    an identification. One pass fills the table for all elements at once; the
    per-record parsing below mirrors the original per-element scan exactly.
    """
    lines = _nist_lines()
    symbols = _symbols()
    index = {}
    for i, line in enumerate(lines):
        for symbol in line.split():
            if symbol not in symbols:
                continue
            Z = re.findall(r"\d+", lines[i - 1])[0]
            isotope = re.findall(r"\d+", lines[i + 1])[0]
            # account for isotopically pure elements
            try:
                tmp2 = lines[i + 3].split()
                comp = [float(tmp2[-1])]
            except ValueError:
                comp = re.findall(r"\d.+(?=\()", lines[i + 3])
            if len(comp) != 0:
                index.setdefault(symbol, {})[Z + symbol + "-" + isotope] = float(comp[0])
    return index


def isotopic_abundance(element):
    """
    Parameters
    ----------
    element : string
        chemical symbol e.g 'H'.

    Returns
    -------
    result : dictionary
        stable isotopes and their respective natural abundance.

    """
    element = element.title()
    if element not in _symbols():
        warning_msg = "Element not found.\nMake sure to write the element symbol only"
        return warning_msg
    # Copy so callers can't mutate the shared cached table.
    return defaultdict(None, _abundance_index().get(element, {}))


def isotopic_abundance_str(element):
    "Return string of isotopic abundances"
    res = isotopic_abundance(element)
    if isinstance(res, str):
        return res
    keys = list(res.keys())
    vals = list(res.values())
    string = ""
    for k, v in zip(keys, vals):
        v2 = round(v * 100, 3)
        string = string + k + " : " + str(v2) + "%" + "\n"
    return string
