"""
nuclide_identificator.py — likely-isotope identification from a list of peak
energies.

Given a list of observed peak energies (keV), this module ranks the most likely
source isotope for each energy. Line strengths (``XS (mb)``, ``Sigma (mb)`` or
``Intensity (%)``) are only meaningful *within* one database, so it works in two
stages:

1. **Per database — top candidates.** Within each library the candidate isotopes
   for an energy are ranked using that library's own (absolute) cross sections /
   intensities, weighted by **natural isotopic abundance** and a **terrestrial
   element-naturalness prior** (both for reaction-on-natural-target libraries)
   and boosted by corroborating lines. The element prior (:func:`element_factor`,
   from :data:`NATURAL_ABUNDANCE_PPM`) favours the common rock/air/biological
   elements (O, Si, Fe, H, C, N…) over rare ones (Gd, Eu, Kr…), reflecting that a
   detector usually looks at natural matter. :func:`identify` returns the top-N
   candidates *for each database, per energy*.

2. **Comparable probability across databases.** The per-database picks are pooled
   and re-scored with a *relative* strength (each matched line normalised by its
   isotope's strongest line) × abundance × corroboration, giving probabilities
   that are comparable between libraries (:func:`rank_candidates`).

Why abundance matters: at 846.8 keV the TALYS 14 MeV library has both a 57Fe and
a 56Fe (n,n') line; 57Fe's bare cross section is larger, but 56Fe is ~92 %
abundant vs 57Fe's ~2 %, so in natural iron the 846.8 keV peak is 56Fe. Abundance
weighting (target isotope) recovers that.

**Corroborating lines** include the single/double-escape peaks (E − 511, E − 1022
keV) of high-energy gammas — e.g. 16O's strong 6129 keV line plus its escapes and
its 6917 / 3685 keV lines together pin 6129 keV to 16O.

The matching tolerance ``tol`` may be a constant (keV) or a callable
``tol(energy) -> keV``.
"""
import re
from importlib.resources import files

import numpy as np
import pandas as pd

from wara import parse_NIST

__all__ = [
    "DATABASES", "ABUNDANCE_WEIGHTED", "load_database", "standardize",
    "abundance_factor", "element_factor", "NATURAL_ABUNDANCE_PPM",
    "score_database", "identify",
    "rank_candidates", "identify_best", "escape_relations",
]

# Display name → data file (mirrors the GUI's Nuclear Database list).
DATABASES = {
    "Common lab sources":        "Common_lab_sources.csv",
    "Natural radiation":         "Natural_radiation.csv",
    "Delayed activation (IAEA)": "Delayed_activation_IAEA.csv",
    "Neutron capture (CapGam)":  "Capture_CapGam.csv",
    "Neutron capture (IAEA)":    "Capture_IAEA.csv",
    "Inelastic (Baghdad)":       "Inelastic_Baghdad.csv",
    "TALYS 14 MeV":              "Talys-14MeV.csv",
}

# Libraries whose ``Isotope`` is a natural *target* nucleus, so the line yield
# scales with the target's natural abundance. (Source/decay libraries —
# Common lab sources, Natural radiation, Delayed activation — must NOT be
# abundance-weighted: their nuclides are radioactive/prepared, e.g. 40K, 137Cs.)
ABUNDANCE_WEIGHTED = {
    "TALYS 14 MeV", "Inelastic (Baghdad)",
    "Neutron capture (CapGam)", "Neutron capture (IAEA)",
}

# Strength columns, in preference order — the first one present is used.
STRENGTH_COLUMNS = ("XS (mb)", "Sigma (mb)", "Intensity (%)")

ELECTRON_MASS_KEV = 511.0
PAIR_THRESHOLD_KEV = 1022.0          # gammas above this can show escape peaks
ESCAPE_PEAKS = (("SEP", ELECTRON_MASS_KEV, 0.5),
                ("DEP", 2 * ELECTRON_MASS_KEV, 0.25))

DEFAULT_TOL = 2.0


# ── Database loading / normalisation ──────────────────────────────────────────
def load_database(name):
    """Load a shipped database by display name into a raw DataFrame."""
    if name not in DATABASES:
        raise KeyError(f"Unknown database '{name}'. Choose from {list(DATABASES)}")
    path = str(files("wara").joinpath("nuclear-data", DATABASES[name]))
    return pd.read_csv(path)


def standardize(df):
    """Reduce any database to the columns ``Isotope``, ``Energy`` (keV) and
    ``Strength`` (cross section / intensity; 1.0 when the library has none)."""
    col = next((c for c in STRENGTH_COLUMNS if c in df.columns), None)
    out = pd.DataFrame({
        "Isotope": df["Isotope"].astype(str).str.strip(),
        "Energy": pd.to_numeric(df["Energy (keV)"], errors="coerce"),
        "Strength": pd.to_numeric(df[col], errors="coerce") if col else 1.0,
    })
    out = out.dropna(subset=["Energy"]).reset_index(drop=True)
    out["Strength"] = out["Strength"].fillna(0.0).clip(lower=0.0)
    return out


# ── Natural isotopic abundance ────────────────────────────────────────────────
_ISO_RE = re.compile(r"^\s*(\d+)\s*([A-Za-z]{1,3})")
# Element symbol with an optional mass-number prefix ('16O', '00Fe', or 'Fe').
_ELEMENT_RE = re.compile(r"^\s*\d*\s*([A-Za-z]{1,3})")
_ABUNDANCE_CACHE = {}


def _abundance_table(element):
    """{mass number: natural fraction} for an element (cached); {} if unknown."""
    if element not in _ABUNDANCE_CACHE:
        try:
            raw = parse_NIST.isotopic_abundance(element)
        except Exception:  # noqa: BLE001 — missing element / parse error → no data
            raw = None
        table = {}
        if isinstance(raw, dict):
            for key, frac in raw.items():           # keys look like '26Fe-56'
                m = re.search(r"-(\d+)$", str(key))
                if m:
                    table[int(m.group(1))] = float(frac)
        _ABUNDANCE_CACHE[element] = table
    return _ABUNDANCE_CACHE[element]


def abundance_factor(isotope):
    """Natural isotopic abundance (0–1) of *isotope* (e.g. ``'56Fe'`` → 0.9175).

    Returns 1.0 (neutral) for a natural-element entry (``'00Fe'``), an
    unparseable name, or a mass not in the natural-abundance table (e.g. a
    radioactive nuclide) — so weighting never erases such candidates, it only
    down-ranks rare *stable* isotopes against abundant ones."""
    m = _ISO_RE.match(str(isotope))
    if not m:
        return 1.0
    mass = int(m.group(1))
    if mass == 0:                       # '00Fe' = natural element target
        return 1.0
    table = _abundance_table(m.group(2).capitalize())
    return table.get(mass, 1.0)


# ── Natural element abundance (terrestrial "naturalness" prior) ───────────────
# Approximate natural abundance of each *element*, in ppm by mass, for samples a
# detector normally sees: Earth's-crust composition, with the major atmospheric
# / biological / hydrological elements (H, C, N, O, Ar) raised to reflect that
# they are ubiquitous in real samples (water, organics, air) even when their
# bare crustal figure is small (N is ~19 ppm in crust but ~78 % of air). Only
# the ratios matter — scores are renormalised per energy — and the value feeds a
# log scale, so this is a firm but non-absolute prior: a rare element can still
# win on strong spectral evidence. Edit freely to match your sample matrix.
NATURAL_ABUNDANCE_PPM = {
    "O": 461000, "Si": 282000, "H": 140000, "Al": 82300, "Fe": 56300,
    "Ca": 41500, "Na": 23600, "Mg": 23300, "K": 20900, "C": 20000,
    "N": 5000, "Ti": 5650, "P": 1050, "Mn": 950, "F": 585, "Ba": 425,
    "Sr": 370, "S": 350, "Zr": 165, "Cl": 145, "V": 120, "Cr": 102,
    "Rb": 90, "Ni": 84, "Zn": 70, "Ce": 66, "Cu": 60, "Y": 33, "La": 39,
    "Nd": 41, "Co": 25, "Sc": 22, "Li": 20, "Nb": 20, "Ga": 19, "Pb": 14,
    "B": 10, "Th": 9.6, "Pr": 9.2, "Sm": 7.0, "Gd": 6.2, "Dy": 5.2,
    "Hf": 3.0, "Cs": 3.0, "Be": 2.8, "Sn": 2.3, "Br": 2.4, "U": 2.7,
    "Er": 3.5, "Yb": 3.2, "Ta": 2.0, "Eu": 2.0, "As": 1.8, "Ge": 1.5,
    "Ho": 1.3, "Mo": 1.2, "W": 1.25, "Tb": 1.2, "Tl": 0.85, "Lu": 0.8,
    "Tm": 0.52, "I": 0.45, "In": 0.25, "Sb": 0.2, "Cd": 0.15, "Ar": 1.2,
    "Ag": 0.075, "Hg": 0.085, "Se": 0.05, "He": 0.008, "Pd": 0.015,
    "Bi": 0.009, "Pt": 0.005, "Au": 0.004, "Te": 0.001, "Ru": 0.001,
    "Rh": 0.001, "Ir": 0.001, "Os": 0.0015, "Re": 0.0007, "Ne": 0.00007,
    "Kr": 0.0001, "Xe": 0.00003,
}

# Below-floor (and unknown-element) candidates keep this weight, so the prior
# only down-ranks rare elements rather than erasing them.
NATURAL_WEIGHT_FLOOR = 0.5


def element_factor(isotope):
    """Terrestrial "naturalness" weight for *isotope*'s element.

    Log-compressed :data:`NATURAL_ABUNDANCE_PPM`, so common rock/air/biological
    elements (O, Si, Fe, H, C, N…) outweigh rare ones (Gd, Eu, Kr…) by a firm
    but bounded factor. Returns 1.0 (neutral) for an unknown element so the prior
    only re-ranks elements we have data for. Works for ``'16O'``, ``'00Fe'`` and
    bare ``'Fe'`` alike."""
    m = _ELEMENT_RE.match(str(isotope))   # mass number optional
    if not m:
        return 1.0
    ppm = NATURAL_ABUNDANCE_PPM.get(m.group(1).capitalize())
    if ppm is None:
        return 1.0
    return max(NATURAL_WEIGHT_FLOOR, float(np.log10(ppm)) + 1.0)


# ── Observable-peak table (full + escape peaks) ───────────────────────────────
def _observable_table(std, escape_peaks=True):
    """Expand each line into its observable peaks: the full-energy peak plus,
    for gammas above the pair-production threshold, its single/double-escape
    peaks."""
    line = std["Energy"].to_numpy(float)
    iso = std["Isotope"].to_numpy()
    strength = std["Strength"].to_numpy(float)

    energies, isos, lines = [line], [iso], [line]
    strengths, weights = [strength], [np.ones_like(line)]
    types = [np.full(line.shape, "full", dtype=object)]

    if escape_peaks:
        high = line > PAIR_THRESHOLD_KEV
        n = int(high.sum())
        for label, offset, weight in ESCAPE_PEAKS:
            energies.append(line[high] - offset)
            isos.append(iso[high])
            lines.append(line[high])
            strengths.append(strength[high])
            weights.append(np.full(n, weight))
            types.append(np.full(n, label, dtype=object))

    return pd.DataFrame({
        "obs_energy": np.concatenate(energies),
        "isotope": np.concatenate(isos),
        "line": np.concatenate(lines),
        "strength": np.concatenate(strengths),
        "weight": np.concatenate(weights),
        "type": np.concatenate(types),
    })


def _nearest_distance(values, reference):
    """Distance from each ``values`` entry to the nearest ``reference`` value
    (sorted ascending), plus that nearest value."""
    pos = np.searchsorted(reference, values)
    left = np.clip(pos - 1, 0, len(reference) - 1)
    right = np.clip(pos, 0, len(reference) - 1)
    d_left = np.abs(values - reference[left])
    d_right = np.abs(values - reference[right])
    take_left = d_left <= d_right
    return np.minimum(d_left, d_right), np.where(take_left, reference[left], reference[right])


# ── Per-database evidence ─────────────────────────────────────────────────────
def _abundance_series(std):
    isos = std["Isotope"].unique()
    return pd.Series({iso: abundance_factor(iso) for iso in isos}, dtype=float)


def _element_series(std):
    isos = std["Isotope"].unique()
    return pd.Series({iso: element_factor(iso) for iso in isos}, dtype=float)


def _database_evidence(std, escape_peaks, weight_abundance, weight_element):
    obs = _observable_table(std, escape_peaks)
    max_strength = std.groupby("Isotope")["Strength"].max()
    abundance = _abundance_series(std) if weight_abundance else None
    element = _element_series(std) if weight_element else None
    return obs, max_strength, abundance, element


def _mark_present(obs, ref, tol, max_strength):
    """Flag observable peaks present in ``ref``; compute per-isotope ``support``
    (Σ over present peaks of weight·strength normalised by the isotope's
    strongest line — pure spectral shape, abundance-independent), the observed
    energies backing each isotope, and the set of ``(isotope, line)`` whose
    full-energy peak is observed (used to validate escape peaks)."""
    obs_energy = obs["obs_energy"].to_numpy()
    dist, nearest = _nearest_distance(obs_energy, ref)
    tol_at = np.array([float(tol(e)) for e in nearest]) if callable(tol) else float(tol)
    obs = obs.assign(present=dist <= tol_at, matched=nearest)
    present = obs[obs["present"]].copy()
    denom = present["isotope"].map(max_strength).replace(0.0, np.nan)
    present["evidence"] = (present["weight"] * present["strength"] / denom).fillna(
        present["weight"])
    support = present.groupby("isotope")["evidence"].sum()
    support_energies = present.groupby("isotope")["matched"].apply(
        lambda s: sorted(set(np.round(s, 3))))
    full = present[present["type"] == "full"]
    full_present = set(zip(full["isotope"].to_numpy(),
                           np.round(full["line"].to_numpy(), 3)))
    return obs, support, support_energies, full_present


def _candidates_for_energy(energy, obs, support, support_energies, max_strength,
                           abundance, element, tol, full_present):
    """Best matching line per isotope at *energy*, scored two ways (both folding
    in natural isotopic abundance and the element-naturalness prior):

    * ``raw_abs`` = strength · abundance · element · support — within-database.
    * ``raw_rel`` = (strength / strongest line) · abundance · element · support —
      comparable across databases.

    A SEP/DEP candidate is kept only when its parent full-energy line is itself
    observed (``full_present``): an escape peak never appears without its — always
    stronger — full peak, so a lone escape match is spurious.
    """
    tv = float(tol(energy)) if callable(tol) else float(tol)
    obs_energy = obs["obs_energy"].to_numpy()
    cand = obs[np.abs(obs_energy - energy) <= tv].copy()
    if not cand.empty:
        is_escape = cand["type"].isin(("SEP", "DEP")).to_numpy()
        if is_escape.any():
            keys = zip(cand["isotope"].to_numpy(), np.round(cand["line"].to_numpy(), 3))
            parent_seen = np.array([k in full_present for k in keys])
            cand = cand[~is_escape | parent_seen]
    if cand.empty:
        return cand, tv
    cand["abundance"] = (cand["isotope"].map(abundance).fillna(1.0)
                         if abundance is not None else 1.0)
    cand["element"] = (cand["isotope"].map(element).fillna(1.0)
                       if element is not None else 1.0)
    cand["line_abs"] = cand["strength"] * cand["weight"] * cand["abundance"] * cand["element"]
    denom = cand["isotope"].map(max_strength).replace(0.0, np.nan)
    cand["rel_line"] = (cand["weight"] * cand["strength"] / denom).fillna(cand["weight"])
    # Keep, per isotope, its strongest matching line at this energy.
    best = cand.loc[cand.groupby("isotope")["line_abs"].idxmax()].copy()
    best["support"] = best["isotope"].map(support).fillna(0.0)
    best["raw_abs"] = best["line_abs"] * best["support"]
    best["raw_rel"] = best["rel_line"] * best["abundance"] * best["element"] * best["support"]
    best["lines_seen"] = best["isotope"].map(lambda i: len(support_energies.get(i, [])))
    best["supporting"] = best["isotope"].map(
        lambda i: [e for e in support_energies.get(i, []) if abs(e - energy) > tv])
    return best, tv


# ── Stage 1: per-database ranking (top-N candidates per energy) ───────────────
_SCORE_COLS = ["Energy", "Rank", "Isotope", "Probability", "Matched line",
               "Match type", "Lines seen", "Supporting energies"]


def score_database(energies, std, tol=DEFAULT_TOL, escape_peaks=True, top_n=3,
                   weight_abundance=True, weight_element=True):
    """Top ``top_n`` candidate isotopes for each input energy **within one
    database** (long format: one row per candidate). ``Probability`` is
    normalised within this database. Set ``weight_abundance=False`` for
    source/decay libraries; ``weight_element=False`` disables the terrestrial
    element-naturalness prior."""
    ref = np.unique(np.asarray(list(energies), dtype=float))
    if ref.size == 0 or std.empty:
        return pd.DataFrame(columns=_SCORE_COLS)

    obs, max_strength, abundance, element = _database_evidence(
        std, escape_peaks, weight_abundance, weight_element)
    obs, support, support_energies, full_present = _mark_present(obs, ref, tol, max_strength)

    rows = []
    for energy in ref:
        best, _ = _candidates_for_energy(
            energy, obs, support, support_energies, max_strength, abundance,
            element, tol, full_present)
        if best.empty:
            rows.append({"Energy": float(energy), "Rank": 1, "Isotope": None,
                         "Probability": 0.0, "Matched line": None, "Match type": None,
                         "Lines seen": 0, "Supporting energies": []})
            continue
        score = best["raw_abs"].to_numpy(float)
        if score.sum() == 0:                       # strengthless library
            score = best["raw_rel"].to_numpy(float)
        best = best.iloc[np.argsort(score)[::-1]]
        score = np.sort(score)[::-1]
        total = score.sum()
        probs = score / total if total > 0 else np.zeros_like(score)
        for rank, (row, prob) in enumerate(
                zip(best.head(top_n).itertuples(), probs[:top_n]), start=1):
            rows.append({
                "Energy": float(energy), "Rank": rank, "Isotope": row.isotope,
                "Probability": round(float(prob), 4),
                "Matched line": round(float(row.line), 2), "Match type": row.type,
                "Lines seen": int(row.lines_seen), "Supporting energies": row.supporting,
            })
    return pd.DataFrame(rows, columns=_SCORE_COLS)


def identify(energies, databases=None, tol=DEFAULT_TOL, escape_peaks=True,
             top_n=3, weight_abundance=True, weight_element=True):
    """Stage 1 for every database: ``{database_name: score_database(...)}`` —
    the top ``top_n`` candidates for each database, per energy. Abundance and
    element-naturalness weighting are applied only to the reaction-on-natural-
    target libraries (:data:`ABUNDANCE_WEIGHTED`); source/decay libraries (lab
    sources, natural radiation, delayed activation) carry specific radionuclides,
    not sample elements, so neither prior applies."""
    names = list(DATABASES) if databases is None else list(databases)
    return {
        name: score_database(
            energies, standardize(load_database(name)), tol=tol,
            escape_peaks=escape_peaks, top_n=top_n,
            weight_abundance=weight_abundance and name in ABUNDANCE_WEIGHTED,
            weight_element=weight_element and name in ABUNDANCE_WEIGHTED)
        for name in names
    }


# ── Stage 2: comparable probabilities across databases ────────────────────────
_RANK_COLS = ["Energy", "Rank", "Database", "Isotope", "Probability",
              "Matched line", "Match type", "Lines seen", "Supporting energies"]


def rank_candidates(energies, databases=None, tol=DEFAULT_TOL, escape_peaks=True,
                    per_database=1, top_n=None, weight_abundance=True,
                    weight_element=True):
    """Pool the most likely isotope(s) from **each** database and give each a
    comparable probability (relative strength × abundance × element-naturalness ×
    corroboration). Long DataFrame, one row per candidate per energy, sorted by
    descending probability within each energy."""
    names = list(DATABASES) if databases is None else list(databases)
    ref = np.unique(np.asarray(list(energies), dtype=float))

    prepared = {}
    for name in names:
        std = standardize(load_database(name))
        if std.empty:
            continue
        weighted = weight_abundance and name in ABUNDANCE_WEIGHTED
        elemental = weight_element and name in ABUNDANCE_WEIGHTED
        obs, max_strength, abundance, element = _database_evidence(
            std, escape_peaks, weighted, elemental)
        obs, support, support_energies, full_present = _mark_present(obs, ref, tol, max_strength)
        prepared[name] = (obs, support, support_energies, max_strength, abundance,
                          element, full_present)

    records = []
    for energy in ref:
        pool = []
        for name, (obs, support, support_energies, max_strength, abundance, element, full_present) in prepared.items():
            best, _ = _candidates_for_energy(
                energy, obs, support, support_energies, max_strength, abundance,
                element, tol, full_present)
            if best.empty:
                continue
            score_abs = best["raw_abs"].to_numpy(float)
            if score_abs.sum() == 0:
                score_abs = best["raw_rel"].to_numpy(float)
            best = best.iloc[np.argsort(score_abs)[::-1]].head(per_database)
            for r in best.itertuples():
                pool.append({"Database": name, "Isotope": r.isotope,
                             "raw_rel": float(r.raw_rel),
                             "Matched line": round(float(r.line), 2),
                             "Match type": r.type, "Lines seen": int(r.lines_seen),
                             "Supporting energies": r.supporting})
        if not pool:
            records.append({"Energy": float(energy), "Rank": 1, "Database": None,
                            "Isotope": None, "Probability": 0.0, "Matched line": None,
                            "Match type": None, "Lines seen": 0, "Supporting energies": []})
            continue
        total = sum(p["raw_rel"] for p in pool) or 1.0
        for p in pool:
            p["Probability"] = round(p["raw_rel"] / total, 4)
        pool.sort(key=lambda p: p["Probability"], reverse=True)
        if top_n is not None:
            pool = pool[:top_n]
        for rank, p in enumerate(pool, start=1):
            records.append({
                "Energy": float(energy), "Rank": rank, "Database": p["Database"],
                "Isotope": p["Isotope"], "Probability": p["Probability"],
                "Matched line": p["Matched line"], "Match type": p["Match type"],
                "Lines seen": p["Lines seen"], "Supporting energies": p["Supporting energies"],
            })
    return pd.DataFrame(records, columns=_RANK_COLS)


def escape_relations(energies, tol=DEFAULT_TOL):
    """Flag which observed energies may be single/double-escape peaks of *other*
    observed energies (E ≈ parent − 511 keV for a SEP, parent − 1022 for a DEP).

    Returns ``{energy: [(label, parent_energy), …]}`` (empty list when none).
    Escape-peak amplitudes depend on detector size, so this is reported as a
    possibility — kept out of the probability ranking in :func:`identify`."""
    arr = sorted({float(e) for e in energies})
    relations = {e: [] for e in arr}
    # Escape peaks come from pair production, which requires the parent gamma to
    # be above the pair-production threshold (2 mₑc² = 1022 keV); a lower-energy
    # peak cannot have escape peaks, so don't propose it as a parent.
    pair_threshold = 2 * ELECTRON_MASS_KEV
    # An escape peak's parent must be a *full-energy* peak, never another escape
    # peak — otherwise an escape chains to a higher escape (e.g. a DEP at
    # parent−1022 also matches "SEP of (parent−511)", which is itself an SEP).
    # Resolve top-down: a peak is full-energy unless it is the escape of a
    # higher peak already classified as full-energy, and only full-energy peaks
    # are offered as parents.
    full_energy = set()
    for e in sorted(arr, reverse=True):
        is_escape = False
        for label, offset in (("SEP", ELECTRON_MASS_KEV), ("DEP", 2 * ELECTRON_MASS_KEV)):
            parent = e + offset
            tv = float(tol(parent)) if callable(tol) else float(tol)
            match = min((p for p in full_energy
                         if p > pair_threshold and abs(p - parent) <= tv),
                        key=lambda p: abs(p - parent), default=None)
            if match is not None:
                relations[e].append((label, match))
                is_escape = True
        if not is_escape:
            full_energy.add(e)
    return relations


def identify_best(energies, databases=None, tol=DEFAULT_TOL, escape_peaks=True,
                  weight_abundance=True):
    """The single most likely isotope per energy across all databases (the
    rank-1 row of :func:`rank_candidates`)."""
    ranked = rank_candidates(energies, databases=databases, tol=tol,
                             escape_peaks=escape_peaks, per_database=1,
                             weight_abundance=weight_abundance)
    if ranked.empty:
        return ranked
    return ranked[ranked["Rank"] == 1].drop(columns="Rank").reset_index(drop=True)
