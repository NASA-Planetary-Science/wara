"""Shared isotope-ID helpers for the wara GUI.

Runs nuclide identification for peak energies (wara.nuclide_identificator) and
renders the colour-coded HTML used by both the Spectrum-tab hover overlay and
the Drag-and-Fit window's per-line ID popup: the energy in green, each matching
database's name in amber (matching the "Nuclear Database" button), and its top
isotope candidate(s) — the **isotope names in cyan** — with a probability.

Single/double-escape peaks (SEP/DEP) are kept OUT of the probability ranking
(their amplitude depends on detector size); instead, when an observed energy is
a plausible escape of another observed peak it is shown as a separate
possibility (``possibly SEP of <parent> keV (<isotope>)``). Candidates that round
to 0 % are dropped.
"""
import numpy as np

from wara import nuclide_identificator as nid

from . import theme as T

# Window used when no detector-resolution model is available. The old default
# (~1 % of E, floored at 2 keV) was far narrower than a real photopeak and
# missed true matches; without an FWHM to size the window we open it up.
FALLBACK_FLOOR = 5.0


def default_tol(energy):
    """Energy-matching window when no resolution model is available: at least
    10 keV, widening to ~1 % of E at high energies."""
    return max(FALLBACK_FLOOR, 0.01 * energy)


def fwhm_tol(search, floor=FALLBACK_FLOOR):
    """Build a ``tol(energy) -> keV`` matching window sized to 2×FWHM.

    A two-FWHM window is wide enough to still catch slightly off-calibrated
    lines. *search* is a :class:`wara.peaksearch.PeakSearch`; its
    ``fwhm(channel)`` resolution model gives the FWHM in *channels*, which we
    convert to keV via the spectrum's local calibration dispersion. The window
    is never narrower than *floor* (≥10 keV), and falls back to
    :func:`default_tol` when no usable search/calibration is available.
    """
    spect = getattr(search, "spectrum", None) if search is not None else None
    energies = getattr(spect, "energies", None) if spect is not None else None
    channels = getattr(spect, "channels", None) if spect is not None else None
    if energies is None or channels is None:
        return default_tol
    energies = np.asarray(energies, dtype=float)
    channels = np.asarray(channels, dtype=float)
    disp = np.abs(np.gradient(energies, channels))   # keV per channel

    def tol(energy):
        i = int(np.argmin(np.abs(energies - energy)))
        fwhm_kev = float(search.fwhm(channels[i])) * disp[i]
        return max(floor, 2.0 * fwhm_kev)

    return tol


def identify(energies, tol=None, top_n=2, databases=None, z_range=None):
    """Identify *energies* (per-database top-N), using full-energy lines only —
    escape peaks are excluded from the ranking. ``databases`` restricts which
    libraries are consulted (``None`` = all); ``z_range`` = ``(z_min, z_max)``
    restricts candidates to that atomic-number span (``None`` = all elements)."""
    return nid.identify(energies, databases=databases, tol=tol or default_tol,
                        top_n=top_n, escape_peaks=False, z_range=z_range)


def _daughter_isotope(parent, daughter_symbol):
    """Full daughter isotope from a parent name and the library's daughter
    element (e.g. ``'137Cs'`` + ``'Ba'`` → ``'137Ba'``). The lab-source decays
    (β⁻/EC/β⁺/IT) conserve mass number, so the daughter takes the parent's mass.
    Falls back to the bare element symbol when the parent's mass can't be read."""
    sym = daughter_symbol.strip().capitalize()
    m = nid._ISO_RE.match(str(parent))
    return f"{m.group(1)}{sym}" if m else sym


def best_per_database_table(energies, tol=None, databases=None, z_range=None):
    """Table of the top isotope candidate **per database** for each energy.

    Returns a :class:`pandas.DataFrame` (one row per energy per database that has
    a match), ready to save, with the same per-database probabilities the hover
    overlay shows plus, from the original library, the line strength (cross
    section or intensity) and — for decay libraries — the decay mode and daughter
    isotope (the parent listed under ``Isotope`` decays to the daughter that
    emits the gamma). Respects the ``databases`` and ``z_range`` selection."""
    import pandas as pd

    results = identify(energies, tol=tol, top_n=1, databases=databases,
                       z_range=z_range)
    rows = []
    for e in energies:
        for db, frame in results.items():
            hits = frame[(abs(frame["Energy"] - e) < 1e-6)
                         & frame["Isotope"].notna()]
            if hits.empty:
                continue
            r = hits.iloc[0]
            iso = r["Isotope"]
            m = nid._ELEMENT_RE.match(str(iso))
            element = m.group(1).capitalize() if m else ""
            line = r.get("Matched line")
            details = nid.line_details(db, iso, line if line is not None else e)
            daughter = details.get("daughter")
            rows.append({
                "Energy (keV)": round(float(e), 2),
                "Database": db,
                "Isotope (parent)": iso,
                "Element": element,
                "Daughter": _daughter_isotope(iso, daughter) if daughter else "",
                "Decay": details.get("decay", ""),
                "Line strength": details.get("strength"),
                "Strength type": details.get("strength_label", ""),
                "Probability (%)": round(float(r["Probability"]) * 100, 1),
                "Matched line (keV)": line,
                "Lines seen": int(r.get("Lines seen", 0)),
            })
    columns = ["Energy (keV)", "Database", "Isotope (parent)", "Element",
               "Daughter", "Decay", "Line strength", "Strength type",
               "Probability (%)", "Matched line (keV)", "Lines seen"]
    return pd.DataFrame(rows, columns=columns)


def escape_relations(energies, tol=None):
    """``{energy: [(label, parent_energy), …]}`` escape-peak possibilities."""
    return nid.escape_relations(energies, tol=tol or default_tol)


def _best_isotope(parent_energy, results):
    """Most likely isotope identified for *parent_energy* across the databases.
    Ranked by probability scaled by the element-naturalness prior (so a common
    element like O/Fe outranks a rare one like Gd at similar probability), with
    corroboration (lines seen) as a tiebreaker."""
    best, best_key = None, (-1.0, -1)
    for frame in results.values():
        rows = frame[(abs(frame["Energy"] - parent_energy) < 1e-6)
                     & frame["Isotope"].notna()]
        if rows.empty:
            continue
        r = rows.iloc[0]
        score = float(r["Probability"]) * nid.element_factor(r["Isotope"])
        key = (score, int(r.get("Lines seen", 0)))
        if key > best_key:
            best, best_key = r["Isotope"], key
    return best


def format_html(energy, results, escapes=None):
    """Colour-coded HTML for one energy: full-energy candidates per database plus
    any escape-peak possibilities (``escapes`` = ``[(label, parent_energy), …]``)."""
    lines = [f"<span style='color:{T.ACCENT_GREEN}; font-weight:700'>"
             f"{energy:.1f} keV</span>"]
    for db, frame in results.items():
        rows = frame[(abs(frame["Energy"] - energy) < 1e-6) & frame["Isotope"].notna()]
        cands = []
        for _, r in rows.head(2).iterrows():
            if round(r["Probability"] * 100) < 1:
                continue
            cands.append(
                f"<span style='color:{T.ACCENT_CYAN}'>{r['Isotope']}</span> "
                f"<span style='color:{T.TEXT_PRIMARY}'>{r['Probability'] * 100:.0f}%</span>")
        if cands:
            lines.append(f"<span style='color:{T.ACCENT_AMBER}'>{db}:</span> "
                         + ", ".join(cands))
    for label, parent in (escapes or []):
        iso = _best_isotope(parent, results)
        of = f" (<span style='color:{T.ACCENT_CYAN}'>{iso}</span>)" if iso else ""
        lines.append(f"<span style='color:{T.TEXT_DIM}'>possibly {label} of "
                     f"{parent:.1f} keV{of}</span>")
    if len(lines) == 1:
        lines.append(f"<span style='color:{T.TEXT_DIM}'>no database match</span>")
    return "<br>".join(lines)


def lookup_html(energy, tol=None, top_n=2, context=None):
    """Identify a single energy (with optional *context* energies for escape-peak
    detection) and return its colour-coded HTML directly."""
    energies = [energy] + list(context or [])
    results = identify(energies, tol=tol, top_n=top_n)
    escapes = escape_relations(energies, tol=tol).get(energy)
    return format_html(energy, results, escapes)
