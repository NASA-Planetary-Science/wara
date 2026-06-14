"""Shared isotope-ID helpers for the beta GUI.

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
from wara import nuclide_identificator as nid

from . import theme as T


def default_tol(energy):
    """Energy-matching window: ~0.1 % of E, with a 2 keV floor."""
    return max(2.0, 0.001 * energy)


def identify(energies, tol=None, top_n=2):
    """Identify *energies* across all databases (per-database top-N), using
    full-energy lines only — escape peaks are excluded from the ranking."""
    return nid.identify(energies, tol=tol or default_tol, top_n=top_n,
                        escape_peaks=False)


def escape_relations(energies, tol=None):
    """``{energy: [(label, parent_energy), …]}`` escape-peak possibilities."""
    return nid.escape_relations(energies, tol=tol or default_tol)


def _best_isotope(parent_energy, results):
    """Most likely isotope identified for *parent_energy* across the databases.
    Ranked by corroboration (lines seen) then probability, since probabilities
    are not comparable between databases."""
    best, best_key = None, (-1, -1.0)
    for frame in results.values():
        rows = frame[(abs(frame["Energy"] - parent_energy) < 1e-6)
                     & frame["Isotope"].notna()]
        if rows.empty:
            continue
        r = rows.iloc[0]
        key = (int(r.get("Lines seen", 0)), float(r["Probability"]))
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
