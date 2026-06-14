"""Tests for wara.nuclide_identificator — likely-isotope identification from a
list of peak energies (relative cross sections + corroborating lines / escape
peaks)."""
import numpy as np
import pandas as pd
import pytest

from wara import nuclide_identificator as nid


# ── standardize: unify the different strength columns ─────────────────────────
@pytest.mark.parametrize("db, strength_col", [
    ("TALYS 14 MeV", "XS (mb)"),
    ("Neutron capture (IAEA)", "Sigma (mb)"),
    ("Common lab sources", "Intensity (%)"),
])
def test_standardize_picks_strength_column(db, strength_col):
    raw = nid.load_database(db)
    assert strength_col in raw.columns
    std = nid.standardize(raw)
    assert list(std.columns) == ["Isotope", "Energy", "Strength"]
    assert (std["Strength"] >= 0).all()
    assert std["Energy"].notna().all()


# ── observable table: escape peaks only for high-energy gammas ────────────────
def test_observable_table_adds_escape_peaks_above_threshold():
    std = pd.DataFrame({"Isotope": ["X", "X"], "Energy": [6000.0, 800.0],
                        "Strength": [100.0, 50.0]})
    obs = nid._observable_table(std, escape_peaks=True)
    # 6000 keV → full + SEP + DEP; 800 keV (< 1022) → full only.
    high = obs[obs["line"] == 6000.0]
    low = obs[obs["line"] == 800.0]
    assert set(high["type"]) == {"full", "SEP", "DEP"}
    assert set(low["type"]) == {"full"}
    sep = high[high["type"] == "SEP"]["obs_energy"].iloc[0]
    dep = high[high["type"] == "DEP"]["obs_energy"].iloc[0]
    assert sep == pytest.approx(6000 - 511)
    assert dep == pytest.approx(6000 - 1022)


def test_observable_table_can_disable_escape_peaks():
    std = pd.DataFrame({"Isotope": ["X"], "Energy": [6000.0], "Strength": [100.0]})
    obs = nid._observable_table(std, escape_peaks=False)
    assert set(obs["type"]) == {"full"}


# ── scoring on a small synthetic database ─────────────────────────────────────
@pytest.fixture
def synthetic_std():
    # Isotope X: a strong, corroborated cluster. Isotope Y: a lone weak line
    # near X's main peak (a coincidence to be out-voted).
    return pd.DataFrame({
        "Isotope": ["X", "X", "X", "Y"],
        "Energy": [6000.0, 3000.0, 1500.0, 6000.5],
        "Strength": [100.0, 60.0, 30.0, 8.0],
    })


def test_strength_and_corroboration_pick_the_right_isotope(synthetic_std):
    # X's main line + another X line are present; Y only coincides at 6000.
    res = nid.score_database([6000.0, 3000.0], synthetic_std, tol=2.0)
    row = res.loc[res["Energy"] == 6000.0].iloc[0]
    assert row["Isotope"] == "X"
    assert row["Probability"] > 0.9
    assert row["Lines seen"] == 2                      # 6000 and 3000
    assert 3000.0 in row["Supporting energies"]


def test_escape_peak_is_attributed_to_parent_line(synthetic_std):
    # 5489 = 6000 - 511 → single-escape peak of X's 6000 line.
    res = nid.score_database([6000.0, 5489.0], synthetic_std, tol=2.0)
    sep = res.loc[res["Energy"] == 5489.0].iloc[0]
    assert sep["Isotope"] == "X"
    assert sep["Match type"] == "SEP"
    assert sep["Matched line"] == pytest.approx(6000.0)


def test_escape_peak_dropped_when_parent_absent(synthetic_std):
    # 5489 = 6000 - 511 is X's single-escape peak, but the 6000 line itself is
    # NOT in the observed list → the escape match is spurious and dropped.
    # (The complementary "kept when 6000 is present" case is covered by
    # test_escape_peak_is_attributed_to_parent_line.)
    res = nid.score_database([5489.0], synthetic_std, tol=2.0)
    assert res.iloc[0]["Isotope"] is None


def test_escape_relations_detects_sep_and_dep():
    # 5618.9 = 6129.9 - 511 (SEP); 5107.9 = 6129.9 - 1022 (DEP).
    rel = nid.escape_relations([6129.9, 5618.9, 5107.9], tol=lambda e: max(2.0, 0.001 * e))
    assert ("SEP", 6129.9) in rel[5618.9]
    assert ("DEP", 6129.9) in rel[5107.9]
    assert rel[6129.9] == []        # the full peak is nobody's escape here


def test_cs137_peak_not_outranked_by_escape_peaks():
    # A lone 661.6 keV peak (Cs-137 photopeak): the 60Co 1173-SEP (~662 keV) and
    # other escape coincidences must be gone, since their parent peaks are absent.
    res = nid.identify([661.6], tol=lambda e: max(2.0, 0.001 * e), top_n=3)
    common = res["Common lab sources"]
    common = common[common["Isotope"].notna()]
    assert common.iloc[0]["Isotope"] in ("137Ba", "137Cs")
    assert "60Co" not in set(common["Isotope"])
    # No database should report an escape-peak match for this lone peak.
    for frame in res.values():
        matched = frame[frame["Isotope"].notna()]
        assert (matched["Match type"] == "full").all()


def test_unmatched_energy_returns_no_isotope(synthetic_std):
    res = nid.score_database([100.0], synthetic_std, tol=2.0)
    row = res.iloc[0]
    assert row["Isotope"] is None
    assert row["Probability"] == 0.0


def test_strengthless_database_falls_back_to_corroboration():
    std = pd.DataFrame({"Isotope": ["A", "A", "B"],
                        "Energy": [500.0, 900.0, 500.2], "Strength": [0.0, 0.0, 0.0]})
    res = nid.score_database([500.0, 900.0], std, tol=2.0)
    row = res.loc[res["Energy"] == 500.0].iloc[0]
    # A is corroborated by its 900 keV line; B is a lone coincidence.
    assert row["Isotope"] == "A"


# ── natural isotopic abundance ────────────────────────────────────────────────
def test_abundance_factor_known_and_unknown():
    assert nid.abundance_factor("56Fe") == pytest.approx(0.91754, abs=1e-3)
    assert nid.abundance_factor("57Fe") == pytest.approx(0.02119, abs=1e-3)
    assert nid.abundance_factor("16O") > 0.99
    assert nid.abundance_factor("00Fe") == 1.0     # natural-element target
    assert nid.abundance_factor("137Cs") == 1.0    # radioactive → neutral
    assert nid.abundance_factor("X") == 1.0        # unparseable → neutral


def test_abundance_weighting_picks_56fe_over_57fe():
    # At 846.78 keV TALYS lists both 57Fe (bigger XS) and 56Fe; abundance
    # (~92 % vs ~2 %) must flip the winner to 56Fe.
    std = nid.standardize(nid.load_database("TALYS 14 MeV"))
    weighted = nid.score_database([846.8], std, tol=2.0, weight_abundance=True)
    bare = nid.score_database([846.8], std, tol=2.0, weight_abundance=False)
    assert weighted.iloc[0]["Isotope"] == "56Fe"
    assert bare.iloc[0]["Isotope"] == "57Fe"


def test_identify_abundance_weighted_only_for_reaction_libraries():
    # Source/decay libraries must NOT be abundance-weighted (would erase e.g. 40K).
    assert "TALYS 14 MeV" in nid.ABUNDANCE_WEIGHTED
    assert "Common lab sources" not in nid.ABUNDANCE_WEIGHTED
    assert "Natural radiation" not in nid.ABUNDANCE_WEIGHTED


# ── stage 1 returns ranked candidates per database, per energy ────────────────
def test_score_database_returns_top_n_ranked_rows():
    std = nid.standardize(nid.load_database("TALYS 14 MeV"))
    res = nid.score_database([846.8], std, tol=2.0, top_n=3)
    block = res[res["Energy"] == 846.8]
    assert list(block["Rank"]) == [1, 2, 3]
    assert block["Probability"].is_monotonic_decreasing
    assert block.iloc[0]["Isotope"] == "56Fe"


# ── integration: the 16O example on the real TALYS library ────────────────────
def test_o16_identified_in_talys_with_escape_peaks():
    energies = [6129.9, 5618.9, 5107.9, 6917.1, 3684.5, 2742.0]
    res = nid.score_database(
        energies, nid.standardize(nid.load_database("TALYS 14 MeV")), tol=3.0)

    main = res.loc[res["Energy"] == 6129.9].iloc[0]
    assert main["Isotope"] == "16O"
    assert main["Probability"] > 0.9
    assert main["Lines seen"] >= 5                  # strongly corroborated

    # The escape peaks resolve to 16O's 6129 line, tagged SEP / DEP.
    sep = res.loc[res["Energy"] == 5618.9].iloc[0]
    dep = res.loc[res["Energy"] == 5107.9].iloc[0]
    assert sep["Isotope"] == "16O" and sep["Match type"] == "SEP"
    assert dep["Isotope"] == "16O" and dep["Match type"] == "DEP"
    assert sep["Matched line"] == pytest.approx(6129.89, abs=0.5)


def test_identify_returns_a_frame_per_database():
    out = nid.identify([6129.9, 6917.1], databases=["TALYS 14 MeV", "Common lab sources"])
    assert set(out) == {"TALYS 14 MeV", "Common lab sources"}
    assert out["TALYS 14 MeV"].loc[
        lambda d: d["Energy"] == 6129.9, "Isotope"].iloc[0] == "16O"


def test_identify_best_collapses_across_databases():
    energies = [6129.9, 5618.9, 5107.9, 6917.1, 3684.5, 2742.0]
    best = nid.identify_best(energies, tol=3.0)
    assert len(best) == len(set(energies))
    row = best.loc[best["Energy"] == 6129.9].iloc[0]
    assert row["Isotope"] == "16O"
    assert row["Database"] == "TALYS 14 MeV"


# ── stage 2: comparable probabilities across databases ────────────────────────
def test_rank_candidates_probabilities_are_comparable_and_sum_to_one():
    energies = [6129.9, 5618.9, 5107.9, 6917.1, 3684.5, 2742.0]
    ranked = nid.rank_candidates(energies, tol=3.0)
    # One block per energy; probabilities within an energy form a distribution.
    for energy, block in ranked.groupby("Energy"):
        assert block["Probability"].sum() == pytest.approx(1.0, abs=2e-3)
        assert list(block["Rank"]) == sorted(block["Rank"])      # ranked order
        assert block["Probability"].is_monotonic_decreasing

    main = ranked[(ranked["Energy"] == 6129.9) & (ranked["Rank"] == 1)].iloc[0]
    assert main["Isotope"] == "16O"
    assert main["Database"] == "TALYS 14 MeV"
    # 15N (a 16O activation product) shows up as a cross-database alternative.
    others = ranked[ranked["Energy"] == 6129.9]["Isotope"].tolist()
    assert "15N" in others


def test_rank_candidates_top_n_limits_pool():
    ranked = nid.rank_candidates([6129.9, 6917.1], tol=3.0, top_n=2)
    assert ranked.groupby("Energy").size().max() <= 2


def test_callable_tolerance_widens_window():
    # 16O 6129.89; query 6133 keV is ~3.1 keV away — outside a 2 keV window,
    # inside a generous energy-proportional one.
    std = nid.standardize(nid.load_database("TALYS 14 MeV"))
    tight = nid.score_database([6133.0], std, tol=2.0).iloc[0]
    wide = nid.score_database([6133.0], std, tol=lambda e: 0.001 * e).iloc[0]
    assert tight["Isotope"] != "16O"          # nothing within 2 keV
    assert wide["Isotope"] == "16O"           # ~6.1 keV window catches it
