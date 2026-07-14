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
    assert list(std.columns[:3]) == ["Isotope", "Energy", "Strength"]
    # HalfLife is carried through only for libraries that have it.
    extra = set(std.columns[3:])
    assert extra <= {"HalfLife"}
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
    # Isotope Xx: a strong, corroborated cluster. Isotope Yy: a lone weak line
    # near Xx's main peak (a coincidence to be out-voted). Placeholder symbols
    # are deliberately non-elements so the element-naturalness prior is neutral.
    return pd.DataFrame({
        "Isotope": ["Xx", "Xx", "Xx", "Yy"],
        "Energy": [6000.0, 3000.0, 1500.0, 6000.5],
        "Strength": [100.0, 60.0, 30.0, 8.0],
    })


def test_strength_and_corroboration_pick_the_right_isotope(synthetic_std):
    # Xx's main line + another Xx line are present; Yy only coincides at 6000.
    res = nid.score_database([6000.0, 3000.0], synthetic_std, tol=2.0)
    row = res.loc[res["Energy"] == 6000.0].iloc[0]
    assert row["Isotope"] == "Xx"
    assert row["Probability"] > 0.9
    assert row["Lines seen"] == 2                      # 6000 and 3000
    assert 3000.0 in row["Supporting energies"]


def test_escape_peak_is_attributed_to_parent_line(synthetic_std):
    # 5489 = 6000 - 511 → single-escape peak of Xx's 6000 line.
    res = nid.score_database([6000.0, 5489.0], synthetic_std, tol=2.0)
    sep = res.loc[res["Energy"] == 5489.0].iloc[0]
    assert sep["Isotope"] == "Xx"
    assert sep["Match type"] == "SEP"
    assert sep["Matched line"] == pytest.approx(6000.0)


def test_escape_peak_dropped_when_parent_absent(synthetic_std):
    # 5489 = 6000 - 511 is X's single-escape peak, but the 6000 line itself is
    # NOT in the observed list → the escape match is spurious and dropped.
    # (The complementary "kept when 6000 is present" case is covered by
    # test_escape_peak_is_attributed_to_parent_line.)
    res = nid.score_database([5489.0], synthetic_std, tol=2.0)
    assert res.iloc[0]["Isotope"] is None


def test_element_factor_favours_common_over_rare():
    # Common rock/air/bio elements outrank rare ones, on the log-compressed scale.
    assert nid.element_factor("16O") > nid.element_factor("56Fe") > nid.element_factor("14N")
    assert nid.element_factor("14N") > nid.element_factor("155Gd") > nid.element_factor("84Kr")
    # Mass-tagged, natural-target ('00X') and bare symbols all resolve the element.
    assert nid.element_factor("16O") == nid.element_factor("00O") == nid.element_factor("O")
    # Unknown / unparseable → neutral 1.0, so the prior never erases a candidate.
    assert nid.element_factor("Zz") == 1.0
    assert nid.element_factor("???") == 1.0
    # Rare elements keep a positive floor rather than zero.
    assert nid.element_factor("84Kr") == nid.NATURAL_WEIGHT_FLOOR


def test_element_prior_breaks_common_vs_rare_tie():
    # Two candidate elements at one energy with identical strength: the common
    # element (Fe) must outrank the rare one (Gd) once the prior is on, and the
    # ranking is a pure tie (alphabetical by score) when it is off.
    std = pd.DataFrame({"Isotope": ["56Fe", "155Gd"], "Energy": [1000.0, 1000.0],
                        "Strength": [10.0, 10.0]})
    on = nid.score_database([1000.0], std, tol=2.0, weight_element=True)
    assert on.sort_values("Rank").iloc[0]["Isotope"] == "56Fe"
    assert on.iloc[0]["Probability"] > on.iloc[1]["Probability"]
    off = nid.score_database([1000.0], std, tol=2.0,
                             weight_abundance=False, weight_element=False)
    assert off.iloc[0]["Probability"] == pytest.approx(off.iloc[1]["Probability"])


def test_escape_relations_detects_sep_and_dep():
    # 5618.9 = 6129.9 - 511 (SEP); 5107.9 = 6129.9 - 1022 (DEP).
    rel = nid.escape_relations([6129.9, 5618.9, 5107.9], tol=lambda e: max(2.0, 0.001 * e))
    assert ("SEP", 6129.9) in rel[5618.9]
    assert ("DEP", 6129.9) in rel[5107.9]
    assert rel[6129.9] == []        # the full peak is nobody's escape here


def test_escape_relations_do_not_chain_through_escape_peaks():
    # The DEP (5107.9) is exactly one 511 keV below the SEP (5618.9), so a naive
    # search also flags it as "SEP of 5618.9". But 5618.9 is itself an escape
    # peak, not a real gamma — the DEP must point only at the full-energy parent.
    rel = nid.escape_relations([6129.9, 5618.9, 5107.9],
                               tol=lambda e: max(10.0, 0.002 * e))
    assert rel[5107.9] == [("DEP", 6129.9)]      # no spurious ("SEP", 5618.9)
    assert rel[5618.9] == [("SEP", 6129.9)]


def test_escape_relations_parent_must_clear_pair_threshold():
    # A parent below 2·mₑc² (1022 keV) cannot pair-produce, so 289 keV must not
    # be offered as the SEP of an 800 keV peak.
    rel = nid.escape_relations([289.0, 800.0], tol=lambda e: max(10.0, 0.002 * e))
    assert rel[289.0] == []
    assert rel[800.0] == []


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
    std = pd.DataFrame({"Isotope": ["Aa", "Aa", "Bb"],
                        "Energy": [500.0, 900.0, 500.2], "Strength": [0.0, 0.0, 0.0]})
    res = nid.score_database([500.0, 900.0], std, tol=2.0)
    row = res.loc[res["Energy"] == 500.0].iloc[0]
    # Aa is corroborated by its 900 keV line; Bb is a lone coincidence.
    assert row["Isotope"] == "Aa"


# ── proximity weighting ───────────────────────────────────────────────────────
def test_proximity_prefers_nearest_line():
    # Yy's line is stronger but 4.7 keV off; Xx sits 0.14 keV from the query.
    # Inside a wide window, the near-exact match must win despite the strength.
    std = pd.DataFrame({"Isotope": ["Xx", "Yy"], "Energy": [661.66, 666.5],
                        "Strength": [85.0, 100.0]})
    res = nid.score_database([661.8], std, tol=6.6)
    assert res.iloc[0]["Isotope"] == "Xx"
    assert res.iloc[0]["Probability"] > 0.99


# ── half-life procurability prior (check-source library) ─────────────────────
def test_half_life_factor_scale():
    year = 3.156e7
    assert nid.half_life_factor(30.1 * year) > nid.half_life_factor(272 * 86400)  # 137Cs > 57Co
    assert nid.half_life_factor(272 * 86400) > nid.half_life_factor(8280)         # 57Co > 132I
    assert nid.half_life_factor(8280) == nid.HALF_LIFE_FLOOR                      # short-lived → floor
    assert nid.half_life_factor(432 * year) == nid.HALF_LIFE_CAP                  # 241Am → cap
    # Unknown / invalid half-lives stay neutral.
    assert nid.half_life_factor(None) == 1.0
    assert nid.half_life_factor(float("nan")) == 1.0
    assert nid.half_life_factor(-5.0) == 1.0


def test_half_life_prior_applies_only_to_lab_sources():
    assert nid.HALF_LIFE_WEIGHTED == {"Common lab sources"}


# ── decay-chain duplicate merging ─────────────────────────────────────────────
def test_decay_duplicates_merged_in_lab_sources():
    # 661.657 keV is listed under both 137Cs (the source one owns) and its
    # daughter 137Ba (the actual emitter, 137mBa). standardize must keep only
    # the parent so the two entries don't split one probability.
    std = nid.standardize(nid.load_database("Common lab sources"))
    near = std[std["Energy"].sub(661.657).abs() < 0.1]
    assert "137Cs" in set(near["Isotope"])
    assert "137Ba" not in set(near["Isotope"])


# ── regression: 661.8 keV in a busy spectrum is 137Cs, not 126Sb ──────────────
def test_cs137_wins_661_8_kev_in_lab_sources():
    # With a flat window, 126Sb (99.6 % line at 666.5 keV, 12-day fission
    # product) used to out-rank 137Cs (85.1 % at 661.657 keV, the most common
    # check source): proximity + half-life priors must flip that decisively.
    std = nid.standardize(nid.load_database("Common lab sources"))
    res = nid.score_database([661.8], std, tol=lambda e: max(5.0, 0.01 * e),
                             escape_peaks=False, weight_abundance=False,
                             weight_element=False, weight_half_life=True)
    top = res.iloc[0]
    assert top["Isotope"] == "137Cs"
    assert top["Probability"] > 0.9


def test_identify_gates_half_life_prior_by_library():
    # identify() must apply the procurability prior to Common lab sources but
    # not to Delayed activation (whose products are legitimately short-lived).
    out = nid.identify([661.8], databases=["Common lab sources"],
                       tol=lambda e: max(5.0, 0.01 * e))
    assert out["Common lab sources"].iloc[0]["Isotope"] == "137Cs"


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


# ── atomic number (Z) lookup and range filtering ──────────────────────────────
def test_atomic_number_resolves_isotopes_and_bare_symbols():
    assert nid.atomic_number("16O") == 8
    assert nid.atomic_number("137Cs") == 55
    assert nid.atomic_number("Fe") == 26          # bare symbol
    assert nid.atomic_number("Gd") == 64
    assert nid.atomic_number("???") is None       # unparseable → None


def test_filter_by_z_drops_out_of_range_keeps_unknown():
    std = pd.DataFrame({"Isotope": ["16O", "137Cs", "Gd", "??"],
                        "Energy": [1.0, 2.0, 3.0, 4.0], "Strength": [1.0] * 4})
    kept = nid.filter_by_z(std, (1, 20))          # only O (Z=8) is in range
    assert set(kept["Isotope"]) == {"16O", "??"}  # unknown-Z row is never erased
    # Full-range span is a no-op (returns every row).
    assert len(nid.filter_by_z(std, (1, 118))) == len(std)
    assert len(nid.filter_by_z(std, None)) == len(std)


def test_identify_z_range_excludes_elements():
    # 6129.9 keV is 16O (Z=8) in TALYS; a high-Z window removes it entirely.
    def top(res):
        f = res["TALYS 14 MeV"]
        f = f[f["Isotope"].notna()]
        return None if f.empty else f.iloc[0]["Isotope"]

    dbs = ["TALYS 14 MeV"]
    assert top(nid.identify([6129.9], databases=dbs, top_n=1)) == "16O"
    assert top(nid.identify([6129.9], databases=dbs, top_n=1, z_range=(1, 20))) == "16O"
    assert top(nid.identify([6129.9], databases=dbs, top_n=1, z_range=(50, 80))) is None


def test_identify_database_selection_subsets():
    res = nid.identify([661.7], databases=["Common lab sources"])
    assert list(res) == ["Common lab sources"]


# ── line_details: raw strength + decay/daughter for exports ────────────────────
def test_line_details_decay_library():
    d = nid.line_details("Common lab sources", "137Cs", 661.657)
    assert d["strength_label"] == "Intensity (%)"
    assert d["strength"] > 0
    assert d["decay"] == "B-"
    assert d["daughter"] == "Ba"           # element only; export builds "137Ba"


def test_line_details_reaction_library_has_no_daughter():
    d = nid.line_details("TALYS 14 MeV", "16O", 6129.89)
    assert d["strength_label"] == "XS (mb)" and d["strength"] > 0
    assert "daughter" not in d and "decay" not in d


def test_line_details_no_match_returns_empty():
    assert nid.line_details("TALYS 14 MeV", "16O", 12.3) == {}
