"""Shared test configuration.

Everything here is about keeping *per-test* time small. The suite has two
fixed, process-wide costs that otherwise land on whichever test happens to
touch them first, making that one test look pathologically slow:

* the 13 bundled nuclide databases (read + standardised once, then cached in
  :mod:`wara.nuclide_identificator`),
* the NIST natural-abundance table (one pass over a ~27k-line file), and
* ``dateparser``'s language data, which its first ``parse()`` call loads
  (~0.9 s) and then reuses for the rest of the process.

Warming happens in ``pytest_collection_finish`` rather than in an autouse
session fixture, because a session fixture is *reported* as the setup of
whichever test happens to trigger it first -- that just moves the misleading
number around. Warming after collection charges the ~1.5 s to the collection
phase, where it belongs, so every reported per-test duration reflects only that
test's own work.

It is skipped when a single test module is being run: a targeted run shouldn't
pay 1.5 s it may not need, and there the cost simply stays where it lands.
"""

import os

# Must be set before anything creates a QApplication or picks a MPL backend.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")

import pytest


def _warm_process_caches():
    """Populate the process-wide caches the suite shares."""
    from wara import nuclide_identificator as nid
    for name in nid.DATABASES:
        nid._standardized(name)
    from wara import parse_NIST
    parse_NIST._abundance_index()
    # It is the *failure* path that is slow: an unparseable string makes
    # dateparser try every language, loading all of its locale data (~0.9 s).
    # A string that parses cleanly does not warm this up.
    import dateparser
    dateparser.parse("wara-warmup-not-a-date")
    # wara.gui.fitting imports this lazily (it pulls in the identifier), so
    # without this the import lands on the first test that clicks an ID button.
    from wara.gui import isotope_id  # noqa: F401


def pytest_collection_finish(session):
    """Warm the shared caches once collection knows what will actually run."""
    modules = {getattr(item, "module", None) for item in session.items}
    if len(modules) > 1:
        _warm_process_caches()


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close any figures a test leaves behind.

    Keeps pyplot's global registry from accumulating figures across the ~900
    tests. Worth about 1.5 s over the suite -- modest, but it also stops the
    "More than 20 figures have been opened" warning from firing.
    """
    yield
    import matplotlib.pyplot as plt
    plt.close("all")
