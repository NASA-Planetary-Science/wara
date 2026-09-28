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


#: Tests between the manual garbage collections of :func:`_gc_between_tests`.
GC_EVERY = 10
_tests_since_gc = 0


@pytest.fixture(autouse=True)
def _gc_between_tests():
    """Run Python's cyclic GC only between tests, never inside one.

    A GUI test's ``WaraApp`` and its controllers reference each other
    (``WaraApp.api`` <-> ``controller.app``), so a finished test's window is
    freed by the cyclic GC, whenever some allocation happens to trigger it.
    Some of those allocations come from PyQt wrapping a Qt event in the middle
    of another window's ``show()``; the GC then destroyed hundreds of stale C++
    widgets inside Qt's own call. That is undefined behaviour, and it
    intermittently aborted CI (exit 134 on macOS, silently on Windows) in
    ``test_apply_to_data_writes_psd_column_and_reads_it_back``.

    So automatic collection is off while a test runs, and old windows are
    freed here, at a safe point, every ``GC_EVERY`` tests (a collection after
    every test would add ~30 s to the suite). Explicit ``gc.collect()`` calls
    inside tests still work. (Deleting the windows directly -- ``sip.delete``
    or ``deleteLater`` -- is not an option: it leaves their Python side
    rooted, so the heap and the GC's pauses grow for the rest of the session.)
    """
    import gc
    import sys
    global _tests_since_gc
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        # Tests never run an event loop, so Qt's posted events (layout and
        # repaint requests, matplotlib's draw_idle, ...) are never processed
        # and the queue only grows -- and every widget the GC destroys scans
        # it, which made the collections below take ~1 s late in the suite.
        # Drop them.
        if "PyQt5.QtCore" in sys.modules:
            from PyQt5.QtCore import QCoreApplication
            if QCoreApplication.instance() is not None:
                QCoreApplication.removePostedEvents(None)
        _tests_since_gc += 1
        if _tests_since_gc >= GC_EVERY:
            _tests_since_gc = 0
            gc.collect()
        if was_enabled:
            gc.enable()


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
