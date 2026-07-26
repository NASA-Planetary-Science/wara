"""
Command-line dispatcher for wara.

Routing:
  wara                 → wara GUI (``wara.gui``)

The redesigned interface became the default in v2.0 and is now the only GUI:
the pre-2.0 app was retired in v2.1. ``--legacy`` and ``--beta`` are accepted
and ignored (with a notice) so old invocations do not simply crash.
"""
import sys

_RETIRED = """\
wara: --legacy is retired.

The pre-2.0 GUI was removed in v2.1; `wara` now launches the only interface.
To use the old app, install an earlier release:  pip install "wara<2.1"

Launching the current GUI...\
"""


def main():
    argv = sys.argv[1:]

    if "--legacy" in argv:
        sys.argv.remove("--legacy")
        print(_RETIRED, file=sys.stderr)
    # "--beta" was the pre-2.0 flag for what is now the default GUI;
    # strip it so old invocations keep working.
    if "--beta" in argv:
        sys.argv.remove("--beta")

    from wara.gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
