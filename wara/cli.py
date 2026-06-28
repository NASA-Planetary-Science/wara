"""
Command-line dispatcher for wara.

Routing:
  wara                 → new GUI      (default, v2.0+)
  wara --legacy        → legacy GUI   (the pre-2.0 interface)
  wara --beta          → new GUI      (explicit; same as the default)

The redesigned interface (``wara.gui_beta``) became the default in v2.0. The
previous app remains reachable as ``wara --legacy`` for a transition period.
``--beta`` is kept as an explicit alias so existing muscle memory keeps working.
"""
import sys


def main():
    argv = sys.argv[1:]

    if "--legacy" in argv:
        sys.argv.remove("--legacy")
        from wara.gui import main as legacy_main
        legacy_main()
    else:
        # Default → new GUI. ``--beta`` is accepted as an explicit alias and
        # stripped so the GUI's own argument parser doesn't see it.
        if "--beta" in argv:
            sys.argv.remove("--beta")
        from wara.gui_beta import main as beta_main
        beta_main()


if __name__ == "__main__":
    main()
