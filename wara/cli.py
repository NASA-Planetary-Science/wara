"""
Command-line dispatcher for wara.

Routing:
  wara                 → legacy GUI   (current default)
  wara --legacy        → legacy GUI   (explicit)
  wara --beta          → new GUI      (work in progress)

The redesigned interface is being built under ``wara.gui_beta`` without
disturbing the working legacy app in ``wara.gui``. Once the beta reaches
parity it will become the default (``wara``) and the old one will remain
reachable as ``wara --legacy`` for a transition period.
"""
import sys


def main():
    argv = sys.argv[1:]

    if "--beta" in argv:
        sys.argv.remove("--beta")
        from wara.gui_beta import main as beta_main
        beta_main()
    elif "--legacy" in argv:
        sys.argv.remove("--legacy")
        from wara.gui import main as legacy_main
        legacy_main()
    else:
        # Default → legacy for now. This will switch to the beta GUI later.
        from wara.gui import main as legacy_main
        legacy_main()


if __name__ == "__main__":
    main()
