"""
Command-line dispatcher for wara.

Routing:
  wara                 → wara GUI     (``wara.gui``, default since v2.0)
  wara --legacy        → legacy GUI   (the pre-2.0 interface)

The redesigned interface became the default in v2.0. The previous app remains
reachable as ``wara --legacy`` for a transition period.
"""
import sys


def main():
    argv = sys.argv[1:]

    if "--legacy" in argv:
        sys.argv.remove("--legacy")
        from wara.gui_legacy import main as legacy_main
        legacy_main()
    else:
        # "--beta" was the pre-2.0 flag for what is now the default GUI;
        # strip it so old invocations keep working.
        if "--beta" in argv:
            sys.argv.remove("--beta")
        from wara.gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
