import sys
sys.argv = ['wara']
try:
    import wara.gui.app as m
    import docopt
    result = docopt.docopt(m.__doc__)
    print("Success:", result)
except SystemExit as e:
    print("SystemExit:", e.code)
except Exception as e:
    print("Error:", type(e).__name__, str(e))
