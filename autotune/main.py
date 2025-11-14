"""Entry point for launching the PyQt autotune GUI."""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from gui import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
