"""Enable `python -m dip_hunter`."""
import sys

from dip_hunter.run import main

raise SystemExit(main(sys.argv[1:]))
