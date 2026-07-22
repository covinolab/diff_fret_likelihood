#!/usr/bin/env python
"""Convenience: build the Cython simulator in place. Delegates to setup.py.

    python build_cython.py               # portable -O3 build
    DFL_NATIVE=1 python build_cython.py  # CPU-specific fast-math build

Equivalent to ``python setup.py build_ext --inplace``. The real build logic
(GSL discovery, compile flags, the ``diff_fret_likelihood.simulator`` extension)
lives in ``setup.py`` so there is a single source of truth.
"""

import subprocess
import sys

raise SystemExit(
    subprocess.call([sys.executable, "setup.py", "build_ext", "--inplace"])
)
