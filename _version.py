"""Single source of truth for the package version.

Read at build time by ``pyproject.toml`` (``[tool.setuptools.dynamic]``) and at
runtime via ``diff_fret_likelihood.__version__``.
"""

__version__ = "0.4.0"
