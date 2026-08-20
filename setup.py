import os
import subprocess
import warnings

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize


# --- GSL discovery -----------------------------------------------------------
def _gsl_dirs():
    """Return ``(include_dirs, library_dirs, found)``.

    Resolution order: ``gsl-config`` -> ``pkg-config`` -> ``$EBROOTGSL`` ->
    ``$GSL_DIR`` -> compiler defaults. ``found`` is True when a discovery tool or
    env var reported GSL (even if the returned dir lists are empty because GSL
    lives on the default search path).
    """
    for probe in (["gsl-config", "--cflags"], ["pkg-config", "--cflags-only-I", "gsl"]):
        try:
            cflags = subprocess.check_output(
                probe, stderr=subprocess.DEVNULL, text=True
            ).split()
            libs_cmd = (
                ["gsl-config", "--libs"]
                if probe[0] == "gsl-config"
                else ["pkg-config", "--libs-only-L", "gsl"]
            )
            libs = subprocess.check_output(
                libs_cmd, stderr=subprocess.DEVNULL, text=True
            ).split()
            inc = [f[2:] for f in cflags if f.startswith("-I")]
            lib = [f[2:] for f in libs if f.startswith("-L")]
            return inc, lib, True  # discovery tool present => GSL considered found
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    for env_var in ("EBROOTGSL", "GSL_DIR"):
        root = os.environ.get(env_var, "").strip()
        if root:
            return [os.path.join(root, "include")], [os.path.join(root, "lib")], True

    return [], [], False


_GSL_HINT = (
    "\n*** GSL not found ***\n"
    "diff_fret_likelihood's Cython simulator links libgsl/libgslcblas/libm.\n"
    "Install the GSL development package first:\n"
    "  Debian/Ubuntu : sudo apt-get install libgsl-dev pkg-config\n"
    "  Fedora/RHEL   : sudo dnf install gsl-devel pkgconf-pkg-config\n"
    "  conda-forge   : conda install -c conda-forge gsl pkg-config\n"
    "  macOS (brew)  : brew install gsl pkg-config\n"
    "Or set GSL_DIR=/path/to/gsl (expects $GSL_DIR/include and $GSL_DIR/lib).\n"
)

gsl_inc, gsl_lib, gsl_found = _gsl_dirs()
if not gsl_found and not os.environ.get("DFL_ALLOW_MISSING_GSL"):
    # Could not locate GSL via gsl-config / pkg-config / GSL_DIR / EBROOTGSL. The
    # build may still succeed if GSL sits on the default search path; set
    # DFL_ALLOW_MISSING_GSL=1 to silence this and try anyway.
    warnings.warn(_GSL_HINT, stacklevel=2)


# --- compile flags: portable by default, aggressive/native opt-in ------------
# Default build is reproducible + portable (-O3 only). Set DFL_NATIVE=1 for the
# original CPU-specific, fast-math build used for in-house benchmarking.
# -ffast-math is deliberately opt-in: it relaxes IEEE semantics and changes the
# simulated photon streams bit-for-bit across machines.
_NATIVE = [
    "-ffast-math",
    "-march=native",
    "-mtune=native",
    "-funroll-loops",
    "-fno-plt",
    "-fomit-frame-pointer",
    "-flto",
]
if os.environ.get("DFL_NATIVE") == "1":
    extra_compile_args = ["-O3", *_NATIVE]
    extra_link_args = ["-flto"]
else:
    extra_compile_args = ["-O3"]
    extra_link_args = []


extensions = [
    Extension(
        name="diff_fret_likelihood.simulator",  # in-package, dotted name
        sources=["simulator.pyx"],
        libraries=["gsl", "gslcblas", "m"],
        include_dirs=[np.get_include(), *gsl_inc],
        library_dirs=gsl_lib,
        runtime_library_dirs=gsl_lib,  # rpath: import without LD_LIBRARY_PATH
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="diff_fret_likelihood.doob_tpt",  # Doob h-transform TPT sampler
        sources=["doob_tpt.pyx"],
        libraries=["gsl", "gslcblas", "m"],
        include_dirs=[np.get_include(), *gsl_inc],
        library_dirs=gsl_lib,
        runtime_library_dirs=gsl_lib,  # rpath: import without LD_LIBRARY_PATH
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

ext_modules = cythonize(
    extensions,
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "initializedcheck": False,
        "nonecheck": False,
        "always_allow_keywords": False,
        "embedsignature": False,
        "optimize.use_switch": True,
        "optimize.unpack_method_calls": True,
    },
)


class build_py(_build_py):
    """Keep packaging-only modules out of the installed package.

    The flat layout maps ``package_dir={"diff_fret_likelihood": "."}``, so
    setuptools would otherwise copy every top-level ``*.py`` (including this
    ``setup.py`` and ``build_cython.py``) into the wheel as importable submodules.
    """

    _EXCLUDE = {"setup", "build_cython"}

    def find_package_modules(self, package, package_dir):
        mods = super().find_package_modules(package, package_dir)
        return [m for m in mods if m[1] not in self._EXCLUDE]


setup(ext_modules=ext_modules, cmdclass={"build_py": build_py})
