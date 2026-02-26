"""Utility functions for the smooth package."""

import importlib.metadata
import platform
import sys


def _get_version(package_name):
    """Get installed version of a package, or 'not installed'."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def show_versions():
    """Print system info and installed dependency versions for debugging."""
    print("\nSystem:")
    print(f"  python: {sys.version}")
    print(f"  executable: {sys.executable}")
    print(f"  machine: {platform.platform()}")

    print("\nSmooth:")
    print(f"  smooth: {_get_version('smooth')}")

    print("\nDependencies:")
    for pkg in ["numpy", "pandas", "scipy", "statsmodels", "nlopt", "pybind11"]:
        print(f"  {pkg}: {_get_version(pkg)}")
