"""
Moose allows you to simulate thermal diatomic spectra based on line-by-line databases.

[Simulation](./Simulation) contains the core functionality to work with these databases.


When you import `Moose`, it will check if the [`lmfit`](https://lmfit.github.io/lmfit-py/) package is installed in your environment.

If so, it will import some additional functions in the `Moose` namespace for your convenience.

[FitUtil](./FitUtil) contains these additional functions.

To see `Moose` in action, check out the [examples](../../examples)
"""

import warnings

from ._version import version, version_tuple
from .Simulation import (
    apply_voigt,
    create_stick_spectrum,
    default_params,
    equidistant_mesh,
    match_spectra,
    model_for_fit,
    query_DB,
    thermal_default_params,
)
from .utils.db_io import get_database_files, get_database_path, set_database_path


def __getattr__(name):
    """Temporary compatibility patch to support the deprecated `database_files` attribute.

    Users are strongly encouraged to use `get_database_files()` instead of relying on this deprecated attribute.

    In a future release this attribute will be removed entirely.
    """
    if name == "database_files":
        warnings.warn(
            "`database_files` is deprecated; call `get_database_files()` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_database_files()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


get_database_path()  # triggers migration message.

__all__ = [
    "version",
    "version_tuple",
    "default_params",
    "thermal_default_params",
    "query_DB",
    "create_stick_spectrum",
    "equidistant_mesh",
    "apply_voigt",
    "match_spectra",
    "model_for_fit",
    "get_database_path",
    "set_database_path",
    "get_database_files",
]
