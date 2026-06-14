"""
Moose allows you to simulate thermal diatomic spectra based on line-by-line databases.

[Simulation](./Simulation) contains the core functionality to work with these databases.


When you import `Moose`, it will check if the [`lmfit`](https://lmfit.github.io/lmfit-py/) package is installed in your environment.

If so, it will import some additional functions in the `Moose` namespace for your convenience.

[FitUtil](./FitUtil) contains these additional functions.

To see `Moose` in action, check out the [examples](../../examples)
"""

import importlib.util
from .Simulation import (
    default_params,
    thermal_default_params,
    query_DB,
    create_stick_spectrum,
    equidistant_mesh,
    apply_voigt,
    match_spectra,
    model_for_fit,
)

from .utils.db_io import get_database_path, set_database_path, database_files

get_database_path()  # triggers migration message.

__all__ = [
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
    "database_files",
]
