"""
Moose allows you to simulate thermal diatomic spectra based on line-by-line databases.

[Simulation](./Simulation) contains the core functionality to work with these databases.


When you import `Moose`, it will check if the [`lmfit`](https://lmfit.github.io/lmfit-py/) package is installed in your environment.

If so, it will import some additional functions in the `Moose` namespace for your convenience.

[FitUtil](./FitUtil) contains these additional functions.

To see `Moose` in action, check out the [examples](../../examples)
"""

from .Simulation import *
from .Simulation import _default_params
import importlib.util
if importlib.util.find_spec('lmfit') is not None:
    from .FitUtil import *