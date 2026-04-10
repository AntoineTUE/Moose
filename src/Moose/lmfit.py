"""A module that simplifies using [lmfit](https://lmfit.github.io/lmfit-py/) to fit models.

It is mainly aimed at simplifying working with the more abstract [lmfit.minimizer.Minimizer][] class (or the convenience [lmfit.minimizer.minimize][] function) directly, which has unique requirements for the function signature.

See also https://lmfit.github.io/lmfit-py/fitting.html#using-the-minimizer-class
"""

import importlib.util

if importlib.util.find_spec("lmfit") is None:
    raise ImportError("The 'lmfit' package is not installed in your enviroment, please install it first")
from functools import wraps
import inspect
from collections.abc import Callable
import numpy as np
import lmfit
from .Simulation import create_stick_spectrum, apply_voigt, equidistant_mesh, match_spectra

from numpy.typing import NDArray


def multi_species_objective(
    params: lmfit.Parameters | dict,
    x: NDArray,
    y: NDArray | None = None,
    error: NDArray | None = None,
    normalize: bool = False,
    **dbs,
):
    """Simulate spectrum for one or more species, by providing suitable line-by-line data and model parameters.

    The function signature is made for use with the [lmfit.minimizer.Minimizer][] class and takes a [lmfit.parameter.Parameters][] object as an argument.

    It is made to flexibly model spectra composed of multiple species, simply by modelling through specifying parameters (`params`) and providing suitable line-by-line data (`dbs`).

    For each species you want to fit, you must provide the line-by-line data as a keyword argument, with e.g. the species name as key.

    For each species (with name: `species_name`) you can provide the following parameters: `T_rot_{species_name}`, `T_vib_{species_name}` and `fraction_{species_name}`.

    When not provided, these default respectively to the parameters: `T_rot`, `T_vib` and `A`, which themselves have default values (if not provided) resp: 500, 500, 1.

    Note:
        If only `x` is provided (not `y` and `error`) this function returns the model evaluated at `x`.

        If `x` and `y` are provided, it returns the residual according to `residual=(model-y)`.

        If on top of that `error` is provided, these are used as weights to scale the residual: `residual = (model-y)/error`.

        You thus can use this function to simulate a spectrum, but also to calculate the residual for minimization.

    Example:
    ```python
    import lmfit
    import Moose
    from Moose.lmfit import multi_species_objective

    db_n2 = Moose.query_DB("N2CB")
    db_cn = Moose.query_DB("CNBX")

    x = ...
    y = ...

    # create and add suitable model parameters for fitting two species not in equilibrium
    # T_vib and T_rot will be used for N2, for CN we add new params for these
    params = lmfit.create_params(**Moose.default_params)
    params.add("fraction_N2", 0.5, True, 0, 1)
    params.add("fraction_CN", 0.5, True, 0, 1)
    params.add("T_rot_CN", 800, True, 300, 10000)
    params.add("T_vib_CN", 8000, True, 300, 10000)

    result = lmfit.mimimize(
        multi_species_objective,
        params,
        kws={
            "x": x,
            "y": y,
            "normalize": True
            "N2": db_n2,
            "CN": db_cn[db_cn.Dv==0],
        }
    )
    ```

    Args:
        params:     The model parameters, either a [lmfit.parameter.Parameters][] or a dict mapping parameter names to values.
        x:          The wavelength values
        y:          The data, i.e the spectrum to be fitted.
        error:      The errors in y, to be used as weights for determining the residual during minimization.
        normalize:  Flag to normalize (and scale) the simulated data, or leave it unscaled.

    """
    paramvalues = params.valuesdict() if isinstance(params, lmfit.Parameters) else params
    T_rot = paramvalues.get("T_rot", 500)
    T_vib = paramvalues.get("T_vib", 500)
    pop = paramvalues.get("A", 1)

    sticks = []
    for name, db in dbs.items():
        Tv = paramvalues.get(f"T_vib_{name}", T_vib)
        Tr = paramvalues.get(f"T_rot_{name}", T_rot)
        fraction = paramvalues.get(f"fraction_{name}", 1 if normalize else pop)
        stick = create_stick_spectrum(
            Tv,
            Tr,
            pop=fraction,
            df_db=db,
        )
        sticks.append(stick)
    equid = equidistant_mesh(np.concatenate(sticks), paramvalues["wl_pad"], paramvalues["resolution"])
    simulation = apply_voigt(equid, paramvalues["sigma"], paramvalues["gamma"])
    sim_matched = match_spectra((x - paramvalues["mu"]).reshape(-1, 1), simulation)
    if normalize is True:
        vals = sim_matched[:, 1]
        sim_matched[:, 1] = paramvalues["A"] * (sim_matched[:, 1] - vals.min()) / (vals.max() - vals.min())

    y_sim = sim_matched[:, 1] + paramvalues["b"]
    if y is None:
        return y_sim
    if error is None:
        return y_sim - y
    return (y_sim - y) / error


def as_minimizer_objective(model_function: Callable, **fixed_kwargs):
    """Decorate or wrap a function to adapt to the signature expected by [lmfit.minimizer.Minimizer][].

    By means of inspection it alters the functions signature and injects `fixed_kwargs` as fixed model parameters that will not be varied.

    Make sure that `fixed_kwargs` includes all function args/kwargs that will not be passed as part of an [lmfit.parameter.Parameters][] object.

    The function MUST have the independent variable as its first argument: `def func(x,*args)`, with `x` the independent variable.

    The signature of the wrapped function becomes: `def wrapped_func(params:lmfit.Parameters,x,y=None,eps=None, **fixed_kwargs)`

    `fixed_kwargs` can thus still be overwritten at calltime, even after wrapping.

    You can use this function to wrap e.g. [Moose.Simulation.model_for_fit][], but also your own, custom model function.

    Important:
        When `y` is not None, the function will calculate the residual (`residual = func(x,*params)-y`).

        If the error `eps` is provided as well, this residual will be scaled by dividing by `eps`: `residual = (func(x,*params)-y)/eps`.

    Example use:
        ```python
        import lmfit
        import Moose

        db = Moose.query_DB("OHAX")
        params = lmfit.create_params(**Moose.default_params)
        params.pop("A") # remove A as a parameter, to set it as a fixed argument of the model

        x,y,yerror = ...

        # Flexible model with drop-in replacement of database when fitting
        model = Moose.lmfit.as_minimizer_objective(Moose.model_for_fit, normalize=True, A=1)
        opt = lmfit.Minimizer(model, params, fcn_args=(x,), fcn_kws={"y": y, "eps": yerror, "normalize": True, "sim_db":  db})
        result = opt.minimize()

        # Skip the construction of the Minimizer and minimize directly:
        result = lmfit.minimize(model, params, args=(x,), kws={"y": y, "eps": yerror, "sim_db": db})

        # Create a model that reuses the same database for fitting many similar spectra
        model_fixed = Moose.lmfit.as_minimizer_objective(Moose.model_for_fit, normalize=True, A=1, sim_db=db)
        sim = model_fixed(params, x = np.linspace(300, 330, 1000)) # Simulate spectrum
        result_fixed = lmfit.minimize(model_fixed, params, args=(x,), kws={"y": y, "eps":yerror})
        ```

    Args:
        model_function (Callable):   The model function callable that needs to be evaluated, or minimized. It must take the dependent variable as first argument, e.g.: `def func(x,*args)`
        **fixed_kwargs :        Arguments of `_func` that are NOT fit parameters and not subject to optimization (when not fixed).
    """

    def decorator(func: Callable):
        original_signature = inspect.signature(func)
        parameters = original_signature.parameters

        param_names = list(parameters.keys())
        independent_var = param_names[0]

        # the kwargs y and eps can be provided at runtime to calculate the (scaled) residual.
        # If they are not provided, model simply evaluate as function of x: `_func(x,*args,**kwargs)`.
        special_runtime = {"y", "eps"}

        fit_param_names = [
            name
            for name, p in parameters.items()
            if (
                name != independent_var
                and name not in fixed_kwargs
                and name not in special_runtime
                and p.kind != inspect.Parameter.VAR_KEYWORD
            )
        ]

        @wraps(func)
        def wrapper(params: lmfit.Parameters, *args, **kwargs) -> Callable:
            y = kwargs.pop("y", None)
            eps = kwargs.pop("eps", None)
            x = args[0] if args else kwargs.pop(independent_var)

            bound_inputs = {}
            bound_inputs[independent_var] = x
            for name in fit_param_names:
                if name in params:
                    bound_inputs[name] = params[name].value
            bound_inputs.update(fixed_kwargs)

            # Apply runtime kwargs to allow overriding
            bound_inputs.update(kwargs)

            # Support default value arguments
            bound = original_signature.bind_partial(**bound_inputs)
            bound.apply_defaults()

            model = func(*bound.args, **bound.kwargs)

            # Residual handling
            if y is None:
                return model

            residual = model - y

            if eps is not None:
                residual = residual / eps

            return residual

        return wrapper

    if model_function is None:
        return decorator
    else:
        return decorator(model_function)
