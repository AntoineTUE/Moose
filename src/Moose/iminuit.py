"""A module that simplifies using [iminuit](https://scikit-hep.org/iminuit/) to fit models.

It is mainly aimed at simplifying working with the [iminuit.Minuit][] function minimizer and the [iminuit.cost.LeastSquares] cost function.

See also https://scikit-hep.org/iminuit/about.html
"""

from collections.abc import Callable
from functools import wraps
import inspect


def as_minuit_objective(model_function: Callable, **fixed_kwargs):
    """Prepare a model function for minimization by [iminuit.Minuit][], by wrapping the model and prepopulating kwargs.

    This is specifically aimed at solving `iminuit` not accepting non-parameter arguments for the cost function or minimizer.

    Due to argument inspection [functools.partial[] fails, as the [iminuit.Minuit][] minimizer expects more arguments to be passed.

    To further complicate matters, (keyword) arguments are expected to be floats by `iminuit`, which will cause exceptions when checking array-like objects or booleans.

    With this decorator, you can quickly create a suitable model from e.g. [Moose.Simulation.model_for_fit][], or provide your own, custom model function.

    Example:
    ```py
    db = Moose.query_DB("N2CB")
    objective = minuit_objective_decorator(normalize=True,sim_db=db)(Moose.model_for_fit)

    # or use as a decorator
    @as_minuit_objective(sim_db=db,normalize=False)
    def my_model_function(...):
        ...
    ```
    """

    def decorator(func):
        """Modify signature and compile defaults as well as decorator arguments.

        Raises a KeyError if an argument is missing that will be required.

        Does not perform validation of passed arguments, so passing `None` will satisfy this check.
        """
        sig = inspect.signature(func)

        # Parameters seen by iminuit
        new_params = []

        injected_defaults = {}
        for name, param in sig.parameters.items():
            if name in fixed_kwargs:
                continue

            # VAR_KEYWORD type argument are not allowed/supported by iminuit.
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            # Only retain kwargs (i.e. with a default) that have float or int type
            # Other types must not be passed to iminuit.
            # This is why this manual handling is required, since we cannot use partial binding.
            # TODO: test if more types need to be checked, for now if not passed in decorator, it is assumed missing.
            if param.default is not inspect.Parameter.empty:
                if isinstance(param.default, bool):
                    injected_defaults[name] = param.default
                    continue
                if isinstance(param.default, (int, float)):
                    new_params.append(param)
                    continue
                # non-numeric default: missing unless supplied to decorator.
                raise KeyError(
                    f"Missing argument `{name}` for function `{func.__name__}`, "
                    f"expected non-float kwarg with default value."
                )

            # If no default value, assume it must be passed to (and optimized by) iminuit.
            new_params.append(param)

        # produce modified signature
        new_sig = sig.replace(parameters=new_params)

        @wraps(func)
        def wrapper(*args, **call_kwargs):
            """Run model function at call time.

            Injects default and arguments passed to the decorator.

            Injecting defaults first allows them to be overwritten at model decoration or calltime.
            """
            final_kwargs = {}
            final_kwargs.update(injected_defaults)
            final_kwargs.update(fixed_kwargs)
            final_kwargs.update(call_kwargs)

            return func(*args, **final_kwargs)

        wrapper.__signature__ = new_sig

        return wrapper

    if model_function is not None:
        return decorator(model_function)
    return decorator
