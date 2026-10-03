"""Module containing functions etc. to help in development and maintenance of Moose."""

import sys
from inspect import signature
from functools import wraps
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar
    from typing_extensions import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")

_IMPLEMENTS_DEPRECATED = hasattr(warnings, "deprecated")


def deprecated(arg=None):
    """Decorate a function to be marked as deprecated.

    Can be provided with an additional message to show, in addition to the default deprecation string.
    """

    def make_decorator(func, message=None):
        base_msg = f"The function `{func.__name__}` is deprecated and will be removed in the future."
        if message:
            base_msg += f" {message}"

        # Python 3.13+
        if _IMPLEMENTS_DEPRECATED:
            return warnings.deprecated(base_msg)(func)

        # Python <= 3.12
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(base_msg, DeprecationWarning, stacklevel=3)
            return func(*args, **kwargs)

        return wrapper

    # When used as plain decorator, i.e.: `@deprecated`
    if callable(arg):
        return make_decorator(arg)

    # When used as: `@deprecated("message")`
    def decorator(func):
        return make_decorator(func, arg)

    return decorator


def deprecated_keywords(*kw_names: str, removed_in: str = "a future release"):
    """Issue a DeprecationWarning if a function is called with a deprecated keyword argument.

    Any non-None value is considered a violation.

    Args:
        *kw_names: names of keyword args to watch.
        removed_in: short text saying when it will be removed (included in the message).
    """

    def decorator(func: "Callable[P,R]") -> "Callable[P,R]":
        sig = signature(func)

        @wraps(func)
        def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> "R":
            bound = sig.bind_partial(*args, **kwargs)
            for kw in kw_names:
                if kw in bound.arguments and bound.arguments[kw] is not None:
                    msg = (
                        f"'{kw}' is deprecated and will be removed in {removed_in}; "
                        f"do not pass '{kw}' (it will be ignored)."
                    )
                    warnings.warn(
                        msg,
                        category=DeprecationWarning,
                        stacklevel=2,
                    )
            return func(*args, **kwargs)

        wrapper.__signature__ = sig
        return wrapper

    return decorator


def warn_if_not_imported(module_name):
    """Decorate a function to warn if `module_name` has not been imported.

    This is to warn users that lazy-importing modules have not loaded yet for functions likely used in optimization.

    For example: calling `Moose.apply_voigt` when `scipy.signal` has not been imported will trigger an import.
    This import that can take over a second (or even 5 seconds!) in some scenarios.
    Especially in a fit routine, but probably in most cases, you'd want to avoid that behaviour.
    While some mitigation is in place (importing the modules in `query_DB`), we should inform and warn if it happens.

    Note:
        The decorator will disable when the module has been imported.
    """

    def decorator(func):
        check_enabled = True

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal check_enabled
            if check_enabled and module_name not in sys.modules:
                warnings.warn(
                    f"Lazy loaded {module_name!r} will import now, which negatively impacts function execution time. "
                    "Consider importing it earlier, outside of a critical loop.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                check_enabled = False
            return func(*args, **kwargs)

        return wrapper

    return decorator
