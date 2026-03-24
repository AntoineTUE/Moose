"""Module containing functions etc. to help in development and maintenance of Moose."""

from inspect import signature
from functools import wraps
import warnings
from collections.abc import Callable

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

    def decorator(func: Callable):
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
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

        return wrapper

    return decorator
