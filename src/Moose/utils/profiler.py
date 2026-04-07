"""Utilities for profiling the execution time of functions.

This enables monitoring performance in a more realistic scenario than static microbenchmarks.
"""

from functools import wraps
from time import perf_counter
from dataclasses import dataclass, field
from threading import Lock
import numpy as np


@dataclass
class ProfilerInfo:
    """A dataclass for tracking function execution time over the lifetime of a program."""

    calls: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    standard_deviation: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def update(self, duration: float):
        """Update the tracked information with a new result.

        Runtime standard deviation calculated with Welford's algorithm.
        """
        self.calls += 1
        self.total_time += duration
        delta = duration - self.average_time
        self.average_time += delta / self.calls
        delta2 = duration - self.average_time
        self.standard_deviation = 0 if self.calls <= 1 else np.sqrt(delta * delta2 / (self.calls - 1))

    def reset(self):
        """Reset tracked information to 0."""
        self.calls = 0
        self.total_time = 0.0
        self.average_time = 0.0
        self.standard_deviation = 0.0


def profile(func):
    """Profile function execution time, simply tracking the average execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        duration = perf_counter() - start

        wrapper.prof.update(duration)

        return result

    wrapper.prof = ProfilerInfo()
    return wrapper
