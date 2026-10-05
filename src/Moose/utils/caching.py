"""A module providing tools for implementing caching functionality for functions that are used by Moose.

It mainly addresses the need for a cache that is able to handle numpy.ndarray arguments, which are technically not hashable (and thus not cachable) due to their mutable nature.

However, when using a function such as [model_for_fit][^^.], several internal intermediate result arrays are created that are not modified or exposed to the caller.

These should be safe for caching and it could provide significant benefits during minimization, with speed-ups up to one or two orders of magnitude per function call, if the cache is hit.

Though a least-squares minimization will try many different values, with some small adjustments to the parameters in some cases, profiling and cache statistics during testing indicate that there is still some benefit to be had.

With sufficient depth of the cache, there are also benefits when the minimizer revisits previous trials as it walks the probability space to converge to a solution.

Important:
    To reliably cache numpy arrays quickly, they need to be hashed.

    For hashing the arrays must be in C-contiguous order (also known as row-major), instead of F-contiguous (also known as Fortran-contiguous, or column-major).

    While hasing/caching still works for F-contiguous arrays, they are converted to C-contiguous, before hashing, requiring a copy of memory.

    For C-contiguous arrays, the hashing operation is zero-copy instead.
"""

import functools
import inspect
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np
import xxhash
from numpy.typing import NDArray

from .profiler import profile


def hash_numpy(arr: NDArray) -> str:
    """Hash a numpy array, based on its shape, dtypes and contents, using xxhash.

    Important: to avoid hash-collisions as much as possible, the 128-bit hashing algorithm is used.

    See also: https://github.com/Cyan4973/xxHash/wiki/Collision-ratio-comparison
    """
    h = xxhash.xxh128()
    h.update(str(arr.shape).encode())
    h.update(arr.dtype.str.encode())
    h.update(memoryview(arr))  # zero-copy
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class ArrayKey:
    """A (hashable) key representing a numpy array, used for caching purposes.

    In principle the shape and dtype info is already contained in the (hex)digest of the has.

    However, including this information in this object helps identificiation when debugging or testing.

    It is immediately clear that the key represents an array, and preserves some useful information.

    Note:
        The use of `frozen=True` and `eq=True`  for the dataclass ensures the object is hashable (as it is immutable).
        This enables it to be used as a cache-key itself.
    """

    digest: str
    shape: tuple[int, ...]
    dtype: str


@dataclass()
class CacheInfo:
    """Information about cache use."""

    function: str
    hits: int
    misses: int
    size: int
    max_size: int
    cache_hit_rate: float = field(init=False)

    def __post_init__(self):
        """Compute cache hit rate."""
        self.cache_hit_rate = self.hits / max((self.hits + self.misses), 1) * 100


class ArrayLRUCache:
    """A Least Recently Used cache implementation with support for numpy arrays.

    Since numpy arrays are mutable, they are strictly speaking not hashable, as required by built-in [functools.lru_cache][].

    To still benefit from caching, this cache returns only copies of the original array in cache, keeping the original (sort-of) immutable.

    (Note: the array can still be modified, but any modification done on a returned array are isolated from the original cached version.)

    Note that this cache only supports `numpy.ndarray` on top of standard hashable data types, NOT any 'ArrayLike' object such as dataframes.

    Computing hashes on a Dataframe (i.e. [pandas.util.hash_pandas_object][]) is too time consuming to be beneficial in the use-case of Moose it seems.

    In addition, the numpy array must not use object dtype, as this would break the hashing and caching mechanism.
    """

    def __init__(self, func, maxsize=128):
        """Initialize a cache with support for numpy NDArray types, with object deduplication."""
        self.func = func
        self.signature = inspect.signature(func)
        self.maxsize = maxsize

        self.cache = OrderedDict()  # map key to result

        self.hits = 0
        self.misses = 0

        self.lock = threading.RLock()

    @property
    def func_name(self) -> str:
        """The name of the function being cached as a string."""
        if hasattr(self.func, "__qualname__"):
            return self.func.__qualname__
        return self.func.__name__

    def __call__(self, *args, **kwargs):
        """Compute keys, check/update cache and store objects.

        On cache miss, calls the original function and caches the result.
        """
        # Build key and check if hit
        with self.lock:
            key = self._make_key(args, kwargs)
            if key in self.cache:  # Cache hit; bump LRU and return
                self.hits += 1
                self.cache.move_to_end(key)  # LRU bump
                return self._safe_copy(self.cache[key])
            self.misses += 1

        # Compute result outside lock to avoid blocking
        result = self.func(*args, **kwargs)

        # Store result; evict if needed
        with self.lock:
            self.cache[key] = result

            # Evict LRU if necessary
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

            return self._safe_copy(result)

    def cache_info(self):
        """Return information about the cache and object storage.

        Threadsafe.

        If the cache is disabled, these metrics won't change, but will persist until the cache is cleared.
        """
        with self.lock:
            return CacheInfo(self.func_name, self.hits, self.misses, len(self.cache), self.maxsize)

    def cache_clear(self):
        """Clear the cache and associated object storage.

        Threadsafe.
        """
        with self.lock:
            self.cache.clear()
            self.hits = self.misses = 0

    @staticmethod
    def normalize(x):
        """Normalize an input for constructing a stable cache-key.

        If a numpy array of object dtype is encountered, will raise a TypeError, as this is unsupported.

        - Numpy arrays are converted to a hash-based ArrayKey.
        - Lists and tuples are recursively normalized.
        - Dictionaries are converted to sorted tuples of key-value pairs.
        - Other types are returned as-is.

        Note: for numpy arrays, make sure the array is C-contiguous for zero-copy hashing by xxhash.
        A F-contiguous (Fortran-order) array will be copied and transposed to make it C-contiguous.
        """
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise TypeError("Numpy arrays with Object dtypes are not supported for caching.")
            x = np.ascontiguousarray(x)
            h = hash_numpy(x)
            # Use ArrayKey so it is clear this used to be an array
            return ArrayKey(h, x.shape, x.dtype.str)
        if isinstance(x, (list, tuple)):
            return tuple(ArrayLRUCache.normalize(i) for i in x)

        if isinstance(x, dict):  # sort key-value pairs by key alphabetically.
            return tuple(sorted((k, ArrayLRUCache.normalize(v)) for k, v in x.items()))

        return x

    def _make_key(self, args, kwargs):
        """Construct a hashing key for a function call, based on the args/kwargs.

        If a numpy array of object dtype is encountered, will raise a TypeError, as this is unsupported.
        """
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return tuple(ArrayLRUCache.normalize(arg) for arg in bound.arguments.values())

    @staticmethod
    def _safe_copy(result):
        """Return a copy of the data so that modifications do not affect the cached version, if it is a numpy.ndarray.

        This means that the returned array can be modified without affecting the cached version (or its hash).

        Another function which consumes the result can safely modify (and is allowed to modify) the returned array.
        """
        if isinstance(result, np.ndarray):
            return result.copy()
        return result


def array_cache(maxsize=128):
    """Decorate a function to use the ArrayLRUCache.

    This differs from the built-in [functools.lru_cache][] in that numpy arrays are supported as arguments as well.

    Note that this means it only supports numpy objects, not 'array-like' objects like dataframes.

    For the best performance, any numpy array must be C-contiguous, in which case hasing is zero-copy.

    For F-contiguous arrays, a copy will be made internally.

    See also: [numpy.ascontiguousarray][].
    """

    def decorator(func):
        cache = ArrayLRUCache(func, maxsize=maxsize)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return wrapper._target(*args, **kwargs)

        wrapper._target = cache

        def enable_cache(enabled: bool = True, reset: bool = False):
            """Toggle the use of the cache on or off for the decorated function.

            The optional `reset` flag controls if the cache should be cleared as well.
            """
            wrapper._target = cache if enabled else func
            if reset:
                cache.cache_clear()

        wrapper.cache_info = cache.cache_info
        wrapper.cache_clear = cache.cache_clear
        wrapper.enable_cache = enable_cache

        return wrapper

    return decorator
