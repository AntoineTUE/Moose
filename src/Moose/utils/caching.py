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
import threading
import xxhash
import numpy as np
from collections import OrderedDict, defaultdict

from numpy.typing import NDArray


def hash_numpy(arr: NDArray) -> str:
    """Hash a numpy array, based on its shape, dtypes and contents, using xxhash.

    Important: to avoid hash-collisions as much as possible, the 128-bit hashing algorithm is used.

    See also: https://github.com/Cyan4973/xxHash/wiki/Collision-ratio-comparison
    """
    h = xxhash.xxh128()
    h.update(str(arr.shape))
    h.update(arr.dtype.str)
    h.update(memoryview(arr))  # zero-copy
    return h.hexdigest()


class ArrayLRUCache:
    """A Least Recently Used cache implementation with support for numpy arrays.

    Since numpy arrays are mutable, they are strictly speaking not hashable, as required by built-in [functools.lru_cache][].

    To still benefit from caching, this cache returns only copies of the original array in cache, keeping the original (sort-of) immutable.

    (Note: the array can still be modified, but any modification done on a returned array are isolated from the original cached version.)

    At the cost of some compute overhead it deduplicates arrays (based on hash comparison), to avoid storing many identical copies.

    Without deduplication the following calls would store the same array twice: `my_func(array,1,2)+my_func(array,3,4)`.

    For large arrays (as we may deal with), this can cause significant memory use.

    Note that this cache only supports `numpy.ndarray` on top of standard hashable data types, NOT any 'ArrayLike' object such as dataframes.

    Computing hashes on a Dataframe (i.e. [pandas.util.hash_pandas_object][]) is too time consuming to be beneficial in the use-case of Moose it seems.
    """

    def __init__(self, func, maxsize=128):
        """Initialize a cache with support for numpy NDArray types, with object deduplication."""
        self.func = func
        self.maxsize = maxsize

        self.cache = OrderedDict()  # map key to result
        self.object_store = {}  # map hash to object
        self.ref_counts = defaultdict(int)  # track ref count of hashes

        self.hits = 0
        self.misses = 0

        self.lock = threading.RLock()

    def __call__(self, *args, **kwargs):
        """Compute keys, check/update cache and store objects.

        On cache miss, calls the original function and caches the result.
        """
        # Build key and track hashes
        with self.lock:
            key, obj_hashes = self._make_key(args, kwargs)

            # Cache hit
            if key in self.cache:
                self.hits += 1
                result = self.cache.pop(key)
                self.cache[key] = result  # LRU bump
                return self._safe_copy(result)

            self.misses += 1

        # Compute result outside lock to avoid blocking
        result = self.func(*args, **kwargs)

        # Store result and update ref counts
        with self.lock:
            self.cache[key] = result
            self._inc_refs(obj_hashes)

            # Evict LRU if necessary
            if len(self.cache) > self.maxsize:
                self._evict_lru()

            return self._safe_copy(result)

    def cache_info(self):
        """Return information about the cache and object storage.

        Threadsafe.
        """
        with self.lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "unique_objects": len(self.object_store),
            }

    def cache_clear(self):
        """Clear the cache and associated object storage.

        Threadsafe.
        """
        with self.lock:
            self.cache.clear()
            self.object_store.clear()
            self.ref_counts.clear()
            self.hits = self.misses = 0

    def _make_key(self, args, kwargs):
        """Construct a hashing key for a function call, based on the args/kwargs.

        If any arg/kwarg is a numpy array, add their hash to a separate list as well, for reference-counting.
        """
        obj_hashes = []

        def normalize(x):
            if isinstance(x, np.ndarray):
                # Make sure array is C-contiguous to enable hashing
                # Note: F-contiguous will be copied, C-contiguous just returns the object.
                x = np.ascontiguousarray(x)
                h = hash_numpy(x)
                self._store_object(h, x)
                obj_hashes.append(h)
                return ("__np__", h)
            if isinstance(x, (list, tuple)):
                return tuple(normalize(i) for i in x)

            if isinstance(x, dict):
                return tuple(sorted((k, normalize(v)) for k, v in x.items()))

            return x

        key = (tuple(normalize(a) for a in args), tuple(sorted((k, normalize(v)) for k, v in kwargs.items())))

        return key, obj_hashes

    def _store_object(self, h, obj):
        """Store an objects by its hash key in the object_store."""
        if h not in self.object_store:
            self.object_store[h] = obj

    def _inc_refs(self, hashes):
        """Increment references to tracked hashes for objects in the objects store."""
        for h in hashes:
            self.ref_counts[h] += 1

    def _dec_refs(self, hashes):
        """Decrement references to tracked hashes of objects in the objects_store.

        Remove objects that have no references (i.e. 0 or less counts.)
        """
        for h in hashes:
            self.ref_counts[h] -= 1
            if self.ref_counts[h] <= 0:
                self.ref_counts.pop(h, None)
                self.object_store.pop(h, None)

    def _evict_lru(self):
        """Remove a key from the LRU cache and decrement refs to stored objects that are part of the key."""
        old_key, _ = self.cache.popitem(last=False)
        hashes = self._extract_hashes_from_key(old_key)
        self._dec_refs(hashes)

    def _extract_hashes_from_key(self, key):
        """Find hashes of hashed numpy arrays by resursively traversing a cache key."""
        hashes = []

        def walk(x):
            if isinstance(x, tuple):
                if len(x) == 2 and x[0] in ("__np__",):
                    hashes.append(x[1])
                else:
                    for i in x:
                        walk(i)

        walk(key)
        return hashes

    @staticmethod
    def _safe_copy(result):
        """Return a copy if the data is a numpy.ndarray.

        This means that the returned array can be modified without affecting the cached version (or its hash).
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
            return cache(*args, **kwargs)

        wrapper.cache_info = cache.cache_info
        wrapper.cache_clear = cache.cache_clear

        return wrapper

    return decorator
