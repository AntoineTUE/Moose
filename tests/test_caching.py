from unittest.mock import Mock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

from Moose.utils.caching import ArrayLRUCache, array_cache, hash_numpy

rng = np.random.default_rng()
INT64 = np.iinfo(np.int64)


class TestHashNumpy:
    """Tests for the hash_numpy function."""

    def test_hash_basic(self):
        """Test basic hashing of a simple array.

        Note: important to force dtype=np.int64 for stable dtype across platforms.

        Old numpy versions on Windows defaulted to int32.
        """
        # Force int64 so all platforms/python versions match; default numpy int changed to 64 bit
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        hash_val = hash_numpy(arr)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32  # 128-bit hash in hex = 32 characters
        assert hash_val == "6bacea16df1b95c005b4064f77ee5f4e"

    def test_hash_multi_dimensional(self):
        """Test hashing of a multi-dimensional array.

        Note: important to force dtype=np.int64 for stable dtype across platforms.

        Old numpy versions on Windows defaulted to int32.
        """
        arr = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
        hash_val = hash_numpy(arr)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 32
        assert hash_val == "f77b639891ad8c32688d19279b32ba0f"

    @given(st.lists(st.integers(min_value=INT64.min, max_value=INT64.max), min_size=1, max_size=100))
    def test_hash_deterministic_integers(self, values):
        """Test that hashing produces the same output for same input for integers.

        Note: important to force dtype=np.int64 for stable dtype across platforms.

        Old numpy versions on Windows defaulted to int32.

        Also: make sure hypothesis generates int64, not larger type ints, else it may give overflow errors.
        """
        arr = np.array(values, dtype=np.int64)
        hash1 = hash_numpy(arr)
        hash2 = hash_numpy(arr)

        assert hash1 == hash2

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
    def test_hash_deterministic_floats(self, values):
        """Test that hashing produces the same output for same input for floats"""
        arr = np.array(values, dtype=np.float64)
        hash1 = hash_numpy(arr)
        hash2 = hash_numpy(arr)

        assert hash1 == hash2

    def test_hash_different_value(self):
        """Test that different arrays produce different hashes."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 3, 2])

        hash1 = hash_numpy(arr1)
        hash2 = hash_numpy(arr2)

        assert hash1 != hash2

    def test_hash_different_shape(self):
        arr1 = np.arange(24)
        arr2 = arr1.reshape((2, 3, 4))

        hash1 = hash_numpy(arr1)
        hash2 = hash_numpy(arr2)

        assert hash1 != hash2

    def test_hash_different_dtype(self):
        arr1 = np.array([1, 2, 3], dtype=np.int32)
        arr2 = arr1.astype(np.float32)

        hash1 = hash_numpy(arr1)
        hash2 = hash_numpy(arr2)

        assert hash1 != hash2

    def test_hash_raise_BufferError_Fortran_order(self):
        """Test that Fortran-ordered arrays raise a BufferError.

        Note: important to force dtype=np.int64 for stable dtype across platforms.

        Old numpy versions on Windows defaulted to int32.
        """
        arr_f = np.asfortranarray(np.arange(12, dtype=np.int64).reshape((3, 4)))
        arr_c = np.ascontiguousarray(arr_f)
        with pytest.raises(BufferError, match=r"underlying buffer is not C-contiguous"):
            hash_numpy(arr_f)

        # when C-ordered it should work
        assert hash_numpy(arr_c) == "8f429eee81165ed82ee0d8470517d899"


class TestArrayLRUCache:
    """Test the ArrayLRUCache class."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        func = Mock(return_value=42)
        cache = ArrayLRUCache(func, maxsize=10)

        assert cache.func == func
        assert cache.maxsize == 10
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_hit(self):
        """Test cache hit increments hits counter."""
        func = Mock(return_value=42)
        cache = ArrayLRUCache(func)

        # First call - miss
        result1 = cache(1, 2, 3)
        assert result1 == 42
        assert cache.hits == 0
        assert cache.misses == 1
        assert func.call_count == 1

        # Second call - hit
        result2 = cache(1, 2, 3)
        assert result2 == 42
        assert cache.hits == 1
        assert cache.misses == 1
        assert func.call_count == 1  # Should not be called again

    def test_cache_miss(self):
        """Test cache miss with different arguments."""
        func = Mock(side_effect=[1, 2, 3])
        cache = ArrayLRUCache(func)

        result1 = cache(1)
        result2 = cache(2)
        result3 = cache(3)

        assert result1 == 1
        assert result2 == 2
        assert result3 == 3
        assert cache.misses == 3
        assert cache.hits == 0
        assert func.call_count == 3

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache exceeds maxsize."""
        func = Mock(side_effect=range(10))
        cache = ArrayLRUCache(func, maxsize=3)

        # Fill cache with 3 items
        for i in range(3):
            cache(i)

        assert len(cache.cache) == 3
        assert cache.misses == 3

        # Add 4th item - should evict LRU (0)
        _ = cache(3)
        assert len(cache.cache) == 3
        assert cache.misses == 4

        # Try to access evicted item - should cause a miss
        _ = cache(0)
        assert cache.misses == 5
        assert func.call_count == 5

    def test_cache_lru_bump(self):
        """Test that accessing an item bumps its position in LRU."""
        func = Mock(side_effect=range(10))
        cache = ArrayLRUCache(func, maxsize=3)

        for i in range(3):
            cache(i)

        # Access `0` (cached), then cache new item, should cause `1` to evict.
        cache(0)
        cache(3)

        # Verify 1 was evicted
        _result = cache(1)
        assert cache.misses == 5
        assert func.call_count == 5
        assert cache.hits == 1

    def test_cache_with_numpy_array_arguments(self):
        """Test caching with numpy array arguments.

        Note: important to force dtype=np.int64 for stable dtype across platforms.

        Old numpy versions on Windows defaulted to int32.
        """
        call_count = [0]

        def add_one(arr):
            call_count[0] += 1
            return arr + 1

        cache = ArrayLRUCache(add_one)

        arr = np.array([1, 2, 3], dtype=np.int64)
        result1 = cache(arr)
        result2 = cache(arr)
        assert (
            list(cache.cache.keys())[0][0].digest == "352f2aa49899327a9d0297ac0549865e"
        )  # digest of np.array([1,2,3], dtype =np.int64)

        assert_array_equal(result1, [2, 3, 4])
        assert_array_equal(result2, [2, 3, 4])
        assert call_count[0] == 1
        assert cache.hits == 1
        assert cache.misses == 1

    def test_cache_returns_copy_of_numpy_array(self):
        """Test that cache returns a copy of numpy arrays, not the original.

        This is crucial since numpy arrays are mutable.

        Any operations on the returned array should not affect the cached version (which would invalidate the hash).
        """

        def return_array(x):
            """Return a copy of input, to simulate a real function that does something.

            The pathological case that simply will return `x`, causes modifications of input `x` to propagate to the cache.

            Simply returning the same object is however not a case that needs to be cached.
            """
            return x.copy()

        cache = ArrayLRUCache(return_array)
        arr = np.array([1, 2, 3])

        result1 = cache(arr)
        assert hash_numpy(result1) == hash_numpy(arr)
        assert not np.shares_memory(result1, arr)

        result1[0] = 999
        result2 = cache(arr)  # Hit - returns copy
        assert result2[0] == 1  # Should not be affected by modifications

        # Now modify the cache contents, which should modify what is returned
        key = list(cache.cache)[0]
        cached = cache.cache[key]
        cached[0] = -1  # Modify the cached array directly
        result3 = cache(arr)
        assert result3[0] == -1
        assert hash_numpy(result3) != hash_numpy(arr)

    @pytest.mark.parametrize(
        "args, kwargs, expected_key",
        [
            ((0, 1), {}, (0, 1, (), 2, ())),
            ((0, 1), {"b": 5}, (0, 1, (), 5, ())),
            ((0, 1, 2), {"q": 3}, (0, 1, (2,), 2, (("q", 3),))),
            (
                (0, 1, 2),
                {"q": 3, "c": "c"},
                (
                    0,
                    1,
                    (2,),
                    2,
                    (("c", "c"), ("q", 3)),
                ),
            ),
        ],
    )
    def test_variadic_arg_handling_hashable(self, args, kwargs, expected_key):
        """Test handling of variadic arguments with hashable inputs.

        Note: Not sure if usage with variadic functions is advisable or should be encouraged.
        The test is mainly for constistency sake.

        For handling variadic arguments the following holds:
        - *args is processed as a tuple, when *args is empty the tuple is empty
        - **kwargs is processed as a sorted tuple of key-value pairs, when **kwargs is empty the tuple is empty
        - named keyword arguments (i.e. `b`) are placed before the kwargs tuple.
        """

        def func(x, a, *arg, b=2, **kwargs):
            return -1

        cache = ArrayLRUCache(func)
        key = cache._make_key(args, kwargs)
        assert key == expected_key


class Test_array_cache:
    def test_disable_cache(self):
        """Test that disabling the cache calls the function."""
        calls = [0]

        def func(x):
            calls[0] += 1
            return x

        cache = array_cache(maxsize=5)(func)
        arr = np.array([1, 2, 3])
        result = cache(arr)
        assert not np.shares_memory(result, arr)
        _ = cache(arr)
        info = cache.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert isinstance(cache._target, ArrayLRUCache)

        cache.enable_cache(False)
        result = cache(arr)
        assert np.shares_memory(result, arr)
        info = cache.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert cache._target == func

    def test_disable_cache_and_clear(self):
        """Test that disabling the cache calls the function."""
        calls = [0]

        def func(x):
            calls[0] += 1
            return x

        cache = array_cache(maxsize=5)(func)
        arr = np.array([1, 2, 3])
        result = cache(arr)
        assert not np.shares_memory(result, arr)
        _ = cache(arr)
        info = cache.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        assert isinstance(cache._target, ArrayLRUCache)

        cache.enable_cache(False, True)
        result = cache(arr)
        assert np.shares_memory(result, arr)
        info = cache.cache_info()
        assert info.hits == 0
        assert info.misses == 0
        assert cache._target == func

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves function name and docstring."""

        @array_cache()
        def my_function(x):
            """function docstring"""
            return x * 2

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "function docstring"
