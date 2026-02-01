"""Caching and memoization support."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Generic, overload

from kompoz._async import AsyncCombinator
from kompoz._core import Combinator
from kompoz._types import T, _cache_store

# Thread-safe shared cache for multi-threaded scenarios
_shared_cache: dict[str, tuple[bool, Any]] | None = None
_shared_cache_lock = threading.Lock()


@contextmanager
def use_cache():
    """
    Context manager to enable caching for all cached rules in scope.

    Note: This uses ContextVar which is per-thread. For multi-threaded
    scenarios where threads should share a cache, use use_cache_shared().

    Example:
        with use_cache():
            # Same predicate called multiple times will only execute once
            rule.run(user)
            rule.run(user)  # Uses cached result
    """
    old_cache = _cache_store.get()
    _cache_store.set({})
    try:
        yield
    finally:
        _cache_store.set(old_cache)


@contextmanager
def use_cache_shared():
    """
    Context manager to enable a thread-safe shared cache.

    Unlike use_cache() which uses ContextVar (per-thread), this creates
    a globally shared cache that's safe to use across multiple threads.

    Example:
        with use_cache_shared():
            # Start threads that will all share the same cache
            threads = [Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    """
    global _shared_cache
    with _shared_cache_lock:
        old_cache = _shared_cache
        _shared_cache = {}
    try:
        yield
    finally:
        with _shared_cache_lock:
            _shared_cache = old_cache


class CachedPredicate(Combinator[T]):
    """
    A predicate that caches its result within a use_cache() or use_cache_shared() scope.

    The cache key is based on the predicate name and the context's id or hash.
    Thread-safe: uses locking to prevent duplicate execution in multi-threaded scenarios.

    Cache lookup order:
    1. Thread-local cache (from use_cache())
    2. Shared cache (from use_cache_shared())
    3. No caching if neither is active
    """

    def __init__(
        self,
        fn: Callable[[T], bool],
        name: str | None = None,
        key_fn: Callable[[T], Any] | None = None,
    ):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "cached_predicate")
        self.key_fn = key_fn or id  # Default to using object id
        self._lock = threading.Lock()

    def _get_cache_key(self, ctx: T) -> str:
        """Generate a cache key for this context."""
        ctx_key = self.key_fn(ctx)
        return f"{self.name}:{ctx_key}"

    def _get_cache(self) -> dict[str, tuple[bool, Any]] | None:
        """Get the active cache (thread-local or shared)."""
        # First check thread-local cache
        cache = _cache_store.get()
        if cache is not None:
            return cache
        # Fall back to shared cache
        return _shared_cache

    def _execute(self, ctx: T) -> tuple[bool, T]:
        cache = self._get_cache()

        if cache is not None:
            key = self._get_cache_key(ctx)

            # Fast path: already cached (no lock needed for read)
            if key in cache:
                return cache[key]

            # Slow path: acquire lock and double-check
            with self._lock:
                if key in cache:
                    return cache[key]

                result = self.fn(ctx), ctx
                cache[key] = result
                return result

        return self.fn(ctx), ctx

    def __repr__(self) -> str:
        return f"CachedPredicate({self.name})"


class CachedPredicateFactory(Generic[T]):
    """Factory for parameterized cached predicates."""

    def __init__(
        self,
        fn: Callable[..., bool],
        name: str,
        key_fn: Callable[[T], Any] | None = None,
    ):
        self._fn = fn
        self._name = name
        self._key_fn = key_fn
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> CachedPredicate[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return CachedPredicate(
            lambda ctx: self._fn(ctx, *args, **kwargs), name, self._key_fn
        )

    def __repr__(self) -> str:
        return f"CachedPredicateFactory({self._name})"


@overload
def cached_rule(
    fn: Callable[[T], bool], *, key: Callable[[T], Any] | None = None
) -> CachedPredicate[T]: ...


@overload
def cached_rule(
    fn: None = None, *, key: Callable[[T], Any] | None = None
) -> Callable[[Callable[[T], bool]], CachedPredicate[T]]: ...


def cached_rule(
    fn: Callable[[T], bool] | None = None, *, key: Callable[[T], Any] | None = None
) -> CachedPredicate[T] | Callable[[Callable[[T], bool]], CachedPredicate[T]]:
    """
    Decorator to create a cached rule.

    Results are cached within a use_cache() scope.

    Example:
        @cached_rule
        def expensive_check(user):
            # This will only run once per user within use_cache()
            return slow_database_query(user.id)

        @cached_rule(key=lambda u: u.id)
        def check_by_id(user):
            return api_call(user.id)

        with use_cache():
            rule.run(user)
            rule.run(user)  # Uses cached result
    """

    def decorator(f: Callable[[T], bool]) -> CachedPredicate[T]:
        return CachedPredicate(f, f.__name__, key)

    if fn is not None:
        return decorator(fn)
    return decorator


class AsyncCachedPredicate(AsyncCombinator[T]):
    """
    An async predicate that caches its result within a use_cache() scope.

    Mirrors CachedPredicate but awaits the function.
    Concurrency-safe: uses asyncio.Lock to prevent duplicate execution
    when multiple tasks access the same cache key concurrently.
    """

    def __init__(
        self,
        fn: Callable[[T], Any],
        name: str | None = None,
        key_fn: Callable[[T], Any] | None = None,
    ):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "async_cached_predicate")
        self.key_fn = key_fn or id
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_lock = asyncio.Lock()  # Lock for creating per-key locks

    def _get_cache_key(self, ctx: T) -> str:
        """Generate a cache key for this context."""
        ctx_key = self.key_fn(ctx)
        return f"{self.name}:{ctx_key}"

    async def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a specific cache key."""
        async with self._lock_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        cache = _cache_store.get()

        if cache is not None:
            key = self._get_cache_key(ctx)

            # Fast path: already cached (no lock needed)
            if key in cache:
                return cache[key]

            # Slow path: acquire per-key lock and double-check
            lock = await self._get_lock(key)
            async with lock:
                # Double-check after acquiring lock
                if key in cache:
                    return cache[key]

                result_val = await self.fn(ctx)
                result = bool(result_val), ctx
                cache[key] = result

            # Clean up the per-key lock now that the value is cached
            async with self._lock_lock:
                self._locks.pop(key, None)

            return result

        result_val = await self.fn(ctx)
        return bool(result_val), ctx

    def __repr__(self) -> str:
        return f"AsyncCachedPredicate({self.name})"


class AsyncCachedPredicateFactory(Generic[T]):
    """Factory for parameterized async cached predicates."""

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        key_fn: Callable[[T], Any] | None = None,
    ):
        self._fn = fn
        self._name = name
        self._key_fn = key_fn
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> AsyncCachedPredicate[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return AsyncCachedPredicate(
            lambda ctx: self._fn(ctx, *args, **kwargs), name, self._key_fn
        )

    def __repr__(self) -> str:
        return f"AsyncCachedPredicateFactory({self._name})"


@overload
def async_cached_rule(
    fn: Callable[[T], Any], *, key: Callable[[T], Any] | None = None
) -> AsyncCachedPredicate[T]: ...


@overload
def async_cached_rule(
    fn: None = None, *, key: Callable[[T], Any] | None = None
) -> Callable[[Callable[[T], Any]], AsyncCachedPredicate[T]]: ...


def async_cached_rule(
    fn: Callable[[T], Any] | None = None, *, key: Callable[[T], Any] | None = None
) -> AsyncCachedPredicate[T] | Callable[[Callable[[T], Any]], AsyncCachedPredicate[T]]:
    """
    Decorator to create an async cached rule.

    Results are cached within a use_cache() scope.  Works with async functions.

    Example:
        @async_cached_rule
        async def expensive_check(user):
            return await slow_api_call(user.id)

        @async_cached_rule(key=lambda u: u.id)
        async def check_by_id(user):
            return await api_call(user.id)

        with use_cache():
            await rule.run(user)
            await rule.run(user)  # Uses cached result
    """

    def decorator(f: Callable[[T], Any]) -> AsyncCachedPredicate[T]:
        return AsyncCachedPredicate(f, f.__name__, key)

    if fn is not None:
        return decorator(fn)
    return decorator
