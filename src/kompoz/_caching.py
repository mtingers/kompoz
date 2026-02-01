"""Caching and memoization support."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Generic, overload

from kompoz._async import AsyncCombinator
from kompoz._core import Combinator
from kompoz._types import T, _cache_store


@contextmanager
def use_cache():
    """
    Context manager to enable caching for all cached rules in scope.

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


class CachedPredicate(Combinator[T]):
    """
    A predicate that caches its result within a use_cache() scope.

    The cache key is based on the predicate name and the context's id or hash.
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

    def _get_cache_key(self, ctx: T) -> str:
        """Generate a cache key for this context."""
        ctx_key = self.key_fn(ctx)
        return f"{self.name}:{ctx_key}"

    def _execute(self, ctx: T) -> tuple[bool, T]:
        cache = _cache_store.get()

        if cache is not None:
            key = self._get_cache_key(ctx)
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

    def _get_cache_key(self, ctx: T) -> str:
        """Generate a cache key for this context."""
        ctx_key = self.key_fn(ctx)
        return f"{self.name}:{ctx_key}"

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        cache = _cache_store.get()

        if cache is not None:
            key = self._get_cache_key(ctx)
            if key in cache:
                return cache[key]

            result_val = await self.fn(ctx)
            result = bool(result_val), ctx
            cache[key] = result
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


