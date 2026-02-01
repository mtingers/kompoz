"""Async combinator support."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic

from kompoz._core import Combinator
from kompoz._types import T, TraceConfig, TraceHook, _trace_config, _trace_hook


# =============================================================================
# Async Support
# =============================================================================


class AsyncCombinator(ABC, Generic[T]):
    """
    Base class for async combinators.

    Similar to Combinator but uses async/await.
    Supports tracing via use_tracing() context manager.
    """

    @abstractmethod
    async def _execute(self, ctx: T) -> tuple[bool, T]:
        """Internal execution - subclasses implement this."""
        ...

    async def run(self, ctx: T) -> tuple[bool, T]:
        """
        Execute the combinator asynchronously.

        If tracing is enabled via use_tracing(), this will automatically
        trace the execution.
        """
        hook = _trace_hook.get()
        if hook is not None:
            config = _trace_config.get()
            return await _async_traced_run(self, ctx, hook, config, depth=0)

        return await self._execute(ctx)

    def __and__(self, other: AsyncCombinator[T]) -> AsyncCombinator[T]:
        return _AsyncAnd(self, other)

    def __or__(self, other: AsyncCombinator[T]) -> AsyncCombinator[T]:
        return _AsyncOr(self, other)

    def __invert__(self) -> AsyncCombinator[T]:
        return _AsyncNot(self)

    def __rshift__(self, other: AsyncCombinator[T]) -> AsyncCombinator[T]:
        return _AsyncThen(self, other)

    def if_else(
        self, then_branch: AsyncCombinator[T], else_branch: AsyncCombinator[T]
    ) -> AsyncCombinator[T]:
        """
        Create a conditional: if self succeeds, run then_branch; else run else_branch.

        Example:
            await is_premium.if_else(apply_discount, charge_full_price).run(user)

        Unlike OR (|), this always executes exactly one branch based on the condition.
        """
        return _AsyncIfThenElse(self, then_branch, else_branch)

    async def __call__(self, ctx: T) -> tuple[bool, T]:
        return await self.run(ctx)


def _get_async_combinator_name(combinator: AsyncCombinator) -> str:
    """Get a human-readable name for an async combinator."""
    # Lazy imports to avoid circular dependencies
    from kompoz._async_validation import (
        AsyncValidatingPredicate,
        _AsyncValidatingAnd,
        _AsyncValidatingNot,
        _AsyncValidatingOr,
        _AsyncParallelAnd,
        _AsyncParallelValidatingAnd,
        _AsyncParallelOr,
        _AsyncParallelValidatingOr,
    )
    from kompoz._caching import AsyncCachedPredicate
    from kompoz._retry import AsyncRetry
    from kompoz._concurrency import AsyncTimeout, AsyncLimited, AsyncCircuitBreaker

    # Check validating types before base types
    if isinstance(combinator, AsyncValidatingPredicate):
        return f"AsyncValidatingPredicate({combinator.name})"
    if isinstance(combinator, _AsyncValidatingAnd):
        return "AsyncValidatingAND"
    if isinstance(combinator, _AsyncValidatingOr):
        return "AsyncValidatingOR"
    if isinstance(combinator, _AsyncValidatingNot):
        return "AsyncValidatingNOT"
    if isinstance(combinator, AsyncCachedPredicate):
        return f"AsyncCachedPredicate({combinator.name})"
    if isinstance(combinator, AsyncPredicate):
        return f"AsyncPredicate({combinator.name})"
    if isinstance(combinator, AsyncTransform):
        return f"AsyncTransform({combinator.name})"
    if isinstance(combinator, _AsyncAnd):
        return "AsyncAND"
    if isinstance(combinator, _AsyncOr):
        return "AsyncOR"
    if isinstance(combinator, _AsyncNot):
        return "AsyncNOT"
    if isinstance(combinator, _AsyncThen):
        return "AsyncTHEN"
    if isinstance(combinator, _AsyncIfThenElse):
        return "AsyncIF_THEN_ELSE"
    if isinstance(combinator, AsyncRetry):
        return f"AsyncRetry({combinator.name})"
    if isinstance(combinator, _AsyncParallelAnd):
        return "AsyncParallelAND"
    if isinstance(combinator, _AsyncParallelValidatingAnd):
        return "AsyncParallelValidatingAND"
    if isinstance(combinator, _AsyncParallelOr):
        return "AsyncParallelOR"
    if isinstance(combinator, _AsyncParallelValidatingOr):
        return "AsyncParallelValidatingOR"
    if isinstance(combinator, AsyncTimeout):
        return f"AsyncTimeout({combinator.timeout}s)"
    if isinstance(combinator, AsyncLimited):
        return f"AsyncLimited(max={combinator.max_concurrent})"
    if isinstance(combinator, AsyncCircuitBreaker):
        return f"AsyncCircuitBreaker(threshold={combinator.failure_threshold})"
    return repr(combinator)


async def _async_traced_run(
    combinator: AsyncCombinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig,
    depth: int = 0,
) -> tuple[bool, T]:
    """Execute an async combinator with tracing (iteratively where possible)."""
    return await _async_traced_run_iterative(combinator, ctx, hook, config, depth)


def _is_async_composite(combinator: AsyncCombinator) -> bool:
    """Check if combinator is a composite type for async tracing."""
    from kompoz._async_validation import (
        _AsyncParallelAnd,
        _AsyncParallelValidatingAnd,
        _AsyncParallelOr,
        _AsyncParallelValidatingOr,
    )

    return isinstance(
        combinator,
        (
            _AsyncAnd,
            _AsyncOr,
            _AsyncNot,
            _AsyncThen,
            _AsyncIfThenElse,
            _AsyncParallelAnd,
            _AsyncParallelValidatingAnd,
            _AsyncParallelOr,
            _AsyncParallelValidatingOr,
        ),
    )


async def _async_traced_run_iterative(
    root: AsyncCombinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig,
    initial_depth: int = 0,
) -> tuple[bool, T]:
    """Execute async combinator tree with tracing using explicit work list."""

    # For async, we use a work list approach but still need to await
    # We process the tree by flattening chains where possible

    async def process_node(
        combinator: AsyncCombinator[T],
        current_ctx: T,
        depth: int,
    ) -> tuple[bool, T]:
        """Process a single node with tracing."""

        # Check depth limit
        if config.max_depth is not None and depth > config.max_depth:
            return await combinator._execute(current_ctx)

        name = _get_async_combinator_name(combinator)
        is_composite = _is_async_composite(combinator)

        # Skip composite combinators if leaf_only mode
        if config.include_leaf_only and is_composite:
            if config.nested:
                return await process_composite_no_span(combinator, current_ctx, depth)
            return await combinator._execute(current_ctx)

        # Call on_enter
        span = hook.on_enter(name, current_ctx, depth)
        start = time.perf_counter()

        try:
            if config.nested and is_composite:
                ok, result = await process_composite(
                    combinator, current_ctx, depth, span, name, start
                )
            else:
                ok, result = await combinator._execute(current_ctx)
                duration_ms = (time.perf_counter() - start) * 1000
                hook.on_exit(span, name, ok, duration_ms, depth)

            return ok, result

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_error(span, name, e, duration_ms, depth)
            raise

    async def process_composite(
        combinator: AsyncCombinator[T],
        current_ctx: T,
        depth: int,
        span: Any,
        name: str,
        start: float,
    ) -> tuple[bool, T]:
        """Process composite combinator with proper span handling."""

        if isinstance(combinator, _AsyncAnd):
            # Flatten AND chain
            children = _flatten_async_and(combinator)
            ctx = current_ctx
            for child in children:
                ok, ctx = await process_node(child, ctx, depth + 1)
                if not ok:
                    duration_ms = (time.perf_counter() - start) * 1000
                    hook.on_exit(span, name, False, duration_ms, depth)
                    return False, ctx
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, True, duration_ms, depth)
            return True, ctx

        elif isinstance(combinator, _AsyncOr):
            # Flatten OR chain
            children = _flatten_async_or(combinator)
            ctx = current_ctx
            for child in children:
                ok, ctx = await process_node(child, ctx, depth + 1)
                if ok:
                    duration_ms = (time.perf_counter() - start) * 1000
                    hook.on_exit(span, name, True, duration_ms, depth)
                    return True, ctx
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, False, duration_ms, depth)
            return False, ctx

        elif isinstance(combinator, _AsyncNot):
            # Handle chained NOTs iteratively
            current: AsyncCombinator[T] = combinator
            invert_count = 0
            while isinstance(current, _AsyncNot):
                invert_count += 1
                current = current.inner

            ok, result = await process_node(current, current_ctx, depth + 1)
            if invert_count % 2 == 1:
                ok = not ok

            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, ok, duration_ms, depth)
            return ok, result

        elif isinstance(combinator, _AsyncThen):
            # Flatten THEN chain
            children = _flatten_async_then(combinator)
            ctx = current_ctx
            for child in children[:-1]:
                _, ctx = await process_node(child, ctx, depth + 1)
            ok, ctx = await process_node(children[-1], ctx, depth + 1)
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, ok, duration_ms, depth)
            return ok, ctx

        elif isinstance(combinator, _AsyncIfThenElse):
            cond_ok, new_ctx = await process_node(
                combinator.condition, current_ctx, depth + 1
            )
            if cond_ok:
                ok, result = await process_node(
                    combinator.then_branch, new_ctx, depth + 1
                )
            else:
                ok, result = await process_node(
                    combinator.else_branch, new_ctx, depth + 1
                )
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, ok, duration_ms, depth)
            return ok, result

        elif isinstance(
            combinator, (_AsyncParallelAnd, _AsyncParallelValidatingAnd)
        ):
            # Trace each child concurrently
            async def _trace_child(child: AsyncCombinator[T]) -> tuple[bool, T]:
                return await process_node(child, current_ctx, depth + 1)

            results = await asyncio.gather(
                *(_trace_child(child) for child in combinator.children)
            )
            all_ok = all(ok for ok, _ in results)
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, all_ok, duration_ms, depth)
            return all_ok, current_ctx

        # Fallback
        ok, result = await combinator._execute(current_ctx)
        duration_ms = (time.perf_counter() - start) * 1000
        hook.on_exit(span, name, ok, duration_ms, depth)
        return ok, result

    async def process_composite_no_span(
        combinator: AsyncCombinator[T],
        current_ctx: T,
        depth: int,
    ) -> tuple[bool, T]:
        """Process composite without creating span (leaf_only mode)."""

        if isinstance(combinator, _AsyncAnd):
            children = _flatten_async_and(combinator)
            ctx = current_ctx
            for child in children:
                ok, ctx = await process_node(child, ctx, depth + 1)
                if not ok:
                    return False, ctx
            return True, ctx

        elif isinstance(combinator, _AsyncOr):
            children = _flatten_async_or(combinator)
            ctx = current_ctx
            for child in children:
                ok, ctx = await process_node(child, ctx, depth + 1)
                if ok:
                    return True, ctx
            return False, ctx

        elif isinstance(combinator, _AsyncNot):
            current: AsyncCombinator[T] = combinator
            invert_count = 0
            while isinstance(current, _AsyncNot):
                invert_count += 1
                current = current.inner

            ok, result = await process_node(current, current_ctx, depth + 1)
            if invert_count % 2 == 1:
                ok = not ok
            return ok, result

        elif isinstance(combinator, _AsyncThen):
            children = _flatten_async_then(combinator)
            ctx = current_ctx
            for child in children[:-1]:
                _, ctx = await process_node(child, ctx, depth + 1)
            return await process_node(children[-1], ctx, depth + 1)

        elif isinstance(combinator, _AsyncIfThenElse):
            cond_ok, new_ctx = await process_node(
                combinator.condition, current_ctx, depth + 1
            )
            if cond_ok:
                return await process_node(combinator.then_branch, new_ctx, depth + 1)
            else:
                return await process_node(combinator.else_branch, new_ctx, depth + 1)

        elif isinstance(
            combinator, (_AsyncParallelAnd, _AsyncParallelValidatingAnd)
        ):
            async def _trace_child_no_span(
                child: AsyncCombinator[T],
            ) -> tuple[bool, T]:
                return await process_node(child, current_ctx, depth + 1)

            results = await asyncio.gather(
                *(_trace_child_no_span(child) for child in combinator.children)
            )
            all_ok = all(ok for ok, _ in results)
            return all_ok, current_ctx

        return await combinator._execute(current_ctx)

    return await process_node(root, ctx, initial_depth)


def _flatten_async_and(combinator: AsyncCombinator[T]) -> list[AsyncCombinator[T]]:
    """Flatten nested _AsyncAnd into a list (iterative to avoid recursion)."""
    result: list[AsyncCombinator[T]] = []
    stack: list[AsyncCombinator[T]] = [combinator]
    while stack:
        current = stack.pop()
        if isinstance(current, _AsyncAnd):
            stack.append(current.right)
            stack.append(current.left)
        else:
            result.append(current)
    return result


def _flatten_async_or(combinator: AsyncCombinator[T]) -> list[AsyncCombinator[T]]:
    """Flatten nested _AsyncOr into a list (iterative to avoid recursion)."""
    result: list[AsyncCombinator[T]] = []
    stack: list[AsyncCombinator[T]] = [combinator]
    while stack:
        current = stack.pop()
        if isinstance(current, _AsyncOr):
            stack.append(current.right)
            stack.append(current.left)
        else:
            result.append(current)
    return result


def _flatten_async_then(combinator: AsyncCombinator[T]) -> list[AsyncCombinator[T]]:
    """Flatten nested _AsyncThen into a list (iterative to avoid recursion)."""
    result: list[AsyncCombinator[T]] = []
    stack: list[AsyncCombinator[T]] = [combinator]
    while stack:
        current = stack.pop()
        if isinstance(current, _AsyncThen):
            stack.append(current.right)
            stack.append(current.left)
        else:
            result.append(current)
    return result


@dataclass
class _AsyncAnd(AsyncCombinator[T]):
    left: AsyncCombinator[T]
    right: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        # Flatten chain and iterate to avoid deep recursion
        for combinator in _flatten_async_and(self):
            ok, ctx = await combinator._execute(ctx)
            if not ok:
                return False, ctx
        return True, ctx


@dataclass
class _AsyncOr(AsyncCombinator[T]):
    left: AsyncCombinator[T]
    right: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        # Flatten chain and iterate to avoid deep recursion
        for combinator in _flatten_async_or(self):
            ok, ctx = await combinator._execute(ctx)
            if ok:
                return True, ctx
        return False, ctx


@dataclass
class _AsyncNot(AsyncCombinator[T]):
    inner: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        # Handle chained NOT (e.g., ~~~a) iteratively
        current: AsyncCombinator[T] = self
        invert_count = 0
        while isinstance(current, _AsyncNot):
            invert_count += 1
            current = current.inner
        ok, ctx = await current._execute(ctx)
        # Odd number of inversions flips the result
        if invert_count % 2 == 1:
            ok = not ok
        return ok, ctx


@dataclass
class _AsyncThen(AsyncCombinator[T]):
    left: AsyncCombinator[T]
    right: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        # Flatten chain and iterate to avoid deep recursion
        combinators = _flatten_async_then(self)
        for combinator in combinators[:-1]:
            _, ctx = await combinator._execute(ctx)
        # Return the result of the last combinator
        return await combinators[-1]._execute(ctx)


@dataclass
class _AsyncIfThenElse(AsyncCombinator[T]):
    """
    Async conditional combinator: if condition succeeds, run then_branch; else run else_branch.

    Unlike OR which short-circuits on success, this explicitly branches:
    - condition ? then_branch : else_branch
    - IF condition THEN then_branch ELSE else_branch
    """

    condition: AsyncCombinator[T]
    then_branch: AsyncCombinator[T]
    else_branch: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        cond_ok, new_ctx = await self.condition._execute(ctx)
        if cond_ok:
            return await self.then_branch._execute(new_ctx)
        else:
            return await self.else_branch._execute(new_ctx)


class AsyncPredicate(AsyncCombinator[T]):
    """
    An async predicate that checks a condition.

    Example:
        @async_rule
        async def has_permission(user):
            return await db.check_permission(user.id)
    """

    def __init__(self, fn: Callable[[T], Any], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "async_predicate")

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        result = await self.fn(ctx)
        return bool(result), ctx

    def __repr__(self) -> str:
        return f"AsyncPredicate({self.name})"


class AsyncPredicateFactory(Generic[T]):
    """Factory for parameterized async predicates."""

    def __init__(self, fn: Callable[..., Any], name: str):
        self._fn = fn
        self._name = name
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> AsyncPredicate[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return AsyncPredicate(lambda ctx: self._fn(ctx, *args, **kwargs), name)

    def __repr__(self) -> str:
        return f"AsyncPredicateFactory({self._name})"


class AsyncTransform(AsyncCombinator[T]):
    """
    An async transform that modifies context.

    Example:
        @async_pipe
        async def fetch_profile(user):
            user.profile = await api.get_profile(user.id)
            return user

    For concurrency-safe error access, use run_with_error() which returns
    both the result and any error that occurred, rather than storing
    the error in an instance variable.

    Attributes:
        last_error: The last exception that caused failure (if any).
                   Note: This is not concurrency-safe. For concurrent usage,
                   use run_with_error() instead.
    """

    def __init__(self, fn: Callable[[T], Any], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "async_transform")
        # Deprecated: kept for backwards compatibility but not concurrency-safe
        self.last_error: Exception | None = None

    async def run_with_error(self, ctx: T) -> tuple[bool, T, Exception | None]:
        """
        Execute the transform and return result with error information.
        
        This method is concurrency-safe as it returns the error rather than
        storing it in an instance variable.
        
        Returns:
            Tuple of (success, result_context, error_or_none)
        """
        try:
            result = await self.fn(ctx)
            return True, result, None
        except Exception as e:
            return False, ctx, e

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        ok, result, error = await self.run_with_error(ctx)
        # Update instance var for backwards compatibility (not concurrency-safe)
        self.last_error = error
        return ok, result

    def __repr__(self) -> str:
        return f"AsyncTransform({self.name})"


class AsyncTransformFactory(Generic[T]):
    """Factory for parameterized async transforms."""

    def __init__(self, fn: Callable[..., Any], name: str):
        self._fn = fn
        self._name = name
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> AsyncTransform[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return AsyncTransform(lambda ctx: self._fn(ctx, *args, **kwargs), name)

    def __repr__(self) -> str:
        return f"AsyncTransformFactory({self._name})"


def async_rule(fn: Callable[[T], Any]) -> AsyncPredicate[T]:
    """
    Decorator to create an async predicate.

    Example:
        @async_rule
        async def has_permission(user):
            return await db.check_permission(user.id)

        ok, _ = await has_permission.run(user)
    """
    return AsyncPredicate(fn, fn.__name__)


def async_rule_args(fn: Callable[..., Any]) -> AsyncPredicateFactory[Any]:
    """
    Decorator to create a parameterized async predicate factory.

    Example:
        @async_rule_args
        async def has_role(user, role):
            return await db.check_role(user.id, role)

        ok, _ = await has_role("admin").run(user)
    """
    return AsyncPredicateFactory(fn, fn.__name__)


def async_pipe(fn: Callable[[T], Any]) -> AsyncTransform[T]:
    """
    Decorator to create an async transform.

    Example:
        @async_pipe
        async def enrich_user(user):
            user.profile = await api.get_profile(user.id)
            return user
    """
    return AsyncTransform(fn, fn.__name__)


def async_pipe_args(fn: Callable[..., Any]) -> AsyncTransformFactory[Any]:
    """
    Decorator to create a parameterized async transform factory.

    Example:
        @async_pipe_args
        async def fetch_data(ctx, endpoint):
            ctx.data = await api.get(endpoint)
            return ctx
    """
    return AsyncTransformFactory(fn, fn.__name__)


def async_if_then_else(
    condition: AsyncCombinator[T],
    then_branch: AsyncCombinator[T],
    else_branch: AsyncCombinator[T],
) -> AsyncCombinator[T]:
    """
    Create an async conditional combinator: if condition succeeds, run then_branch; else run else_branch.

    Example:
        from kompoz import async_if_then_else, async_rule, async_pipe

        @async_rule
        async def is_premium(user):
            return await db.is_premium(user.id)

        @async_pipe
        async def apply_discount(user):
            user.discount = 0.2
            return user

        @async_pipe
        async def charge_full_price(user):
            user.discount = 0
            return user

        pricing = async_if_then_else(is_premium, apply_discount, charge_full_price)
        ok, user = await pricing.run(user)

    Unlike OR (|) which is a fallback (try a, if fail try b), async_if_then_else
    explicitly branches based on the condition result.
    """
    return _AsyncIfThenElse(condition, then_branch, else_branch)


