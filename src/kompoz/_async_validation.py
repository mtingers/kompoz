"""Async validation combinator support."""

from __future__ import annotations

import asyncio
import inspect
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

from kompoz._async import AsyncCombinator, _get_async_combinator_name
from kompoz._types import T
from kompoz._validation import ValidationResult

# =============================================================================
# Async Validation Support
# =============================================================================


class AsyncValidatingCombinator(AsyncCombinator[T]):
    """
    Base class for async combinators that support validation with error messages.

    Subclasses must implement the validate() method.
    """

    @abstractmethod
    async def validate(self, ctx: T) -> ValidationResult:
        """Run async validation and return result with errors."""
        ...

    def __and__(self, other: AsyncCombinator[T]) -> AsyncValidatingCombinator[T]:
        """Override & to create async validating AND."""
        return _AsyncValidatingAnd(self, other)

    def __or__(self, other: AsyncCombinator[T]) -> AsyncValidatingCombinator[T]:
        """Override | to create async validating OR."""
        return _AsyncValidatingOr(self, other)

    def __invert__(self) -> AsyncValidatingCombinator[T]:
        """Override ~ to create async validating NOT."""
        return _AsyncValidatingNot(self)


class AsyncValidatingPredicate(AsyncValidatingCombinator[T]):
    """
    An async predicate that provides an error message on failure.

    Example:
        @async_vrule(error="User must be an admin")
        async def is_admin(user):
            return await db.check_admin(user.id)

        result = await is_admin.validate(user)
        if not result.ok:
            print(result.errors)  # ["User must be an admin"]
    """

    def __init__(
        self,
        fn: Callable[[T], Any],
        name: str | None = None,
        error: str | Callable[[T], str] | None = None,
    ):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "async_predicate")
        self._error = error

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        result = await self.fn(ctx)
        return bool(result), ctx

    def get_error(self, ctx: T) -> str:
        """Get the error message for this predicate."""
        if self._error is None:
            return f"Check failed: {self.name}"
        if callable(self._error):
            return self._error(ctx)
        try:
            return self._error.format(ctx=ctx)
        except (KeyError, AttributeError, IndexError):
            return self._error

    async def validate(self, ctx: T) -> ValidationResult:
        """Run async validation and return result with errors."""
        ok, result = await self._execute(ctx)
        errors = [] if ok else [self.get_error(ctx)]
        return ValidationResult(ok=ok, errors=errors, ctx=result)

    def __repr__(self) -> str:
        return f"AsyncValidatingPredicate({self.name})"


class _AsyncValidatingAnd(AsyncValidatingCombinator[T]):
    """Async AND combinator that collects all validation errors."""

    def __init__(self, left: AsyncCombinator[T], right: AsyncCombinator[T]):
        self.left = left
        self.right = right

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        ok1, ctx = await self.left._execute(ctx)
        if not ok1:
            return False, ctx
        return await self.right._execute(ctx)

    async def validate(self, ctx: T) -> ValidationResult:
        """Validate both sides and collect all errors."""
        errors: list[str] = []

        # Flatten the AND chain iteratively
        to_validate: list[AsyncCombinator[T]] = []
        stack: list[AsyncCombinator[T]] = [self]
        while stack:
            current = stack.pop()
            if isinstance(current, _AsyncValidatingAnd):
                stack.append(current.right)
                stack.append(current.left)
            else:
                to_validate.append(current)

        for item in to_validate:
            if isinstance(item, AsyncValidatingCombinator):
                result = await item.validate(ctx)
                errors.extend(result.errors)
                ctx = result.ctx
            else:
                ok, ctx = await item._execute(ctx)
                if not ok:
                    errors.append(f"Check failed: {_get_async_combinator_name(item)}")

        return ValidationResult(ok=len(errors) == 0, errors=errors, ctx=ctx)


class _AsyncValidatingOr(AsyncValidatingCombinator[T]):
    """Async OR combinator for validation - passes if any succeeds."""

    def __init__(self, left: AsyncCombinator[T], right: AsyncCombinator[T]):
        self.left = left
        self.right = right

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        ok1, ctx = await self.left._execute(ctx)
        if ok1:
            return True, ctx
        return await self.right._execute(ctx)

    async def validate(self, ctx: T) -> ValidationResult:
        """Validate - passes if any in the chain passes."""
        to_validate: list[AsyncCombinator[T]] = []
        stack: list[AsyncCombinator[T]] = [self]
        while stack:
            current = stack.pop()
            if isinstance(current, _AsyncValidatingOr):
                stack.append(current.right)
                stack.append(current.left)
            else:
                to_validate.append(current)

        last_result: ValidationResult | None = None
        for item in to_validate:
            if isinstance(item, AsyncValidatingCombinator):
                result = await item.validate(ctx)
                if result.ok:
                    return result
                last_result = result
            else:
                ok, result_ctx = await item._execute(ctx)
                if ok:
                    return ValidationResult(ok=True, errors=[], ctx=result_ctx)
                last_result = ValidationResult(
                    ok=False,
                    errors=[f"Check failed: {_get_async_combinator_name(item)}"],
                    ctx=result_ctx,
                )

        return last_result or ValidationResult(ok=False, errors=["No conditions to check"], ctx=ctx)


class _AsyncValidatingNot(AsyncValidatingCombinator[T]):
    """Async NOT combinator for validation - inverts the result."""

    def __init__(self, inner: AsyncCombinator[T], error: str | None = None):
        self.inner = inner
        self._error = error

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = await self.inner._execute(ctx)
        return not ok, ctx

    async def validate(self, ctx: T) -> ValidationResult:
        """Validate - inverts the inner result."""
        if isinstance(self.inner, AsyncValidatingCombinator):
            inner_result = await self.inner.validate(ctx)
            if inner_result.ok:
                error_msg = self._error or "NOT condition failed (inner passed)"
                return ValidationResult(ok=False, errors=[error_msg], ctx=inner_result.ctx)
            else:
                return ValidationResult(ok=True, errors=[], ctx=inner_result.ctx)
        else:
            ok, result = await self.inner._execute(ctx)
            if ok:
                error_msg = self._error or f"NOT {_get_async_combinator_name(self.inner)} failed"
                return ValidationResult(ok=False, errors=[error_msg], ctx=result)
            else:
                return ValidationResult(ok=True, errors=[], ctx=result)


@overload
def async_vrule(
    fn: Callable[[T], Any], *, error: str | Callable[[T], str] | None = None
) -> AsyncValidatingPredicate[T]: ...


@overload
def async_vrule(
    fn: None = None, *, error: str | Callable[[T], str] | None = None
) -> Callable[[Callable[[T], Any]], AsyncValidatingPredicate[T]]: ...


def async_vrule(
    fn: Callable[[T], Any] | None = None,
    *,
    error: str | Callable[[T], str] | None = None,
) -> AsyncValidatingPredicate[T] | Callable[[Callable[[T], Any]], AsyncValidatingPredicate[T]]:
    """
    Decorator to create an async validating rule with an error message.

    Example:
        @async_vrule(error="User {ctx.name} must be an admin")
        async def is_admin(user):
            return await db.check_admin(user.id)

        @async_vrule(error=lambda u: f"{u.name} is banned!")
        async def is_not_banned(user):
            return not await db.is_banned(user.id)

        result = await is_admin.validate(user)
        result = await (is_admin & is_not_banned).validate(user)  # Collects all errors
    """

    def decorator(f: Callable[[T], Any]) -> AsyncValidatingPredicate[T]:
        return AsyncValidatingPredicate(f, f.__name__, error)

    if fn is not None:
        return decorator(fn)
    return decorator


@overload
def async_vrule_args(
    fn: Callable[..., Any], *, error: str | Callable[..., str] | None = None
) -> Callable[..., AsyncValidatingPredicate]: ...


@overload
def async_vrule_args(
    fn: None = None, *, error: str | Callable[..., str] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., AsyncValidatingPredicate]]: ...


def async_vrule_args(
    fn: Callable[..., Any] | None = None,
    *,
    error: str | Callable[..., str] | None = None,
) -> (
    Callable[..., AsyncValidatingPredicate]
    | Callable[[Callable[..., Any]], Callable[..., AsyncValidatingPredicate]]
):
    """
    Decorator to create a parameterized async validating rule factory.

    Example:
        @async_vrule_args(error="Score {score} is below minimum {min_score}")
        async def score_above(user, min_score):
            score = await db.get_score(user.id)
            return score >= min_score

        result = await score_above(700).validate(user)
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., AsyncValidatingPredicate]:
        sig = inspect.signature(f)

        def factory(*args: Any, **kwargs: Any) -> AsyncValidatingPredicate:
            name = f"{f.__name__}({', '.join(map(repr, args))})"

            def get_bound_params():
                try:
                    bound = sig.bind_partial(None, *args, **kwargs)
                    bound.apply_defaults()
                    params = dict(bound.arguments)
                    first_param_name = next(iter(sig.parameters.keys()))
                    params.pop(first_param_name, None)
                    return params
                except TypeError:
                    return kwargs

            err_msg: str | Callable[[Any], str] | None

            if error is None:
                err_msg = None
            elif callable(error):
                error_fn: Callable[..., str] = error

                def make_error_fn(ctx: Any) -> str:
                    return error_fn(ctx, *args, **kwargs)

                err_msg = make_error_fn
            else:
                template_str: str = error

                def make_formatted_error(ctx: Any) -> str:
                    format_context = {f"arg{i}": v for i, v in enumerate(args)}
                    format_context.update(get_bound_params())
                    format_context["ctx"] = ctx
                    try:
                        return template_str.format(**format_context)
                    except (KeyError, IndexError, AttributeError):
                        return template_str

                err_msg = make_formatted_error

            async def predicate_fn(ctx: Any) -> bool:
                return await f(ctx, *args, **kwargs)

            return AsyncValidatingPredicate(predicate_fn, name, err_msg)

        factory.__name__ = f.__name__
        return factory

    if fn is not None:
        return decorator(fn)
    return decorator


# =============================================================================
# Parallel Async AND
# =============================================================================


@dataclass
class _AsyncParallelAnd(AsyncCombinator[T]):
    """Async AND that runs all children concurrently via asyncio.gather().

    All children receive the **same original context** (not chained).
    Returns (all_ok, original_ctx).

    If fail_fast=True, cancels remaining tasks when first failure is detected.
    """

    children: list[AsyncCombinator[T]]
    fail_fast: bool = False

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        if self.fail_fast:
            return await self._execute_fail_fast(ctx)
        return await self._execute_wait_all(ctx)

    async def _execute_wait_all(self, ctx: T) -> tuple[bool, T]:
        results = await asyncio.gather(*(child._execute(ctx) for child in self.children))
        all_ok = all(ok for ok, _ in results)
        return all_ok, ctx

    async def _execute_fail_fast(self, ctx: T) -> tuple[bool, T]:
        """Execute with fail-fast: cancel remaining tasks on first failure."""
        tasks = [asyncio.create_task(child._execute(ctx)) for child in self.children]

        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    try:
                        ok, _ = task.result()
                        if not ok:
                            # Cancel remaining tasks and return failure
                            for p in pending:
                                p.cancel()
                            # Wait for cancellations to complete
                            if pending:
                                await asyncio.gather(*pending, return_exceptions=True)
                            return False, ctx
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        # On exception, cancel remaining and re-raise
                        for p in pending:
                            p.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        raise

            return True, ctx
        except Exception:
            # Ensure all tasks are cancelled on any exception
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise


@dataclass
class _AsyncParallelValidatingAnd(AsyncValidatingCombinator[T]):
    """Async validating AND that runs all children concurrently via asyncio.gather().

    All children receive the **same original context** (not chained).
    Merges error lists from all children.

    If fail_fast=True, cancels remaining tasks when first failure is detected.
    """

    children: list[AsyncCombinator[T]]
    fail_fast: bool = False

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        if self.fail_fast:
            return await self._execute_fail_fast(ctx)
        return await self._execute_wait_all(ctx)

    async def _execute_wait_all(self, ctx: T) -> tuple[bool, T]:
        results = await asyncio.gather(*(child._execute(ctx) for child in self.children))
        all_ok = all(ok for ok, _ in results)
        return all_ok, ctx

    async def _execute_fail_fast(self, ctx: T) -> tuple[bool, T]:
        """Execute with fail-fast: cancel remaining tasks on first failure."""
        tasks = [asyncio.create_task(child._execute(ctx)) for child in self.children]

        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    try:
                        ok, _ = task.result()
                        if not ok:
                            for p in pending:
                                p.cancel()
                            if pending:
                                await asyncio.gather(*pending, return_exceptions=True)
                            return False, ctx
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        for p in pending:
                            p.cancel()
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)
                        raise

            return True, ctx
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    async def validate(self, ctx: T) -> ValidationResult:
        """Validate all children concurrently and merge errors."""

        async def _validate_one(child: AsyncCombinator[T]) -> list[str]:
            if isinstance(child, AsyncValidatingCombinator):
                result = await child.validate(ctx)
                return result.errors
            else:
                ok, _ = await child._execute(ctx)
                if not ok:
                    return [f"Check failed: {_get_async_combinator_name(child)}"]
                return []

        error_lists = await asyncio.gather(*(_validate_one(child) for child in self.children))
        errors: list[str] = []
        for err_list in error_lists:
            errors.extend(err_list)
        return ValidationResult(ok=len(errors) == 0, errors=errors, ctx=ctx)


def parallel_and(
    *combinators: AsyncCombinator[T],
    fail_fast: bool = False,
) -> AsyncCombinator[T]:
    """Create an async AND that runs all children concurrently.

    Unlike ``&`` which chains sequentially (and passes modified context from
    left to right), ``parallel_and`` runs every child with the **same original
    context** using ``asyncio.gather()``.

    Returns ``(all_ok, original_ctx)`` — context is never modified.

    If all inputs are ``AsyncValidatingCombinator``, returns a validating
    variant that merges error lists.

    Args:
        *combinators: Combinators to run in parallel
        fail_fast: If True, cancel remaining tasks when first failure is detected.
                   This can improve performance when you don't need all results.

    Raises ``ValueError`` when called with no arguments.

    Example::

        result = await parallel_and(check_a, check_b, check_c).run(ctx)

        # With fail-fast for early termination
        result = await parallel_and(check_a, check_b, fail_fast=True).run(ctx)
    """
    if not combinators:
        raise ValueError("parallel_and() requires at least one combinator")

    children = list(combinators)
    if all(isinstance(c, AsyncValidatingCombinator) for c in children):
        return _AsyncParallelValidatingAnd(children, fail_fast=fail_fast)
    return _AsyncParallelAnd(children, fail_fast=fail_fast)


# =============================================================================
# Parallel Async OR
# =============================================================================


@dataclass
class _AsyncParallelOr(AsyncCombinator[T]):
    """Async OR that runs children concurrently, returns first success.

    All children receive the **same original context** (not chained).
    Returns (first_success, original_ctx) or (False, original_ctx) if all fail.

    If cancel_on_success=True (default), cancels remaining tasks when first
    success is detected.
    """

    children: list[AsyncCombinator[T]]
    cancel_on_success: bool = True

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        if self.cancel_on_success:
            return await self._execute_with_cancel(ctx)
        return await self._execute_wait_all(ctx)

    async def _execute_with_cancel(self, ctx: T) -> tuple[bool, T]:
        """Execute with cancellation: cancel remaining tasks on first success."""
        tasks = [asyncio.create_task(child._execute(ctx)) for child in self.children]

        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    try:
                        ok, _result = task.result()
                        if ok:
                            # Cancel remaining tasks and return success
                            for p in pending:
                                p.cancel()
                            if pending:
                                await asyncio.gather(*pending, return_exceptions=True)
                            return True, ctx
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        # Task raised exception, continue checking others
                        pass

            # All tasks completed without success
            return False, ctx
        except Exception:
            # Ensure all tasks are cancelled on any unexpected exception
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    async def _execute_wait_all(self, ctx: T) -> tuple[bool, T]:
        """Execute without cancellation: wait for all tasks."""
        results = await asyncio.gather(
            *(child._execute(ctx) for child in self.children), return_exceptions=True
        )
        for result in results:
            if isinstance(result, tuple):
                ok, _ = result
                if ok:
                    return True, ctx
        return False, ctx


@dataclass
class _AsyncParallelValidatingOr(AsyncValidatingCombinator[T]):
    """Async validating OR that runs children concurrently.

    Returns first success or collects all errors if none succeed.
    """

    children: list[AsyncCombinator[T]]
    cancel_on_success: bool = True

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        if self.cancel_on_success:
            return await self._execute_with_cancel(ctx)
        return await self._execute_wait_all(ctx)

    async def _execute_with_cancel(self, ctx: T) -> tuple[bool, T]:
        """Execute with cancellation on first success."""
        tasks = [asyncio.create_task(child._execute(ctx)) for child in self.children]

        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    try:
                        ok, _ = task.result()
                        if ok:
                            for p in pending:
                                p.cancel()
                            if pending:
                                await asyncio.gather(*pending, return_exceptions=True)
                            return True, ctx
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

            return False, ctx
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    async def _execute_wait_all(self, ctx: T) -> tuple[bool, T]:
        results = await asyncio.gather(
            *(child._execute(ctx) for child in self.children), return_exceptions=True
        )
        for result in results:
            if isinstance(result, tuple):
                ok, _ = result
                if ok:
                    return True, ctx
        return False, ctx

    async def validate(self, ctx: T) -> ValidationResult:
        """Validate - passes if any succeeds, collects all errors if none do."""

        async def _validate_one(child: AsyncCombinator[T]) -> tuple[bool, list[str]]:
            if isinstance(child, AsyncValidatingCombinator):
                result = await child.validate(ctx)
                return result.ok, result.errors
            else:
                ok, _ = await child._execute(ctx)
                if ok:
                    return True, []
                return False, [f"Check failed: {_get_async_combinator_name(child)}"]

        results = await asyncio.gather(*(_validate_one(child) for child in self.children))

        # If any succeeded, return success
        for ok, _ in results:
            if ok:
                return ValidationResult(ok=True, errors=[], ctx=ctx)

        # All failed - collect all errors
        all_errors: list[str] = []
        for _, errors in results:
            all_errors.extend(errors)
        return ValidationResult(ok=False, errors=all_errors, ctx=ctx)


def parallel_or(
    *combinators: AsyncCombinator[T],
    cancel_on_success: bool = True,
) -> AsyncCombinator[T]:
    """Create an async OR that runs all children concurrently, returns first success.

    Unlike ``|`` which chains sequentially (trying each in order until one succeeds),
    ``parallel_or`` runs every child concurrently and returns as soon as one succeeds.

    All children receive the **same original context** (not chained).
    Returns ``(True, original_ctx)`` on first success, or ``(False, original_ctx)``
    if all fail.

    If all inputs are ``AsyncValidatingCombinator``, returns a validating
    variant that collects all errors if none succeed.

    Args:
        *combinators: Combinators to run in parallel
        cancel_on_success: If True (default), cancel remaining tasks when first
                          success is detected. Set to False to let all tasks complete.

    Raises ``ValueError`` when called with no arguments.

    Example::

        # Try multiple data sources concurrently, use first success
        result = await parallel_or(cache_lookup, db_lookup, api_lookup).run(ctx)

        # Wait for all to complete (useful for side effects)
        result = await parallel_or(source_a, source_b, cancel_on_success=False).run(ctx)
    """
    if not combinators:
        raise ValueError("parallel_or() requires at least one combinator")

    children = list(combinators)
    if all(isinstance(c, AsyncValidatingCombinator) for c in children):
        return _AsyncParallelValidatingOr(children, cancel_on_success=cancel_on_success)
    return _AsyncParallelOr(children, cancel_on_success=cancel_on_success)
