"""Retry combinators with configurable backoff."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic

from kompoz._async import AsyncCombinator, AsyncTransform
from kompoz._core import Combinator
from kompoz._transform import Transform
from kompoz._types import T

# =============================================================================
# Retry Result
# =============================================================================


@dataclass
class RetryResult(Generic[T]):
    """
    Result of a retry operation with detailed execution information.

    Attributes:
        ok: Whether the operation ultimately succeeded
        ctx: The final context after execution
        attempts_made: Number of attempts made (1 = succeeded first try)
        last_error: The last exception encountered, if any
    """

    ok: bool
    ctx: T
    attempts_made: int
    last_error: Exception | None


# =============================================================================
# Retry Logic
# =============================================================================


class Retry(Combinator[T]):
    """
    A combinator that retries on failure with configurable backoff.

    Thread-safe: execution state is local to each run() call, not stored
    in instance variables. Use run_with_info() to get detailed retry metadata.

    Example:
        # Retry up to 3 times with exponential backoff
        fetch = Retry(fetch_from_api, max_attempts=3, backoff=1.0, exponential=True)

        # Retry with jitter to avoid thundering herd
        fetch = Retry(fetch_from_api, max_attempts=5, backoff=0.5, jitter=0.1)

        # With observability hook
        def on_retry(attempt, error, delay):
            print(f"Retry {attempt}: {error}, waiting {delay}s")

        fetch = Retry(fetch_from_api, max_attempts=3, on_retry=on_retry)

        # Get detailed retry information
        result = fetch.run_with_info(ctx)
        print(f"Took {result.attempts_made} attempts")

    Args:
        inner: The combinator or callable to retry
        max_attempts: Maximum number of attempts (default: 3)
        backoff: Base delay between retries in seconds (default: 0)
        exponential: Use exponential backoff (default: False)
        jitter: Random jitter to add to delay (default: 0)
        name: Optional name for debugging
        on_retry: Optional callback(attempt, error, delay) called before each retry
    """

    def __init__(
        self,
        inner: Combinator[T] | Callable[[T], T],
        max_attempts: int = 3,
        backoff: float = 0.0,
        exponential: bool = False,
        jitter: float = 0.0,
        name: str | None = None,
        on_retry: Callable[[int, Exception | None, float], None] | None = None,
    ):
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if backoff < 0:
            raise ValueError(f"backoff must be >= 0, got {backoff}")
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")

        if isinstance(inner, Combinator):
            self.inner = inner
            self.name = name or repr(inner)
        else:
            self.inner = Transform(inner, getattr(inner, "__name__", "retry_fn"))
            self.name = name or getattr(inner, "__name__", "retry")

        self.max_attempts = max_attempts
        self.backoff = backoff
        self.exponential = exponential
        self.jitter = jitter
        self.on_retry = on_retry

        # Deprecated: these are kept for backwards compatibility but are
        # updated after each run. For concurrent usage, use run_with_info().
        self.last_error: Exception | None = None
        self.attempts_made: int = 0

    def _get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff <= 0:
            return 0

        delay = self.backoff * (2**attempt) if self.exponential else self.backoff

        if self.jitter > 0:
            delay += random.uniform(0, self.jitter)

        return delay

    def run_with_info(self, ctx: T) -> RetryResult[T]:
        """
        Execute with detailed retry information.

        This method is thread-safe and returns all execution metadata
        in the result object rather than storing it in instance variables.

        Returns:
            RetryResult with ok, ctx, attempts_made, and last_error
        """
        last_ctx = ctx
        last_error: Exception | None = None
        attempts_made = 0

        for attempt in range(self.max_attempts):
            attempts_made = attempt + 1
            try:
                ok, result = self.inner._execute(last_ctx)
                if ok:
                    return RetryResult(
                        ok=True,
                        ctx=result,
                        attempts_made=attempts_made,
                        last_error=None,
                    )
                last_ctx = result
                last_error = None
            except Exception as e:
                last_error = e
                # Continue to retry on exception

            # Don't sleep after last attempt
            if attempt < self.max_attempts - 1:
                delay = self._get_delay(attempt)

                # Call the retry hook if provided
                if self.on_retry is not None:
                    self.on_retry(attempt + 1, last_error, delay)

                if delay > 0:
                    time.sleep(delay)

        return RetryResult(
            ok=False, ctx=last_ctx, attempts_made=attempts_made, last_error=last_error
        )

    def _execute(self, ctx: T) -> tuple[bool, T]:
        result = self.run_with_info(ctx)
        # Update instance vars for backwards compatibility (not thread-safe)
        self.last_error = result.last_error
        self.attempts_made = result.attempts_made
        return result.ok, result.ctx

    def __repr__(self) -> str:
        return f"Retry({self.name}, max_attempts={self.max_attempts})"


class AsyncRetry(AsyncCombinator[T]):
    """
    Async version of Retry combinator.

    Concurrency-safe: execution state is local to each run() call, not stored
    in instance variables. Use run_with_info() to get detailed retry metadata.

    Example:
        fetch = AsyncRetry(fetch_from_api, max_attempts=3, backoff=1.0)
        ok, result = await fetch.run(request)

        # With observability hook
        async def on_retry(attempt, error, delay):
            print(f"Retry {attempt}: {error}, waiting {delay}s")

        fetch = AsyncRetry(fetch_from_api, max_attempts=3, on_retry=on_retry)

        # Get detailed retry information
        result = await fetch.run_with_info(ctx)
        print(f"Took {result.attempts_made} attempts")

    Args:
        inner: The async combinator or callable to retry
        max_attempts: Maximum number of attempts (default: 3)
        backoff: Base delay between retries in seconds (default: 0)
        exponential: Use exponential backoff (default: False)
        jitter: Random jitter to add to delay (default: 0)
        name: Optional name for debugging
        on_retry: Optional async callback(attempt, error, delay) called before each retry
    """

    def __init__(
        self,
        inner: AsyncCombinator[T] | Callable[[T], Any],
        max_attempts: int = 3,
        backoff: float = 0.0,
        exponential: bool = False,
        jitter: float = 0.0,
        name: str | None = None,
        on_retry: Callable[[int, Exception | None, float], Any] | None = None,
    ):
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if backoff < 0:
            raise ValueError(f"backoff must be >= 0, got {backoff}")
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")

        if isinstance(inner, AsyncCombinator):
            self.inner = inner
            self.name = name or repr(inner)
        else:
            self.inner = AsyncTransform(inner, getattr(inner, "__name__", "retry_fn"))
            self.name = name or getattr(inner, "__name__", "retry")

        self.max_attempts = max_attempts
        self.backoff = backoff
        self.exponential = exponential
        self.jitter = jitter
        self.on_retry = on_retry

        # Deprecated: these are kept for backwards compatibility but are
        # updated after each run. For concurrent usage, use run_with_info().
        self.last_error: Exception | None = None
        self.attempts_made: int = 0

    def _get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff <= 0:
            return 0

        delay = self.backoff * (2**attempt) if self.exponential else self.backoff

        if self.jitter > 0:
            delay += random.uniform(0, self.jitter)

        return delay

    async def run_with_info(self, ctx: T) -> RetryResult[T]:
        """
        Execute with detailed retry information.

        This method is concurrency-safe and returns all execution metadata
        in the result object rather than storing it in instance variables.

        Returns:
            RetryResult with ok, ctx, attempts_made, and last_error
        """
        last_ctx = ctx
        last_error: Exception | None = None
        attempts_made = 0

        for attempt in range(self.max_attempts):
            attempts_made = attempt + 1
            try:
                ok, result = await self.inner._execute(last_ctx)
                if ok:
                    return RetryResult(
                        ok=True,
                        ctx=result,
                        attempts_made=attempts_made,
                        last_error=None,
                    )
                last_ctx = result
                last_error = None
            except Exception as e:
                last_error = e
                # Continue to retry

            # Don't sleep after last attempt
            if attempt < self.max_attempts - 1:
                delay = self._get_delay(attempt)

                # Call the retry hook if provided (supports both sync and async)
                if self.on_retry is not None:
                    result = self.on_retry(attempt + 1, last_error, delay)
                    if asyncio.iscoroutine(result):
                        await result

                if delay > 0:
                    await asyncio.sleep(delay)

        return RetryResult(
            ok=False, ctx=last_ctx, attempts_made=attempts_made, last_error=last_error
        )

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        result = await self.run_with_info(ctx)
        # Update instance vars for backwards compatibility (not concurrency-safe)
        self.last_error = result.last_error
        self.attempts_made = result.attempts_made
        return result.ok, result.ctx

    def __repr__(self) -> str:
        return f"AsyncRetry({self.name}, max_attempts={self.max_attempts})"
