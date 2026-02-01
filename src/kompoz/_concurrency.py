"""Concurrency utilities: timeout, rate limiting, circuit breaker."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, ClassVar, Generic

from kompoz._async import AsyncCombinator
from kompoz._types import T

# =============================================================================
# Timeout
# =============================================================================


@dataclass
class AsyncTimeout(AsyncCombinator[T]):
    """
    Wrap an async combinator with a timeout.

    If the inner combinator doesn't complete within the timeout, returns
    (False, ctx) or calls the on_timeout handler if provided.

    Example:
        # Basic timeout
        result = await with_timeout(slow_api_check, timeout=5.0).run(ctx)

        # With custom timeout handler
        def handle_timeout(ctx):
            ctx.timed_out = True
            return ctx

        result = await with_timeout(
            slow_check,
            timeout=5.0,
            on_timeout=handle_timeout
        ).run(ctx)

    Attributes:
        inner: The combinator to wrap
        timeout: Timeout in seconds
        on_timeout: Optional callback to modify context on timeout
        timed_out: Whether the last execution timed out (not concurrency-safe,
                   use for debugging only)
    """

    inner: AsyncCombinator[T]
    timeout: float
    on_timeout: Callable[[T], T] | None = None
    timed_out: bool = field(default=False, init=False, repr=False)

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        try:
            result = await asyncio.wait_for(self.inner._execute(ctx), timeout=self.timeout)
            self.timed_out = False
            return result
        except asyncio.TimeoutError:
            self.timed_out = True
            if self.on_timeout:
                return False, self.on_timeout(ctx)
            return False, ctx

    def __repr__(self) -> str:
        return f"AsyncTimeout({self.inner!r}, timeout={self.timeout})"


def with_timeout(
    combinator: AsyncCombinator[T],
    timeout: float,
    on_timeout: Callable[[T], T] | None = None,
) -> AsyncTimeout[T]:
    """
    Wrap an async combinator with a timeout.

    Args:
        combinator: The combinator to wrap
        timeout: Timeout in seconds
        on_timeout: Optional callback(ctx) -> ctx called on timeout

    Returns:
        AsyncTimeout combinator

    Example:
        # Timeout after 5 seconds
        result = await with_timeout(slow_api, timeout=5.0).run(ctx)

        # With timeout handler
        result = await with_timeout(
            slow_api,
            timeout=5.0,
            on_timeout=lambda ctx: ctx._replace(error="timeout")
        ).run(ctx)
    """
    return AsyncTimeout(inner=combinator, timeout=timeout, on_timeout=on_timeout)


# =============================================================================
# Concurrency Limiter (Semaphore)
# =============================================================================


class AsyncLimited(AsyncCombinator[T], Generic[T]):
    """
    Limit concurrent executions of a combinator using a semaphore.

    Useful for rate-limiting API calls or database connections.

    Example:
        # Max 5 concurrent API calls
        limited_api = limited(api_check, max_concurrent=5)

        # Run many tasks, but only 5 at a time
        results = await asyncio.gather(*[
            limited_api.run(ctx) for ctx in many_contexts
        ])

    Attributes:
        inner: The combinator to wrap
        max_concurrent: Maximum number of concurrent executions
        name: Optional name for shared semaphores
    """

    # Class-level semaphore registry for named/shared semaphores
    _named_semaphores: ClassVar[dict[str, asyncio.Semaphore]] = {}
    _registry_lock: ClassVar[asyncio.Lock | None] = None

    def __init__(
        self,
        inner: AsyncCombinator[T],
        max_concurrent: int,
        name: str | None = None,
    ):
        self.inner = inner
        self.max_concurrent = max_concurrent
        self.name = name
        self._instance_semaphore: asyncio.Semaphore | None = None

    @classmethod
    async def _get_registry_lock(cls) -> asyncio.Lock:
        """Get or create the registry lock."""
        if cls._registry_lock is None:
            cls._registry_lock = asyncio.Lock()
        return cls._registry_lock

    async def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the semaphore for this limiter."""
        if self.name is not None:
            # Named/shared semaphore
            lock = await self._get_registry_lock()
            async with lock:
                if self.name not in self._named_semaphores:
                    self._named_semaphores[self.name] = asyncio.Semaphore(self.max_concurrent)
                return self._named_semaphores[self.name]
        else:
            # Instance-specific semaphore
            if self._instance_semaphore is None:
                self._instance_semaphore = asyncio.Semaphore(self.max_concurrent)
            return self._instance_semaphore

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        semaphore = await self._get_semaphore()
        async with semaphore:
            return await self.inner._execute(ctx)

    def __repr__(self) -> str:
        name_str = f", name={self.name!r}" if self.name else ""
        return f"AsyncLimited({self.inner!r}, max_concurrent={self.max_concurrent}{name_str})"


def limited(
    combinator: AsyncCombinator[T],
    max_concurrent: int,
    name: str | None = None,
) -> AsyncLimited[T]:
    """
    Limit concurrent executions of a combinator.

    Args:
        combinator: The combinator to wrap
        max_concurrent: Maximum number of concurrent executions
        name: Optional name to share semaphore across multiple limiters.
              Limiters with the same name share the same semaphore.

    Returns:
        AsyncLimited combinator

    Example:
        # Instance-specific limit
        limited_api = limited(api_check, max_concurrent=5)

        # Shared limit across multiple combinators
        check_a = limited(api_check_a, max_concurrent=10, name="api_pool")
        check_b = limited(api_check_b, max_concurrent=10, name="api_pool")
        # check_a and check_b share the same 10-slot semaphore
    """
    return AsyncLimited(inner=combinator, max_concurrent=max_concurrent, name=name)


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit tripped, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_success_time: float | None
    half_open_successes: int


class AsyncCircuitBreaker(AsyncCombinator[T], Generic[T]):
    """
    Circuit breaker pattern for fault tolerance.

    The circuit breaker monitors failures and "trips" (opens) when failures
    exceed a threshold, preventing cascading failures by rejecting requests
    immediately without executing the inner combinator.

    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Too many failures, requests rejected immediately
        - HALF_OPEN: Testing if service recovered

    Example:
        # Basic circuit breaker
        protected = circuit_breaker(
            flaky_api,
            failure_threshold=5,    # Open after 5 failures
            recovery_timeout=30.0,  # Try again after 30 seconds
        )

        # With callbacks
        def on_state_change(old_state, new_state, stats):
            print(f"Circuit {old_state} -> {new_state}")

        protected = circuit_breaker(
            flaky_api,
            failure_threshold=5,
            recovery_timeout=30.0,
            on_state_change=on_state_change,
        )

    Attributes:
        inner: The combinator to protect
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        half_open_max_calls: Max calls allowed in half-open state
        on_state_change: Optional callback(old_state, new_state, stats)
    """

    def __init__(
        self,
        inner: AsyncCombinator[T],
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        on_state_change: Callable[[CircuitState, CircuitState, CircuitBreakerStats], Any]
        | None = None,
    ):
        self.inner = inner
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._last_success_time: float | None = None
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics."""
        return CircuitBreakerStats(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            half_open_successes=self._half_open_successes,
        )

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return monotonic() - self._last_failure_time >= self.recovery_timeout

    async def _change_state(self, new_state: CircuitState) -> None:
        """Change state and notify callback if provided."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state

            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._half_open_successes = 0
            elif new_state == CircuitState.CLOSED:
                self._failure_count = 0

            if self.on_state_change:
                result = self.on_state_change(old_state, new_state, self.get_stats())
                if asyncio.iscoroutine(result):
                    await result

    async def _record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            self._success_count += 1
            self._last_success_time = monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_max_calls:
                    await self._change_state(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def _record_failure(self) -> None:
        """Record a failed execution."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens
                await self._change_state(CircuitState.OPEN)
            elif (
                self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold
            ):
                await self._change_state(CircuitState.OPEN)

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        # Check and potentially transition state
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._change_state(CircuitState.HALF_OPEN)
                else:
                    # Circuit is open, reject immediately
                    return False, ctx

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    # Already at max half-open calls, reject
                    return False, ctx
                self._half_open_calls += 1

        # Execute the inner combinator
        try:
            ok, result = await self.inner._execute(ctx)
            if ok:
                await self._record_success()
            else:
                await self._record_failure()
            return ok, result
        except Exception:
            await self._record_failure()
            raise

    async def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            await self._change_state(CircuitState.CLOSED)
            self._failure_count = 0
            self._half_open_calls = 0
            self._half_open_successes = 0

    def __repr__(self) -> str:
        return (
            f"AsyncCircuitBreaker({self.inner!r}, "
            f"failure_threshold={self.failure_threshold}, "
            f"recovery_timeout={self.recovery_timeout})"
        )


def circuit_breaker(
    combinator: AsyncCombinator[T],
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 1,
    on_state_change: Callable[[CircuitState, CircuitState, CircuitBreakerStats], Any] | None = None,
) -> AsyncCircuitBreaker[T]:
    """
    Wrap an async combinator with circuit breaker protection.

    Args:
        combinator: The combinator to protect
        failure_threshold: Number of consecutive failures before opening circuit
        recovery_timeout: Seconds to wait in open state before trying half-open
        half_open_max_calls: Number of test calls allowed in half-open state
        on_state_change: Optional callback(old_state, new_state, stats) on transitions

    Returns:
        AsyncCircuitBreaker combinator

    Example:
        # Protect an API call
        protected_api = circuit_breaker(
            api_call,
            failure_threshold=5,
            recovery_timeout=30.0,
        )

        # Check circuit state
        if protected_api.state == CircuitState.OPEN:
            print("Circuit is open, service likely down")

        # Get detailed stats
        stats = protected_api.get_stats()
        print(f"Failures: {stats.failure_count}")
    """
    return AsyncCircuitBreaker(
        inner=combinator,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_max_calls=half_open_max_calls,
        on_state_change=on_state_change,
    )
