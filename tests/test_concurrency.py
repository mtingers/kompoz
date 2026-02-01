"""Tests for concurrency utilities: timeout, limiter, circuit breaker, and parallel_or."""

from __future__ import annotations

import asyncio
import time

import pytest

from kompoz import (
    CircuitBreakerStats,
    CircuitState,
    async_rule,
    async_vrule,
    circuit_breaker,
    limited,
    parallel_and,
    parallel_or,
    with_timeout,
)

# =============================================================================
# Timeout Tests
# =============================================================================


class TestAsyncTimeout:
    def test_completes_within_timeout(self):
        @async_rule
        async def fast_check(ctx):
            await asyncio.sleep(0.01)
            return True

        result = with_timeout(fast_check, timeout=1.0)
        ok, out = asyncio.run(result.run(42))
        assert ok is True
        assert out == 42

    def test_times_out(self):
        @async_rule
        async def slow_check(ctx):
            await asyncio.sleep(1.0)
            return True

        result = with_timeout(slow_check, timeout=0.05)
        ok, out = asyncio.run(result.run(42))
        assert ok is False
        assert out == 42
        assert result.timed_out is True

    def test_on_timeout_callback(self):
        @async_rule
        async def slow_check(ctx):
            await asyncio.sleep(1.0)
            return True

        def handle_timeout(ctx):
            return ctx * 2

        result = with_timeout(slow_check, timeout=0.05, on_timeout=handle_timeout)
        ok, out = asyncio.run(result.run(21))
        assert ok is False
        assert out == 42  # Doubled by callback

    def test_timeout_repr(self):
        @async_rule
        async def check(ctx):
            return True

        result = with_timeout(check, timeout=5.0)
        assert "AsyncTimeout" in repr(result)
        assert "5.0" in repr(result)


# =============================================================================
# Limited (Semaphore) Tests
# =============================================================================


class TestAsyncLimited:
    def test_basic_execution(self):
        @async_rule
        async def check(ctx):
            return ctx > 0

        result = limited(check, max_concurrent=5)
        ok, out = asyncio.run(result.run(42))
        assert ok is True
        assert out == 42

    def test_limits_concurrency(self):
        """Verify that concurrency is actually limited."""
        concurrent_count = 0
        max_concurrent_seen = 0

        @async_rule
        async def track_concurrency(ctx):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return True

        limiter = limited(track_concurrency, max_concurrent=3)

        async def run_many():
            tasks = [limiter.run(i) for i in range(10)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_many())
        assert all(ok for ok, _ in results)
        assert max_concurrent_seen <= 3

    def test_named_shared_semaphore(self):
        """Named limiters share the same semaphore."""
        concurrent_count = 0
        max_concurrent_seen = 0

        @async_rule
        async def check_a(ctx):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return True

        @async_rule
        async def check_b(ctx):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return True

        limiter_a = limited(check_a, max_concurrent=2, name="shared_pool")
        limiter_b = limited(check_b, max_concurrent=2, name="shared_pool")

        async def run_both():
            tasks_a = [limiter_a.run(i) for i in range(5)]
            tasks_b = [limiter_b.run(i) for i in range(5)]
            return await asyncio.gather(*tasks_a, *tasks_b)

        results = asyncio.run(run_both())
        assert all(ok for ok, _ in results)
        # Both limiters share the same 2-slot semaphore
        assert max_concurrent_seen <= 2

    def test_repr(self):
        @async_rule
        async def check(ctx):
            return True

        result = limited(check, max_concurrent=5)
        assert "AsyncLimited" in repr(result)
        assert "5" in repr(result)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestAsyncCircuitBreaker:
    def test_closed_state_passes_through(self):
        @async_rule
        async def check(ctx):
            return ctx > 0

        cb = circuit_breaker(check, failure_threshold=3)
        ok, _out = asyncio.run(cb.run(42))
        assert ok is True
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        call_count = 0

        @async_rule
        async def failing_check(ctx):
            nonlocal call_count
            call_count += 1
            return False

        cb = circuit_breaker(failing_check, failure_threshold=3, recovery_timeout=10.0)

        async def run_failures():
            for _ in range(5):
                await cb.run(1)
            return cb.state

        state = asyncio.run(run_failures())
        assert state == CircuitState.OPEN
        # After opening, calls should be rejected without executing
        assert call_count == 3  # Only first 3 executed

    def test_open_circuit_rejects_immediately(self):
        call_count = 0

        @async_rule
        async def failing_check(ctx):
            nonlocal call_count
            call_count += 1
            return False

        cb = circuit_breaker(failing_check, failure_threshold=2, recovery_timeout=100.0)

        async def test():
            # Trigger failures to open circuit
            await cb.run(1)
            await cb.run(1)
            assert cb.state == CircuitState.OPEN

            # These should be rejected without calling the function
            initial_count = call_count
            for _ in range(5):
                ok, _ = await cb.run(1)
                assert ok is False

            # No additional calls made
            assert call_count == initial_count

        asyncio.run(test())

    def test_half_open_after_recovery_timeout(self):
        @async_rule
        async def failing_check(ctx):
            return False

        cb = circuit_breaker(failing_check, failure_threshold=2, recovery_timeout=0.05)

        async def test():
            # Open the circuit
            await cb.run(1)
            await cb.run(1)
            assert cb.state == CircuitState.OPEN

            # Wait for recovery timeout
            await asyncio.sleep(0.1)

            # Next call should transition to half-open
            await cb.run(1)
            # Since it failed, it goes back to open
            assert cb.state == CircuitState.OPEN

        asyncio.run(test())

    def test_closes_after_success_in_half_open(self):
        fail_count = 0

        @async_rule
        async def sometimes_fail(ctx):
            nonlocal fail_count
            if fail_count < 2:
                fail_count += 1
                return False
            return True

        cb = circuit_breaker(
            sometimes_fail,
            failure_threshold=2,
            recovery_timeout=0.05,
            half_open_max_calls=1,
        )

        async def test():
            # Open the circuit
            await cb.run(1)
            await cb.run(1)
            assert cb.state == CircuitState.OPEN

            # Wait for recovery timeout
            await asyncio.sleep(0.1)

            # Next call should succeed and close circuit
            ok, _ = await cb.run(1)
            assert ok is True
            assert cb.state == CircuitState.CLOSED

        asyncio.run(test())

    def test_state_change_callback(self):
        state_changes = []

        def on_change(old_state, new_state, stats):
            state_changes.append((old_state, new_state))

        @async_rule
        async def failing_check(ctx):
            return False

        cb = circuit_breaker(
            failing_check, failure_threshold=2, on_state_change=on_change
        )

        async def test():
            await cb.run(1)
            await cb.run(1)

        asyncio.run(test())
        assert (CircuitState.CLOSED, CircuitState.OPEN) in state_changes

    def test_get_stats(self):
        @async_rule
        async def check(ctx):
            return ctx > 0

        cb = circuit_breaker(check, failure_threshold=5)

        asyncio.run(cb.run(1))
        asyncio.run(cb.run(-1))

        stats = cb.get_stats()
        assert isinstance(stats, CircuitBreakerStats)
        assert stats.state == CircuitState.CLOSED
        assert stats.success_count >= 1
        assert stats.failure_count >= 1

    def test_manual_reset(self):
        @async_rule
        async def failing_check(ctx):
            return False

        cb = circuit_breaker(failing_check, failure_threshold=2)

        async def test():
            # Open the circuit
            await cb.run(1)
            await cb.run(1)
            assert cb.state == CircuitState.OPEN

            # Manual reset
            await cb.reset()
            assert cb.state == CircuitState.CLOSED

        asyncio.run(test())


# =============================================================================
# Parallel OR Tests
# =============================================================================


class TestParallelOr:
    def test_first_success_wins(self):
        @async_rule
        async def fail_check(ctx):
            return False

        @async_rule
        async def pass_check(ctx):
            return True

        result = parallel_or(fail_check, pass_check, fail_check)
        ok, out = asyncio.run(result.run(42))
        assert ok is True
        assert out == 42

    def test_all_fail(self):
        @async_rule
        async def fail_a(ctx):
            return False

        @async_rule
        async def fail_b(ctx):
            return False

        result = parallel_or(fail_a, fail_b)
        ok, out = asyncio.run(result.run(42))
        assert ok is False
        assert out == 42

    def test_concurrent_execution(self):
        """Children should run concurrently."""

        @async_rule
        async def slow_fail(ctx):
            await asyncio.sleep(0.1)
            return False

        @async_rule
        async def slow_pass(ctx):
            await asyncio.sleep(0.1)
            return True

        combo = parallel_or(slow_fail, slow_pass, slow_fail)
        start = time.perf_counter()
        ok, _ = asyncio.run(combo.run(1))
        elapsed = time.perf_counter() - start

        assert ok is True
        # Should complete in ~0.1s (concurrent), not ~0.3s (sequential)
        assert elapsed < 1.0

    def test_cancels_on_success(self):
        """With cancel_on_success=True, remaining tasks should be cancelled."""
        completed = []

        @async_rule
        async def slow_check(ctx):
            await asyncio.sleep(0.2)
            completed.append("slow")
            return True

        @async_rule
        async def fast_pass(ctx):
            await asyncio.sleep(0.01)
            completed.append("fast")
            return True

        combo = parallel_or(slow_check, fast_pass, cancel_on_success=True)
        asyncio.run(combo.run(1))

        # Give time for slow task to complete if it wasn't cancelled
        asyncio.run(asyncio.sleep(0.3))

        assert "fast" in completed
        # Slow task should have been cancelled
        assert "slow" not in completed

    def test_no_cancel_on_success(self):
        """With cancel_on_success=False, all tasks should complete."""
        completed = []

        @async_rule
        async def slow_check(ctx):
            await asyncio.sleep(0.1)
            completed.append("slow")
            return False

        @async_rule
        async def fast_pass(ctx):
            completed.append("fast")
            return True

        combo = parallel_or(slow_check, fast_pass, cancel_on_success=False)
        ok, _ = asyncio.run(combo.run(1))

        # Wait for slow task
        asyncio.run(asyncio.sleep(0.2))

        assert ok is True
        assert "fast" in completed
        assert "slow" in completed

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            parallel_or()

    def test_single_combinator(self):
        @async_rule
        async def check(ctx):
            return ctx > 0

        ok, out = asyncio.run(parallel_or(check).run(5))
        assert ok is True
        assert out == 5


class TestParallelValidatingOr:
    def test_returns_validating_variant(self):
        from kompoz import AsyncValidatingCombinator

        @async_vrule(error="must be positive")
        async def check_a(ctx):
            return ctx > 0

        @async_vrule(error="must be < 100")
        async def check_b(ctx):
            return ctx < 100

        result = parallel_or(check_a, check_b)
        assert isinstance(result, AsyncValidatingCombinator)

    def test_success_no_errors(self):
        @async_vrule(error="fail a")
        async def check_a(ctx):
            return False

        @async_vrule(error="fail b")
        async def check_b(ctx):
            return True

        combo = parallel_or(check_a, check_b)
        vr = asyncio.run(combo.validate(1))  # type: ignore[reportAttributeAccessIssue]
        assert vr.ok is True
        assert vr.errors == []

    def test_all_fail_collects_errors(self):
        @async_vrule(error="must be positive")
        async def check_pos(ctx):
            return ctx > 0

        @async_vrule(error="must be even")
        async def check_even(ctx):
            return ctx % 2 == 0

        combo = parallel_or(check_pos, check_even)
        vr = asyncio.run(combo.validate(-3))  # type: ignore[reportAttributeAccessIssue]
        assert vr.ok is False
        assert "must be positive" in vr.errors
        assert "must be even" in vr.errors


# =============================================================================
# Parallel AND with fail_fast Tests
# =============================================================================


class TestParallelAndFailFast:
    def test_fail_fast_cancels_remaining(self):
        """With fail_fast=True, remaining tasks should be cancelled on first failure."""
        completed = []

        @async_rule
        async def slow_pass(ctx):
            await asyncio.sleep(0.2)
            completed.append("slow")
            return True

        @async_rule
        async def fast_fail(ctx):
            await asyncio.sleep(0.01)
            completed.append("fast")
            return False

        combo = parallel_and(slow_pass, fast_fail, fail_fast=True)
        ok, _ = asyncio.run(combo.run(1))

        # Give time for slow task to complete if it wasn't cancelled
        asyncio.run(asyncio.sleep(0.3))

        assert ok is False
        assert "fast" in completed
        # Slow task should have been cancelled
        assert "slow" not in completed

    def test_fail_fast_all_pass(self):
        """When all pass with fail_fast, should still succeed."""

        @async_rule
        async def check_a(ctx):
            await asyncio.sleep(0.01)
            return True

        @async_rule
        async def check_b(ctx):
            await asyncio.sleep(0.01)
            return True

        combo = parallel_and(check_a, check_b, fail_fast=True)
        ok, out = asyncio.run(combo.run(42))
        assert ok is True
        assert out == 42

    def test_without_fail_fast_waits_all(self):
        """Without fail_fast, all tasks complete even if one fails early."""
        completed = []

        @async_rule
        async def slow_pass(ctx):
            await asyncio.sleep(0.1)
            completed.append("slow")
            return True

        @async_rule
        async def fast_fail(ctx):
            completed.append("fast")
            return False

        combo = parallel_and(slow_pass, fast_fail, fail_fast=False)
        ok, _ = asyncio.run(combo.run(1))

        assert ok is False
        assert "fast" in completed
        assert "slow" in completed


# =============================================================================
# Integration Tests
# =============================================================================


class TestConcurrencyIntegration:
    def test_timeout_with_retry(self):
        """Combine timeout with circuit breaker."""

        call_count = 0

        @async_rule
        async def flaky(ctx):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                await asyncio.sleep(0.5)  # Will timeout
            return True

        protected = circuit_breaker(
            with_timeout(flaky, timeout=0.1), failure_threshold=5
        )

        async def test():
            # First two calls timeout, third succeeds
            for _ in range(3):
                await protected.run(1)
            return protected.state

        state = asyncio.run(test())
        # Should still be closed since we didn't hit threshold
        assert state == CircuitState.CLOSED

    def test_limited_with_parallel_and(self):
        """Combine limiter with parallel execution."""
        concurrent_count = 0
        max_concurrent_seen = 0

        @async_rule
        async def track(ctx):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return True

        # Each check is limited to 2 concurrent
        check_a = limited(track, max_concurrent=2, name="pool")
        check_b = limited(track, max_concurrent=2, name="pool")
        check_c = limited(track, max_concurrent=2, name="pool")

        # Run them all in parallel
        combo = parallel_and(check_a, check_b, check_c)

        async def run_many():
            tasks = [combo.run(i) for i in range(5)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_many())
        assert all(ok for ok, _ in results)
        assert max_concurrent_seen <= 2
