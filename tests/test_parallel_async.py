"""Tests for parallel_and: concurrent async AND with same-context semantics."""

from __future__ import annotations

import asyncio
import time

import pytest

from kompoz import (
    AsyncValidatingCombinator,
    async_rule,
    async_vrule,
    parallel_and,
)


class TestParallelAnd:
    def test_all_pass(self):
        @async_rule
        async def check_a(ctx):
            return ctx > 0

        @async_rule
        async def check_b(ctx):
            return ctx < 100

        result = parallel_and(check_a, check_b)
        ok, out = asyncio.run(result.run(50))
        assert ok is True
        assert out == 50  # original context returned

    def test_one_fails(self):
        @async_rule
        async def always_pass(ctx):
            return True

        @async_rule
        async def always_fail(ctx):
            return False

        result = parallel_and(always_pass, always_fail)
        ok, out = asyncio.run(result.run(42))
        assert ok is False
        assert out == 42  # original context returned even on failure

    def test_concurrent_execution(self):
        """Children should run concurrently, not sequentially."""

        @async_rule
        async def slow_a(ctx):
            await asyncio.sleep(0.1)
            return True

        @async_rule
        async def slow_b(ctx):
            await asyncio.sleep(0.1)
            return True

        @async_rule
        async def slow_c(ctx):
            await asyncio.sleep(0.1)
            return True

        combo = parallel_and(slow_a, slow_b, slow_c)
        start = time.perf_counter()
        ok, _ = asyncio.run(combo.run(1))
        elapsed = time.perf_counter() - start

        assert ok is True
        # If sequential, would take ~0.3s. Parallel should be ~0.1s.
        assert elapsed < 1.0, f"Took {elapsed:.3f}s — expected concurrent execution"

    def test_same_context_semantics(self):
        """All children receive the same original context, not chained."""
        calls = []

        @async_rule
        async def record_a(ctx):
            calls.append(("a", ctx))
            return True

        @async_rule
        async def record_b(ctx):
            calls.append(("b", ctx))
            return True

        asyncio.run(parallel_and(record_a, record_b).run(99))
        # Both received the original context
        assert all(ctx == 99 for _, ctx in calls)

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            parallel_and()

    def test_single_combinator(self):
        @async_rule
        async def check(ctx):
            return ctx > 0

        ok, out = asyncio.run(parallel_and(check).run(5))
        assert ok is True
        assert out == 5

    def test_original_context_returned_not_child_context(self):
        """Even if a child would modify context (e.g. a transform), parallel_and
        returns the original context."""
        from kompoz import async_pipe

        @async_pipe
        async def double(ctx):
            return ctx * 2

        @async_rule
        async def is_positive(ctx):
            return ctx > 0

        ok, out = asyncio.run(parallel_and(double, is_positive).run(5))
        assert ok is True
        assert out == 5  # original, not doubled


class TestParallelValidatingAnd:
    def test_returns_validating_variant(self):
        @async_vrule(error="must be positive")
        async def check_a(ctx):
            return ctx > 0

        @async_vrule(error="must be < 100")
        async def check_b(ctx):
            return ctx < 100

        result = parallel_and(check_a, check_b)
        assert isinstance(result, AsyncValidatingCombinator)

    def test_collects_all_errors(self):
        @async_vrule(error="must be positive")
        async def check_pos(ctx):
            return ctx > 0

        @async_vrule(error="must be even")
        async def check_even(ctx):
            return ctx % 2 == 0

        combo = parallel_and(check_pos, check_even)
        vr = asyncio.run(combo.validate(-3))  # type: ignore[reportAttributeAccessIssue]
        assert vr.ok is False
        assert "must be positive" in vr.errors
        assert "must be even" in vr.errors

    def test_all_pass_no_errors(self):
        @async_vrule(error="fail a")
        async def check_a(ctx):
            return True

        @async_vrule(error="fail b")
        async def check_b(ctx):
            return True

        combo = parallel_and(check_a, check_b)
        vr = asyncio.run(combo.validate(1))  # type: ignore[reportAttributeAccessIssue]
        assert vr.ok is True
        assert vr.errors == []

    def test_mixed_validating_and_regular_returns_base(self):
        """If not all inputs are AsyncValidatingCombinator, returns base variant."""

        @async_vrule(error="fail")
        async def validating_check(ctx):
            return True

        @async_rule
        async def regular_check(ctx):
            return True

        result = parallel_and(validating_check, regular_check)
        # Should still work but not be an AsyncValidatingCombinator
        assert not isinstance(result, AsyncValidatingCombinator)
        ok, _ = asyncio.run(result.run(1))
        assert ok is True

    def test_validating_concurrent_timing(self):
        """Validation should also run concurrently."""

        @async_vrule(error="slow a failed")
        async def slow_a(ctx):
            await asyncio.sleep(0.1)
            return False

        @async_vrule(error="slow b failed")
        async def slow_b(ctx):
            await asyncio.sleep(0.1)
            return False

        combo = parallel_and(slow_a, slow_b)
        start = time.perf_counter()
        vr = asyncio.run(combo.validate(1))  # type: ignore[reportAttributeAccessIssue]
        elapsed = time.perf_counter() - start

        assert vr.ok is False
        assert len(vr.errors) == 2
        assert elapsed < 1.0, f"Took {elapsed:.3f}s — expected concurrent validation"
