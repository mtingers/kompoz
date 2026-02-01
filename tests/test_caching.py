"""Tests for caching: cached_rule, CachedPredicate, CachedPredicateFactory, use_cache,
async_cached_rule, AsyncCachedPredicate, AsyncCachedPredicateFactory."""

from __future__ import annotations

import asyncio

from kompoz import (
    AsyncCachedPredicate,
    AsyncCachedPredicateFactory,
    CachedPredicate,
    CachedPredicateFactory,
    async_cached_rule,
    async_rule,
    cached_rule,
    rule,
    use_cache,
)


class TestCachedRule:
    def test_basic_decorator(self):
        call_count = 0

        @cached_rule
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x > 0

        with use_cache():
            ok1, _ = expensive.run(5)
            ok2, _ = expensive.run(5)

        assert ok1 is True
        assert ok2 is True
        assert call_count == 1  # only executed once

    def test_without_cache_scope_runs_every_time(self):
        call_count = 0

        @cached_rule
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x > 0

        expensive.run(5)
        expensive.run(5)
        assert call_count == 2

    def test_different_contexts_different_cache_keys(self):
        call_count = 0

        @cached_rule
        def check(x):
            nonlocal call_count
            call_count += 1
            return x > 0

        with use_cache():
            check.run(5)
            check.run(10)

        assert call_count == 2

    def test_custom_key_fn(self):
        call_count = 0

        @cached_rule(key=lambda x: x % 10)
        def check(x):
            nonlocal call_count
            call_count += 1
            return x > 0

        with use_cache():
            check.run(15)
            check.run(25)  # same key (5) due to % 10

        assert call_count == 1

    def test_cache_isolation_between_scopes(self):
        call_count = 0

        @cached_rule
        def check(x):
            nonlocal call_count
            call_count += 1
            return True

        with use_cache():
            check.run(1)
            check.run(1)

        with use_cache():
            check.run(1)
            check.run(1)

        assert call_count == 2  # once per scope

    def test_repr(self):
        @cached_rule
        def my_check(x):
            return True

        assert repr(my_check) == "CachedPredicate(my_check)"

    def test_isinstance(self):
        @cached_rule
        def check(x):
            return True

        assert isinstance(check, CachedPredicate)


class TestCachedPredicateFactory:
    def test_basic(self):
        call_count = 0

        factory = CachedPredicateFactory(
            lambda x, threshold: (
                _inc_and_check(x, threshold, call_counter := {"n": 0}),
            ),
            "gt",
        )
        # Simpler approach:
        call_count = 0

        def check_fn(x, threshold):
            nonlocal call_count
            call_count += 1
            return x > threshold

        factory = CachedPredicateFactory(check_fn, "gt")
        pred = factory(10)
        assert isinstance(pred, CachedPredicate)
        assert repr(factory) == "CachedPredicateFactory(gt)"

        with use_cache():
            ok1, _ = pred.run(15)
            ok2, _ = pred.run(15)

        assert ok1 is True
        assert call_count == 1


class TestCacheWithComposition:
    def test_cached_in_and_chain(self):
        count_a = 0
        count_b = 0

        @cached_rule
        def check_a(x):
            nonlocal count_a
            count_a += 1
            return x > 0

        @cached_rule
        def check_b(x):
            nonlocal count_b
            count_b += 1
            return x < 100

        combined = check_a & check_b

        with use_cache():
            combined.run(50)
            combined.run(50)

        assert count_a == 1
        assert count_b == 1

    def test_cached_mixed_with_regular(self):
        cached_count = 0

        @cached_rule
        def cached_check(x):
            nonlocal cached_count
            cached_count += 1
            return x > 0

        @rule
        def regular_check(x):
            return x < 100

        combined = cached_check & regular_check

        with use_cache():
            ok, _ = combined.run(50)
            assert ok
            ok, _ = combined.run(50)
            assert ok

        assert cached_count == 1


# =============================================================================
# Async Caching Tests
# =============================================================================


class TestAsyncCachedRule:
    def test_basic_decorator(self):
        call_count = 0

        @async_cached_rule
        async def expensive(x):
            nonlocal call_count
            call_count += 1
            return x > 0

        with use_cache():
            ok1, _ = asyncio.run(expensive.run(5))
            ok2, _ = asyncio.run(expensive.run(5))

        assert ok1 is True
        assert ok2 is True
        assert call_count == 1

    def test_without_cache_scope_runs_every_time(self):
        call_count = 0

        @async_cached_rule
        async def expensive(x):
            nonlocal call_count
            call_count += 1
            return x > 0

        asyncio.run(expensive.run(5))
        asyncio.run(expensive.run(5))
        assert call_count == 2

    def test_different_contexts_different_cache_keys(self):
        call_count = 0

        @async_cached_rule
        async def check(x):
            nonlocal call_count
            call_count += 1
            return x > 0

        with use_cache():
            asyncio.run(check.run(5))
            asyncio.run(check.run(10))

        assert call_count == 2

    def test_custom_key_fn(self):
        call_count = 0

        @async_cached_rule(key=lambda x: x % 10)
        async def check(x):
            nonlocal call_count
            call_count += 1
            return x > 0

        with use_cache():
            asyncio.run(check.run(15))
            asyncio.run(check.run(25))  # same key (5) due to % 10

        assert call_count == 1

    def test_cache_isolation_between_scopes(self):
        call_count = 0

        @async_cached_rule
        async def check(x):
            nonlocal call_count
            call_count += 1
            return True

        with use_cache():
            asyncio.run(check.run(1))
            asyncio.run(check.run(1))

        with use_cache():
            asyncio.run(check.run(1))
            asyncio.run(check.run(1))

        assert call_count == 2  # once per scope

    def test_repr(self):
        @async_cached_rule
        async def my_check(x):
            return True

        assert repr(my_check) == "AsyncCachedPredicate(my_check)"

    def test_isinstance(self):
        @async_cached_rule
        async def check(x):
            return True

        assert isinstance(check, AsyncCachedPredicate)


class TestAsyncCachedPredicateFactory:
    def test_basic(self):
        call_count = 0

        def check_fn(x, threshold):
            nonlocal call_count
            call_count += 1
            return x > threshold

        async def async_check_fn(x, threshold):
            nonlocal call_count
            call_count += 1
            return x > threshold

        factory = AsyncCachedPredicateFactory(async_check_fn, "gt")
        pred = factory(10)
        assert isinstance(pred, AsyncCachedPredicate)
        assert repr(factory) == "AsyncCachedPredicateFactory(gt)"

        with use_cache():
            ok1, _ = asyncio.run(pred.run(15))
            ok2, _ = asyncio.run(pred.run(15))

        assert ok1 is True
        assert call_count == 1


class TestAsyncCacheWithComposition:
    def test_cached_in_and_chain(self):
        count_a = 0
        count_b = 0

        @async_cached_rule
        async def check_a(x):
            nonlocal count_a
            count_a += 1
            return x > 0

        @async_cached_rule
        async def check_b(x):
            nonlocal count_b
            count_b += 1
            return x < 100

        combined = check_a & check_b

        with use_cache():
            asyncio.run(combined.run(50))
            asyncio.run(combined.run(50))

        assert count_a == 1
        assert count_b == 1

    def test_cached_mixed_with_regular_async(self):
        cached_count = 0

        @async_cached_rule
        async def cached_check(x):
            nonlocal cached_count
            cached_count += 1
            return x > 0

        @async_rule
        async def regular_check(x):
            return x < 100

        combined = cached_check & regular_check

        with use_cache():
            ok, _ = asyncio.run(combined.run(50))
            assert ok
            ok, _ = asyncio.run(combined.run(50))
            assert ok

        assert cached_count == 1


# =============================================================================
# Concurrent Cache Access Tests
# =============================================================================


class TestConcurrentCacheAccess:
    """Tests for thread-safe and async-safe cache access."""

    def test_async_concurrent_cache_only_executes_once(self):
        """Verify expensive function runs once even with concurrent access."""
        call_count = 0

        @async_cached_rule
        async def expensive(ctx):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate slow operation
            return True

        async def run_concurrent():
            with use_cache():
                # Run 10 concurrent executions with the same context
                results = await asyncio.gather(*[
                    expensive.run(42) for _ in range(10)
                ])
                return results

        results = asyncio.run(run_concurrent())
        
        # Should only execute once due to locking
        assert call_count == 1
        assert all(ok for ok, _ in results)

    def test_async_concurrent_different_keys_execute_separately(self):
        """Different cache keys should execute separately."""
        call_count = 0

        @async_cached_rule
        async def check(ctx):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return ctx > 0

        async def run_concurrent():
            with use_cache():
                # Run with different contexts (different cache keys)
                results = await asyncio.gather(*[
                    check.run(i) for i in range(5)
                ])
                return results

        results = asyncio.run(run_concurrent())
        
        # Each unique context should execute once
        assert call_count == 5

    def test_async_concurrent_with_parallel_and(self):
        """Test cache with parallel_and combinator."""
        from kompoz import parallel_and

        check_count = 0

        @async_cached_rule
        async def expensive_check(ctx):
            nonlocal check_count
            check_count += 1
            await asyncio.sleep(0.1)
            return ctx > 0

        # parallel_and runs all children with same context
        combo = parallel_and(expensive_check, expensive_check, expensive_check)

        async def run_test():
            with use_cache():
                return await combo.run(42)

        ok, _ = asyncio.run(run_test())
        
        assert ok is True
        # All three children use same context, so cache should work
        assert check_count == 1

    def test_sync_thread_safe_cache(self):
        """Test that sync CachedPredicate is thread-safe with shared cache."""
        import threading
        from kompoz import use_cache_shared
        
        call_count = 0
        lock = threading.Lock()

        @cached_rule
        def expensive(ctx):
            nonlocal call_count
            with lock:
                call_count += 1
            import time
            time.sleep(0.05)  # Simulate slow operation
            return ctx > 0

        results = []
        errors = []

        def run_check():
            try:
                ok, _ = expensive.run(42)
                results.append(ok)
            except Exception as e:
                errors.append(e)

        # Use use_cache_shared() for thread-safe caching across threads
        # (use_cache() uses ContextVar which is per-thread)
        with use_cache_shared():
            threads = [threading.Thread(target=run_check) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(results)
        # With thread-safe locking, should only execute once
        assert call_count == 1
