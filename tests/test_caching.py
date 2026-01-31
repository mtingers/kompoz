"""Tests for caching: cached_rule, CachedPredicate, CachedPredicateFactory, use_cache."""

from __future__ import annotations

from kompoz import (
    CachedPredicate,
    CachedPredicateFactory,
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
