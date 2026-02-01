"""Tests for Retry and AsyncRetry combinators."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from kompoz import (
    AsyncRetry,
    Retry,
    RetryResult,
    pipe,
    rule,
)

# ---------------------------------------------------------------------------
# RetryResult Tests
# ---------------------------------------------------------------------------


class TestRetryResult:
    def test_retry_result_fields(self):
        result = RetryResult(ok=True, ctx=42, attempts_made=3, last_error=None)
        assert result.ok is True
        assert result.ctx == 42
        assert result.attempts_made == 3
        assert result.last_error is None

    def test_retry_result_with_error(self):
        error = ValueError("test error")
        result = RetryResult(ok=False, ctx=42, attempts_made=5, last_error=error)
        assert result.ok is False
        assert result.last_error is error


# ---------------------------------------------------------------------------
# Sync Retry
# ---------------------------------------------------------------------------


class TestRetry:
    def test_succeeds_first_try(self):
        @pipe
        def ok_fn(x):
            return x + 1

        r = Retry(ok_fn, max_attempts=3)
        ok, result = r.run(10)
        assert ok
        assert result == 11
        assert r.attempts_made == 1

    def test_succeeds_after_failures(self):
        attempts = 0

        @pipe
        def flaky(x):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("not yet")
            return x + 1

        r = Retry(flaky, max_attempts=5)
        ok, result = r.run(10)
        assert ok
        assert result == 11
        assert r.attempts_made == 3

    def test_exhausts_all_attempts(self):
        @pipe
        def always_fail(x):
            raise ValueError("nope")

        r = Retry(always_fail, max_attempts=3)
        ok, result = r.run(10)
        assert not ok
        assert result == 10  # original context
        assert r.attempts_made == 3
        # Transform catches the exception internally, so Retry sees (False, ctx)
        # rather than a raised exception — but Transform.last_error captures it
        assert always_fail.last_error is not None

    def test_exhausts_with_raising_predicate(self):
        """When inner is a Predicate that raises, Retry captures last_error."""

        @rule
        def always_raise(x):
            raise ValueError("boom")

        r = Retry(always_raise, max_attempts=3)
        ok, _ = r.run(10)
        assert not ok
        assert r.attempts_made == 3
        assert r.last_error is not None
        assert str(r.last_error) == "boom"

    def test_with_predicate_failure(self):
        """Retry should also work when a predicate returns False (not just exceptions)."""
        attempts = 0

        @rule
        def flaky_check(x):
            nonlocal attempts
            attempts += 1
            return attempts >= 3

        r = Retry(flaky_check, max_attempts=5)
        ok, _ = r.run(1)
        assert ok
        assert r.attempts_made == 3

    def test_accepts_raw_callable(self):
        r = Retry(lambda x: x * 2, max_attempts=2)
        ok, result = r.run(5)
        assert ok
        assert result == 10

    def test_backoff(self):
        """Verify backoff doesn't crash (timing not tested precisely)."""

        @pipe
        def fail_once(x):
            if not hasattr(fail_once, "_called"):
                fail_once._called = True
                raise ValueError("first")
            return x

        r = Retry(fail_once, max_attempts=2, backoff=0.001)
        ok, _ = r.run(1)
        assert ok

    def test_exponential_backoff(self):
        delays = []

        def on_retry(attempt, error, delay):
            delays.append(delay)

        attempts = 0

        @pipe
        def always_fail(x):
            nonlocal attempts
            attempts += 1
            raise ValueError("fail")

        r = Retry(
            always_fail,
            max_attempts=4,
            backoff=0.001,
            exponential=True,
            on_retry=on_retry,
        )
        r.run(1)

        # Delays should be 0.001, 0.002, 0.004 (exponential, no jitter)
        assert len(delays) == 3
        assert delays[0] < delays[1] < delays[2]

    def test_on_retry_callback(self):
        callback = MagicMock()
        attempts = 0

        @rule
        def fail_twice(x):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError(f"fail {attempts}")
            return True

        r = Retry(fail_twice, max_attempts=3, on_retry=callback)
        ok, _ = r.run(1)
        assert ok
        assert callback.call_count == 2

        # Verify callback args: (attempt, error, delay)
        first_call = callback.call_args_list[0]
        assert first_call[0][0] == 1  # attempt
        assert isinstance(first_call[0][1], ValueError)  # error

    def test_repr(self):
        r = Retry(lambda x: x, max_attempts=5, name="fetch")
        assert "fetch" in repr(r)
        assert "max_attempts=5" in repr(r)

    def test_last_error_none_on_predicate_failure(self):
        """When a predicate returns False (no exception), last_error should be None."""

        @rule
        def always_false(x):
            return False

        r = Retry(always_false, max_attempts=2)
        ok, _ = r.run(1)
        assert not ok
        assert r.last_error is None

    def test_run_with_info(self):
        """Test run_with_info returns RetryResult with all metadata."""
        attempts = 0

        @pipe
        def flaky(x):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError(f"fail {attempts}")
            return x * 2

        r = Retry(flaky, max_attempts=5)
        result = r.run_with_info(10)

        assert isinstance(result, RetryResult)
        assert result.ok is True
        assert result.ctx == 20
        assert result.attempts_made == 3
        assert result.last_error is None

    def test_run_with_info_failure(self):
        """Test run_with_info on complete failure."""

        @rule
        def always_raise(x):
            raise ValueError("boom")

        r = Retry(always_raise, max_attempts=3)
        result = r.run_with_info(10)

        assert result.ok is False
        assert result.ctx == 10
        assert result.attempts_made == 3
        assert result.last_error is not None
        assert str(result.last_error) == "boom"

    def test_run_with_info_thread_safe(self):
        """Test that run_with_info is thread-safe."""
        import threading

        attempts_per_thread = {}
        results = []
        errors = []

        def flaky_fn(x):
            thread_id = threading.current_thread().ident
            if thread_id not in attempts_per_thread:
                attempts_per_thread[thread_id] = 0
            attempts_per_thread[thread_id] += 1
            if attempts_per_thread[thread_id] < 2:
                raise ValueError(f"fail {thread_id}")
            return x * 2

        r = Retry(flaky_fn, max_attempts=5)

        def run_retry():
            try:
                result = r.run_with_info(10)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_retry) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5
        # Each result should have its own independent attempt count
        assert all(r.ok is True for r in results)
        assert all(r.attempts_made == 2 for r in results)


# ---------------------------------------------------------------------------
# Async Retry
# ---------------------------------------------------------------------------


class TestAsyncRetry:
    def test_succeeds_first_try(self):
        async def ok_fn(x):
            return x + 1

        r = AsyncRetry(ok_fn, max_attempts=3)
        ok, result = asyncio.run(r.run(10))
        assert ok
        assert result == 11
        assert r.attempts_made == 1

    def test_succeeds_after_failures(self):
        attempts = 0

        async def flaky(x):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("not yet")
            return x + 1

        r = AsyncRetry(flaky, max_attempts=5)
        ok, result = asyncio.run(r.run(10))
        assert ok
        assert result == 11

    def test_exhausts_all_attempts(self):
        async def always_fail(x):
            raise ValueError("nope")

        r = AsyncRetry(always_fail, max_attempts=3)
        ok, result = asyncio.run(r.run(10))
        assert not ok
        assert r.attempts_made == 3
        # AsyncTransform catches exceptions internally, so last_error is None
        # The inner AsyncTransform.last_error has the exception instead
        assert r.inner.last_error is not None

    def test_exhausts_with_raising_async_predicate(self):
        from kompoz import async_rule

        @async_rule
        async def always_raise(x):
            raise ValueError("boom")

        r = AsyncRetry(always_raise, max_attempts=3)
        ok, _ = asyncio.run(r.run(10))
        assert not ok
        assert r.attempts_made == 3
        assert r.last_error is not None
        assert str(r.last_error) == "boom"

    def test_accepts_async_combinator(self):
        from kompoz import async_pipe

        attempts = 0

        @async_pipe
        async def flaky(x):
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("fail")
            return x * 2

        r = AsyncRetry(flaky, max_attempts=3)
        ok, result = asyncio.run(r.run(5))
        assert ok
        assert result == 10

    def test_on_retry_sync_callback(self):
        callback = MagicMock()

        async def fail_once(x):
            if not hasattr(fail_once, "_done"):
                fail_once._done = True
                raise ValueError("first")
            return x

        r = AsyncRetry(fail_once, max_attempts=2, on_retry=callback)
        ok, _ = asyncio.run(r.run(1))
        assert ok
        assert callback.call_count == 1

    def test_on_retry_async_callback(self):
        calls = []

        async def on_retry(attempt, error, delay):
            calls.append(attempt)

        async def fail_once(x):
            if not hasattr(fail_once, "_done"):
                fail_once._done = True
                raise ValueError("first")
            return x

        r = AsyncRetry(fail_once, max_attempts=2, on_retry=on_retry)
        ok, _ = asyncio.run(r.run(1))
        assert ok
        assert calls == [1]

    def test_repr(self):
        r = AsyncRetry(lambda x: x, max_attempts=3, name="fetch")
        assert "fetch" in repr(r)
        assert "max_attempts=3" in repr(r)

    def test_run_with_info(self):
        """Test async run_with_info returns RetryResult."""
        attempts = 0

        async def flaky(x):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError(f"fail {attempts}")
            return x * 2

        r = AsyncRetry(flaky, max_attempts=5)
        result = asyncio.run(r.run_with_info(10))

        assert isinstance(result, RetryResult)
        assert result.ok is True
        assert result.ctx == 20
        assert result.attempts_made == 3
        assert result.last_error is None

    def test_run_with_info_failure(self):
        """Test async run_with_info on complete failure."""
        from kompoz import async_rule

        @async_rule
        async def always_raise(x):
            raise ValueError("boom")

        r = AsyncRetry(always_raise, max_attempts=3)
        result = asyncio.run(r.run_with_info(10))

        assert result.ok is False
        assert result.ctx == 10
        assert result.attempts_made == 3
        assert result.last_error is not None
        assert str(result.last_error) == "boom"

    def test_run_with_info_concurrent_isolated(self):
        """Test that concurrent run_with_info calls have isolated state."""
        from kompoz import async_pipe

        @async_pipe
        async def pass_through(x):
            return x

        r = AsyncRetry(pass_through, max_attempts=5)

        async def run_many():
            results = await asyncio.gather(*[r.run_with_info(i) for i in range(10)])
            return results

        results = asyncio.run(run_many())

        # All should succeed with attempts_made = 1
        assert all(res.ok is True for res in results)
        assert all(res.attempts_made == 1 for res in results)
