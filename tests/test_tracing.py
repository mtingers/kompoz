"""Tests for tracing (use_tracing, PrintHook, LoggingHook, TraceConfig) and explain()."""

from __future__ import annotations

import asyncio
import logging
from io import StringIO
from unittest.mock import MagicMock

from kompoz import (
    Always,
    Debug,
    Never,
    Retry,
    TraceConfig,
    async_rule,
    explain,
    if_then_else,
    pipe,
    rule,
    rule_args,
    run_async_traced,
    run_traced,
    use_tracing,
    PrintHook,
    LoggingHook,
)


# ---------------------------------------------------------------------------
# explain()
# ---------------------------------------------------------------------------


class TestExplain:
    def test_simple_predicate(self):
        @rule
        def is_admin(u):
            return True

        result = explain(is_admin)
        assert "Check: is_admin" in result

    def test_simple_transform(self):
        @pipe
        def double(x):
            return x * 2

        result = explain(double)
        assert "Transform: double" in result

    def test_and(self):
        @rule
        def a(x):
            return True

        @rule
        def b(x):
            return True

        result = explain(a & b)
        assert "ALL of:" in result
        assert "a" in result
        assert "b" in result

    def test_or(self):
        @rule
        def a(x):
            return True

        @rule
        def b(x):
            return True

        result = explain(a | b)
        assert "ANY of:" in result

    def test_not(self):
        @rule
        def is_banned(x):
            return True

        result = explain(~is_banned)
        assert "NOT:" in result
        assert "is_banned" in result

    def test_then(self):
        @pipe
        def a(x):
            return x

        @pipe
        def b(x):
            return x

        result = explain(a >> b)
        assert "Sequence" in result or "sequence" in result or "Execute" in result

    def test_if_then_else(self):
        @rule
        def cond(x):
            return True

        @pipe
        def a(x):
            return x

        @pipe
        def b(x):
            return x

        result = explain(if_then_else(cond, a, b))
        assert "IF" in result
        assert "THEN" in result
        assert "ELSE" in result

    def test_always_never(self):
        assert "Always" in explain(Always())
        assert "fail" in explain(Never()).lower() or "Never" in explain(Never())

    def test_debug(self):
        result = explain(Debug("test"))
        assert "Debug" in result
        assert "test" in result

    def test_retry(self):
        @pipe
        def fetch(x):
            return x

        result = explain(Retry(fetch, max_attempts=5))
        assert "Retry" in result
        assert "5" in result

    def test_complex_nested(self):
        @rule
        def is_admin(u):
            return True

        @rule
        def is_active(u):
            return True

        @rule
        def is_banned(u):
            return False

        @rule_args
        def older_than(u, days):
            return True

        combined = is_admin | (is_active & ~is_banned & older_than(30))
        result = explain(combined)
        assert "ANY of:" in result
        assert "ALL of:" in result
        assert "NOT:" in result
        assert "is_admin" in result
        assert "older_than(30)" in result


# ---------------------------------------------------------------------------
# PrintHook
# ---------------------------------------------------------------------------


class TestPrintHook:
    def test_basic_tracing(self, capsys):
        @rule
        def is_positive(x):
            return x > 0

        with use_tracing(PrintHook()):
            is_positive.run(5)

        output = capsys.readouterr().out
        assert "Predicate(is_positive)" in output
        assert "->" in output
        assert "<-" in output

    def test_nested_tracing(self, capsys):
        @rule
        def a(x):
            return True

        @rule
        def b(x):
            return True

        combined = a & b
        with use_tracing(PrintHook()):
            combined.run(1)

        output = capsys.readouterr().out
        assert "AND" in output
        assert "Predicate(a)" in output
        assert "Predicate(b)" in output

    def test_show_ctx(self, capsys):
        @rule
        def check(x):
            return True

        with use_tracing(PrintHook(show_ctx=True)):
            check.run(42)

        output = capsys.readouterr().out
        assert "ctx=42" in output

    def test_success_failure_markers(self, capsys):
        @rule
        def pass_check(x):
            return True

        @rule
        def fail_check(x):
            return False

        with use_tracing(PrintHook()):
            pass_check.run(1)
            fail_check.run(1)

        output = capsys.readouterr().out
        # Uses unicode checkmark/cross
        assert "\u2714" in output or "✔" in output
        assert "\u2717" in output or "✗" in output


class TestRunTraced:
    def test_explicit_tracing(self, capsys):
        @rule
        def check(x):
            return True

        ok, _ = run_traced(check, 1, PrintHook())
        assert ok

        output = capsys.readouterr().out
        assert "Predicate(check)" in output

    def test_async_tracing(self, capsys):
        @async_rule
        async def check(x):
            return True

        ok, _ = asyncio.run(run_async_traced(check, 1, PrintHook()))
        assert ok

        output = capsys.readouterr().out
        assert "AsyncPredicate(check)" in output


# ---------------------------------------------------------------------------
# LoggingHook
# ---------------------------------------------------------------------------


class TestLoggingHook:
    def test_basic(self):
        logger = MagicMock()

        @rule
        def check(x):
            return True

        with use_tracing(LoggingHook(logger)):
            check.run(1)

        assert logger.log.call_count >= 2  # on_enter + on_exit

    def test_error_logging(self):
        """on_error fires when a Predicate raises (Transform catches exceptions internally)."""
        logger = MagicMock()

        @rule
        def boom(x):
            raise RuntimeError("kaboom")

        hook = LoggingHook(logger)
        try:
            run_traced(boom, 1, hook)
        except RuntimeError:
            pass

        logger.error.assert_called()
        assert "kaboom" in str(logger.error.call_args)


# ---------------------------------------------------------------------------
# TraceConfig
# ---------------------------------------------------------------------------


class TestTraceConfig:
    def test_max_depth(self, capsys):
        @rule
        def a(x):
            return True

        @rule
        def b(x):
            return True

        combined = a & b

        with use_tracing(PrintHook(), TraceConfig(max_depth=0)):
            combined.run(1)

        output = capsys.readouterr().out
        # At max_depth=0, only the root AND is traced, children are not
        assert "AND" in output
        # Children should not appear as traced spans
        lines = [l for l in output.strip().split("\n") if "Predicate" in l]
        assert len(lines) == 0

    def test_include_leaf_only(self, capsys):
        @rule
        def a(x):
            return True

        @rule
        def b(x):
            return True

        combined = a & b

        with use_tracing(PrintHook(), TraceConfig(include_leaf_only=True)):
            combined.run(1)

        output = capsys.readouterr().out
        # Should trace leaf predicates but not the AND node
        assert "Predicate(a)" in output
        assert "Predicate(b)" in output
        # AND should not have its own span arrows
        and_lines = [
            l
            for l in output.strip().split("\n")
            if "AND" in l and ("->" in l or "<-" in l)
        ]
        assert len(and_lines) == 0

    def test_nested_false(self, capsys):
        @rule
        def a(x):
            return True

        @rule
        def b(x):
            return True

        combined = a & b

        with use_tracing(PrintHook(), TraceConfig(nested=False)):
            combined.run(1)

        output = capsys.readouterr().out
        # Only the root AND is traced, no children
        assert "AND" in output
        assert "Predicate" not in output


# ---------------------------------------------------------------------------
# Tracing with use_tracing context manager
# ---------------------------------------------------------------------------


class TestUseTracing:
    def test_tracing_disabled_outside_scope(self):
        """Without use_tracing, no tracing happens."""
        hook = MagicMock()

        @rule
        def check(x):
            return True

        check.run(1)
        hook.on_enter.assert_not_called()

    def test_tracing_enabled_in_scope(self):
        hook = MagicMock()
        hook.on_enter.return_value = None

        @rule
        def check(x):
            return True

        with use_tracing(hook):
            check.run(1)

        hook.on_enter.assert_called()
        hook.on_exit.assert_called()

    def test_tracing_restored_after_scope(self):
        hook = MagicMock()
        hook.on_enter.return_value = None

        @rule
        def check(x):
            return True

        with use_tracing(hook):
            check.run(1)

        hook.reset_mock()
        check.run(1)
        hook.on_enter.assert_not_called()

    def test_nested_scopes(self):
        outer_hook = MagicMock()
        outer_hook.on_enter.return_value = None
        inner_hook = MagicMock()
        inner_hook.on_enter.return_value = None

        @rule
        def check(x):
            return True

        with use_tracing(outer_hook):
            check.run(1)
            with use_tracing(inner_hook):
                check.run(2)
            check.run(3)

        assert outer_hook.on_enter.call_count == 2  # calls 1 and 3
        assert inner_hook.on_enter.call_count == 1  # call 2
