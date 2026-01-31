"""
Tests for DSL Modifiers (:retry and :cached)

Run with: pytest tests/test_dsl_modifiers.py -v
"""

from dataclasses import dataclass

import pytest

from kompoz import (
    CachedPredicate,
    Registry,
    Retry,
    parse_expression,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class Context:
    value: int = 0
    calls: int = 0


# =============================================================================
# Parser Tests for Modifiers
# =============================================================================


class TestModifierParser:
    """Test parsing of :retry and :cached modifiers."""

    def test_simple_retry(self):
        result = parse_expression("fetch:retry(3)")
        assert result == {"retry": {"inner": "fetch", "args": [3]}}

    def test_retry_with_backoff(self):
        result = parse_expression("fetch:retry(3, 1.0)")
        assert result == {"retry": {"inner": "fetch", "args": [3, 1.0]}}

    def test_retry_with_exponential(self):
        result = parse_expression("fetch:retry(3, 1.0, true)")
        assert result == {"retry": {"inner": "fetch", "args": [3, 1.0, True]}}

    def test_retry_all_args(self):
        result = parse_expression("fetch:retry(3, 1.0, true, 0.1)")
        assert result == {"retry": {"inner": "fetch", "args": [3, 1.0, True, 0.1]}}

    def test_retry_false_exponential(self):
        result = parse_expression("fetch:retry(3, 1.0, false)")
        assert result == {"retry": {"inner": "fetch", "args": [3, 1.0, False]}}

    def test_cached_simple(self):
        result = parse_expression("check:cached")
        assert result == {"cached": "check"}

    def test_cached_parameterized_rule(self):
        result = parse_expression("older_than(30):cached")
        assert result == {"cached": {"older_than": [30]}}

    def test_retry_grouped_or(self):
        result = parse_expression("(a | b):retry(5)")
        assert result == {"retry": {"inner": {"or": ["a", "b"]}, "args": [5]}}

    def test_retry_grouped_and(self):
        result = parse_expression("(a & b):retry(3)")
        assert result == {"retry": {"inner": {"and": ["a", "b"]}, "args": [3]}}

    def test_chained_modifiers_cached_retry(self):
        result = parse_expression("rule:cached:retry(3)")
        assert result == {"retry": {"inner": {"cached": "rule"}, "args": [3]}}

    def test_chained_modifiers_retry_cached(self):
        # Note: retry first, then cached
        result = parse_expression("rule:retry(3):cached")
        assert result == {"cached": {"retry": {"inner": "rule", "args": [3]}}}

    def test_modifier_in_and_chain(self):
        result = parse_expression("a & b:retry(3) & c")
        assert result == {
            "and": ["a", {"retry": {"inner": "b", "args": [3]}}, "c"]
        }

    def test_modifier_in_or_chain(self):
        result = parse_expression("a | b:retry(3) | c")
        assert result == {
            "or": ["a", {"retry": {"inner": "b", "args": [3]}}, "c"]
        }

    def test_complex_expression_with_modifier(self):
        result = parse_expression(
            "is_admin | (is_active & ~is_banned & fetch:retry(3, 1.0))"
        )
        expected = {
            "or": [
                "is_admin",
                {
                    "and": [
                        "is_active",
                        {"not": "is_banned"},
                        {"retry": {"inner": "fetch", "args": [3, 1.0]}},
                    ]
                },
            ]
        }
        assert result == expected

    def test_multiline_with_modifier(self):
        result = parse_expression("""
            is_admin
            | fetch_data:retry(3)
        """)
        assert result == {
            "or": ["is_admin", {"retry": {"inner": "fetch_data", "args": [3]}}]
        }

    def test_unknown_modifier_raises(self):
        with pytest.raises(ValueError, match="Unknown modifier"):
            parse_expression("rule:unknown(3)")

    def test_unknown_modifier_with_valid_name(self):
        with pytest.raises(ValueError, match="Unknown modifier.*timeout"):
            parse_expression("rule:timeout(5)")

    def test_modifier_without_parens_for_retry(self):
        # retry() without args should still work (will use defaults)
        result = parse_expression("fetch:retry()")
        assert result == {"retry": {"inner": "fetch", "args": []}}

    def test_parameterized_rule_with_modifier(self):
        result = parse_expression("older_than(30):retry(3)")
        assert result == {
            "retry": {"inner": {"older_than": [30]}, "args": [3]}
        }


# =============================================================================
# Registry Execution Tests for Modifiers
# =============================================================================


class TestModifierExecution:
    """Test execution of rules with :retry and :cached modifiers."""

    @pytest.fixture
    def registry(self):
        reg = Registry[Context]()
        self.fail_count = [0]

        @reg.predicate
        def is_positive(ctx: Context) -> bool:
            return ctx.value > 0

        @reg.predicate
        def is_even(ctx: Context) -> bool:
            return ctx.value % 2 == 0

        fail_count = self.fail_count

        @reg.predicate
        def flaky(ctx: Context) -> bool:
            """Fails first 2 times, then succeeds."""
            fail_count[0] += 1
            ctx.calls = fail_count[0]
            if fail_count[0] < 3:
                raise ConnectionError("Simulated failure")
            return True

        @reg.predicate
        def always_fail(ctx: Context) -> bool:
            raise Exception("Always fails")

        @reg.predicate
        def always_pass(ctx: Context) -> bool:
            return True

        return reg

    def test_retry_succeeds_after_failures(self, registry):
        self.fail_count[0] = 0
        rule = registry.load("flaky:retry(5)")
        ctx = Context(value=1)
        ok, result = rule.run(ctx)
        assert ok is True
        assert result.calls == 3  # Took 3 attempts

    def test_retry_exhausted(self, registry):
        rule = registry.load("always_fail:retry(3)")
        ctx = Context(value=1)
        ok, _ = rule.run(ctx)
        assert ok is False

    def test_retry_with_and_chain(self, registry):
        self.fail_count[0] = 0
        rule = registry.load("is_positive & flaky:retry(5)")
        ctx = Context(value=1)
        ok, _ = rule.run(ctx)
        assert ok is True

    def test_retry_with_or_chain(self, registry):
        self.fail_count[0] = 0
        rule = registry.load("always_fail:retry(2) | always_pass")
        ctx = Context(value=1)
        ok, _ = rule.run(ctx)
        assert ok is True  # Falls through to always_pass

    def test_retry_grouped_or(self, registry):
        self.fail_count[0] = 0
        rule = registry.load("(is_positive | flaky):retry(5)")
        ctx = Context(value=1)
        ok, _ = rule.run(ctx)
        assert ok is True  # is_positive passes immediately

    def test_cached_creates_cached_predicate(self, registry):
        rule = registry.load("is_positive:cached")
        assert isinstance(rule, CachedPredicate)

    def test_cached_with_parameterized_rule(self, registry):
        # This should work even though older_than isn't defined
        # We're just testing the parsing and wrapping
        @registry.predicate
        def older_than(ctx: Context, days: int) -> bool:
            return ctx.value > days

        rule = registry.load("older_than(30):cached")
        assert isinstance(rule, CachedPredicate)

    def test_retry_creates_retry_combinator(self, registry):
        rule = registry.load("is_positive:retry(3)")
        assert isinstance(rule, Retry)

    def test_chained_cached_retry(self, registry):
        self.fail_count[0] = 0
        rule = registry.load("flaky:cached:retry(5)")
        # Should be Retry wrapping a cached version
        assert isinstance(rule, Retry)
        ctx = Context(value=1)
        ok, _ = rule.run(ctx)
        assert ok is True

    def test_complex_expression_with_retry(self, registry):
        self.fail_count[0] = 0
        rule = registry.load("""
            is_positive 
            & is_even:retry(2)
            & flaky:retry(5)
        """)
        ctx = Context(value=2)  # positive and even
        ok, _ = rule.run(ctx)
        assert ok is True


# =============================================================================
# Edge Cases
# =============================================================================


class TestModifierEdgeCases:
    """Test edge cases for modifiers."""

    def test_modifier_on_not_expression(self):
        # ~rule:retry(3) should parse as ~(rule:retry(3))
        result = parse_expression("~rule:retry(3)")
        assert result == {"not": {"retry": {"inner": "rule", "args": [3]}}}

    def test_nested_grouping_with_modifiers(self):
        result = parse_expression("((a | b):cached):retry(3)")
        assert result == {
            "retry": {"inner": {"cached": {"or": ["a", "b"]}}, "args": [3]}
        }

    def test_bool_case_insensitive(self):
        # TRUE and True should both work
        result1 = parse_expression("fetch:retry(3, 1.0, true)")
        result2 = parse_expression("fetch:retry(3, 1.0, TRUE)")
        result3 = parse_expression("fetch:retry(3, 1.0, True)")
        # All should parse to the same thing
        assert result1["retry"]["args"][2] is True
        assert result2["retry"]["args"][2] is True
        assert result3["retry"]["args"][2] is True

    def test_false_values(self):
        result = parse_expression("fetch:retry(3, 1.0, false)")
        assert result["retry"]["args"][2] is False

    def test_modifier_preserves_comments(self):
        result = parse_expression("""
            fetch:retry(3)  # retry this flaky call
            & process
        """)
        assert result == {
            "and": [{"retry": {"inner": "fetch", "args": [3]}}, "process"]
        }
