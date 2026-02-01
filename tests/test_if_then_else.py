"""Tests for sync if_then_else and Combinator.if_else()."""

from __future__ import annotations

from kompoz import (
    Always,
    Never,
    if_then_else,
    pipe,
    rule,
)


class TestIfElseMethod:
    def setup_method(self):
        @rule
        def is_positive(x):
            return x > 0

        @pipe
        def double(x):
            return x * 2

        @pipe
        def negate(x):
            return -x

        self.is_positive = is_positive
        self.double = double
        self.negate = negate

    def test_then_branch(self):
        cond = self.is_positive.if_else(self.double, self.negate)
        ok, result = cond.run(5)
        assert ok
        assert result == 10

    def test_else_branch(self):
        cond = self.is_positive.if_else(self.double, self.negate)
        ok, result = cond.run(-3)
        assert ok
        assert result == 3

    def test_condition_context_passed_to_branch(self):
        """The branch receives the context from the condition evaluation."""

        @rule
        def always_true(x):
            return True

        @pipe
        def add_one(x):
            return x + 1

        @pipe
        def identity(x):
            return x

        cond = always_true.if_else(add_one, identity)
        ok, result = cond.run(10)
        assert result == 11

    def test_nested_if_else(self):
        @rule
        def gt10(x):
            return x > 10

        @rule
        def gt5(x):
            return x > 5

        @pipe
        def label_high(x):
            return "high"

        @pipe
        def label_mid(x):
            return "mid"

        @pipe
        def label_low(x):
            return "low"

        nested = gt10.if_else(label_high, gt5.if_else(label_mid, label_low))

        _, result = nested.run(15)
        assert result == "high"

        _, result = nested.run(7)
        assert result == "mid"

        _, result = nested.run(2)
        assert result == "low"


class TestIfThenElseFunction:
    def test_basic(self):
        @rule
        def is_even(x):
            return x % 2 == 0

        @pipe
        def half(x):
            return x // 2

        @pipe
        def triple(x):
            return x * 3

        cond = if_then_else(is_even, half, triple)
        _, result = cond.run(10)
        assert result == 5

        _, result = cond.run(3)
        assert result == 9

    def test_with_always_never(self):
        @pipe
        def inc(x):
            return x + 1

        @pipe
        def dec(x):
            return x - 1

        always_inc = if_then_else(Always(), inc, dec)
        _, result = always_inc.run(10)
        assert result == 11

        always_dec = if_then_else(Never(), inc, dec)
        _, result = always_dec.run(10)
        assert result == 9

    def test_branch_failure_propagates(self):
        @rule
        def is_positive(x):
            return x > 0

        cond = if_then_else(is_positive, Never(), Always())

        ok, _ = cond.run(5)
        assert not ok  # then_branch is Never

        ok, _ = cond.run(-1)
        assert ok  # else_branch is Always

    def test_in_and_chain(self):
        @rule
        def is_positive(x):
            return x > 0

        @rule
        def is_even(x):
            return x % 2 == 0

        @pipe
        def double(x):
            return x * 2

        @pipe
        def identity(x):
            return x

        chain = is_positive & if_then_else(is_even, double, identity)

        ok, result = chain.run(4)
        assert ok and result == 8

        ok, result = chain.run(3)
        assert ok and result == 3

        ok, _ = chain.run(-2)
        assert not ok

    def test_in_or_chain(self):
        @rule
        def always_fail(x):
            return False

        cond = if_then_else(always_fail, Never(), Always())
        chain = always_fail | cond

        ok, _ = chain.run(1)
        assert ok  # cond takes else branch -> Always
