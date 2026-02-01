"""Tests for async combinators: async_rule, async_pipe, operators, and if_then_else."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from kompoz import (
    AsyncPredicate,
    AsyncPredicateFactory,
    AsyncTransform,
    AsyncTransformFactory,
    async_if_then_else,
    async_pipe,
    async_pipe_args,
    async_rule,
    async_rule_args,
)


@dataclass
class Ctx:
    value: int
    label: str = ""


# ---------------------------------------------------------------------------
# AsyncPredicate / async_rule
# ---------------------------------------------------------------------------


class TestAsyncRule:
    def test_basic(self):
        @async_rule
        async def is_positive(ctx):
            return ctx.value > 0

        assert isinstance(is_positive, AsyncPredicate)
        ok, result = asyncio.run(is_positive.run(Ctx(5)))
        assert ok
        assert result.value == 5

    def test_failure(self):
        @async_rule
        async def is_positive(ctx):
            return ctx.value > 0

        ok, result = asyncio.run(is_positive.run(Ctx(-1)))
        assert not ok

    def test_repr(self):
        @async_rule
        async def my_check(ctx):
            return True

        assert repr(my_check) == "AsyncPredicate(my_check)"

    def test_callable_shorthand(self):
        @async_rule
        async def is_positive(ctx):
            return ctx.value > 0

        ok, _ = asyncio.run(is_positive(Ctx(5)))
        assert ok


class TestAsyncRuleArgs:
    def test_basic(self):
        @async_rule_args
        async def greater_than(ctx, threshold):
            return ctx.value > threshold

        assert isinstance(greater_than, AsyncPredicateFactory)
        pred = greater_than(10)
        assert isinstance(pred, AsyncPredicate)

        ok, _ = asyncio.run(pred.run(Ctx(15)))
        assert ok

        ok, _ = asyncio.run(pred.run(Ctx(5)))
        assert not ok

    def test_repr(self):
        @async_rule_args
        async def gt(ctx, n):
            return ctx.value > n

        assert repr(gt) == "AsyncPredicateFactory(gt)"
        assert repr(gt(10)) == "AsyncPredicate(gt(10))"


# ---------------------------------------------------------------------------
# AsyncTransform / async_pipe
# ---------------------------------------------------------------------------


class TestAsyncPipe:
    def test_basic(self):
        @async_pipe
        async def double(ctx):
            return Ctx(ctx.value * 2, ctx.label)

        assert isinstance(double, AsyncTransform)
        ok, result = asyncio.run(double.run(Ctx(5)))
        assert ok
        assert result.value == 10

    def test_exception_returns_false(self):
        @async_pipe
        async def fail(ctx):
            raise ValueError("boom")

        ok, result = asyncio.run(fail.run(Ctx(1)))
        assert not ok
        assert result.value == 1  # original context preserved
        assert fail.last_error is not None
        assert str(fail.last_error) == "boom"

    def test_last_error_cleared_on_success(self):
        @async_pipe
        async def maybe_fail(ctx):
            if ctx.value < 0:
                raise ValueError("negative")
            return Ctx(ctx.value + 1)

        asyncio.run(maybe_fail.run(Ctx(-1)))
        assert maybe_fail.last_error is not None

        asyncio.run(maybe_fail.run(Ctx(1)))
        assert maybe_fail.last_error is None

    def test_repr(self):
        @async_pipe
        async def xform(ctx):
            return ctx

        assert repr(xform) == "AsyncTransform(xform)"


class TestAsyncPipeArgs:
    def test_basic(self):
        @async_pipe_args
        async def add(ctx, n):
            return Ctx(ctx.value + n, ctx.label)

        assert isinstance(add, AsyncTransformFactory)
        t = add(10)
        assert isinstance(t, AsyncTransform)

        ok, result = asyncio.run(t.run(Ctx(5)))
        assert ok
        assert result.value == 15


# ---------------------------------------------------------------------------
# Async operators
# ---------------------------------------------------------------------------


class TestAsyncOperators:
    def setup_method(self):
        @async_rule
        async def is_positive(ctx):
            return ctx.value > 0

        @async_rule
        async def is_even(ctx):
            return ctx.value % 2 == 0

        @async_pipe
        async def double(ctx):
            return Ctx(ctx.value * 2, ctx.label)

        self.is_positive = is_positive
        self.is_even = is_even
        self.double = double

    def test_and_both_pass(self):
        combined = self.is_positive & self.is_even
        ok, _ = asyncio.run(combined.run(Ctx(4)))
        assert ok

    def test_and_first_fails(self):
        combined = self.is_positive & self.is_even
        ok, _ = asyncio.run(combined.run(Ctx(-2)))
        assert not ok

    def test_and_second_fails(self):
        combined = self.is_positive & self.is_even
        ok, _ = asyncio.run(combined.run(Ctx(3)))
        assert not ok

    def test_or_first_passes(self):
        combined = self.is_positive | self.is_even
        ok, _ = asyncio.run(combined.run(Ctx(3)))
        assert ok

    def test_or_second_passes(self):
        combined = self.is_positive | self.is_even
        ok, _ = asyncio.run(combined.run(Ctx(-2)))
        assert ok

    def test_or_both_fail(self):
        combined = self.is_positive | self.is_even
        ok, _ = asyncio.run(combined.run(Ctx(-3)))
        assert not ok

    def test_not(self):
        not_positive = ~self.is_positive
        ok, _ = asyncio.run(not_positive.run(Ctx(-1)))
        assert ok

        ok, _ = asyncio.run(not_positive.run(Ctx(1)))
        assert not ok

    def test_double_not(self):
        double_not = ~~self.is_positive
        ok, _ = asyncio.run(double_not.run(Ctx(1)))
        assert ok

    def test_then(self):
        pipeline = self.is_positive >> self.double
        ok, result = asyncio.run(pipeline.run(Ctx(5)))
        assert ok
        assert result.value == 10

    def test_then_runs_both_regardless(self):
        """>> runs both sides even if first fails."""

        @async_rule
        async def always_fail(ctx):
            return False

        pipeline = always_fail >> self.double
        ok, result = asyncio.run(pipeline.run(Ctx(5)))
        assert ok  # double succeeds
        assert result.value == 10

    def test_complex_composition(self):
        combined = self.is_positive & (self.is_even | ~self.is_even) & self.double
        ok, result = asyncio.run(combined.run(Ctx(3)))
        assert ok
        assert result.value == 6


# ---------------------------------------------------------------------------
# Async if_then_else
# ---------------------------------------------------------------------------


class TestAsyncIfThenElse:
    def setup_method(self):
        @async_rule
        async def is_premium(ctx):
            return ctx.value > 100

        @async_pipe
        async def apply_discount(ctx):
            return Ctx(int(ctx.value * 0.8), "discounted")

        @async_pipe
        async def full_price(ctx):
            return Ctx(ctx.value, "full")

        self.is_premium = is_premium
        self.apply_discount = apply_discount
        self.full_price = full_price

    def test_then_branch(self):
        pricing = self.is_premium.if_else(self.apply_discount, self.full_price)
        ok, result = asyncio.run(pricing.run(Ctx(200)))
        assert ok
        assert result.value == 160
        assert result.label == "discounted"

    def test_else_branch(self):
        pricing = self.is_premium.if_else(self.apply_discount, self.full_price)
        ok, result = asyncio.run(pricing.run(Ctx(50)))
        assert ok
        assert result.value == 50
        assert result.label == "full"

    def test_standalone_function(self):
        pricing = async_if_then_else(
            self.is_premium, self.apply_discount, self.full_price
        )
        ok, result = asyncio.run(pricing.run(Ctx(200)))
        assert ok
        assert result.value == 160

    def test_nested_if_else(self):
        @async_rule
        async def is_vip(ctx):
            return ctx.value > 500

        @async_pipe
        async def vip_discount(ctx):
            return Ctx(int(ctx.value * 0.6), "vip")

        outer = is_vip.if_else(
            vip_discount,
            self.is_premium.if_else(self.apply_discount, self.full_price),
        )

        ok, result = asyncio.run(outer.run(Ctx(600)))
        assert result.label == "vip"

        ok, result = asyncio.run(outer.run(Ctx(200)))
        assert result.label == "discounted"

        ok, result = asyncio.run(outer.run(Ctx(50)))
        assert result.label == "full"

    def test_if_else_in_and_chain(self):
        @async_rule
        async def is_active(ctx):
            return ctx.value > 0

        pricing = self.is_premium.if_else(self.apply_discount, self.full_price)
        combined = is_active & pricing

        ok, result = asyncio.run(combined.run(Ctx(200)))
        assert ok and result.value == 160

        ok, _ = asyncio.run(combined.run(Ctx(-5)))
        assert not ok
