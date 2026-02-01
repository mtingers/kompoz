"""Tests for sync and async validation (vrule, vrule_args, async_vrule, async_vrule_args)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from kompoz import (
    AsyncValidatingCombinator,
    AsyncValidatingPredicate,
    ValidatingCombinator,
    ValidatingPredicate,
    ValidationResult,
    async_vrule,
    async_vrule_args,
    rule,
    vrule,
    vrule_args,
)


@dataclass
class User:
    name: str
    is_admin: bool = False
    is_active: bool = True
    is_banned: bool = False
    age: int = 25
    credit_score: int = 700


# ---------------------------------------------------------------------------
# Sync vrule
# ---------------------------------------------------------------------------


class TestVrule:
    def test_vrule_decorator_with_error(self):
        @vrule(error="must be admin")
        def is_admin(u):
            return u.is_admin

        assert isinstance(is_admin, ValidatingPredicate)

        result = is_admin.validate(User("Alice", is_admin=True))
        assert result.ok
        assert result.errors == []

        result = is_admin.validate(User("Bob"))
        assert not result.ok
        assert result.errors == ["must be admin"]

    def test_vrule_decorator_without_error(self):
        @vrule
        def is_active(u):
            return u.is_active

        result = is_active.validate(User("Alice", is_active=False))
        assert not result.ok
        assert result.errors == ["Check failed: is_active"]

    def test_vrule_callable_error(self):
        @vrule(error=lambda u: f"{u.name} is not admin")  # type: ignore[reportAttributeAccessIssue]
        def is_admin(u):
            return u.is_admin

        result = is_admin.validate(User("Bob"))
        assert result.errors == ["Bob is not admin"]

    def test_vrule_format_string_error(self):
        @vrule(error="User {ctx.name} must be admin")
        def is_admin(u):
            return u.is_admin

        result = is_admin.validate(User("Bob"))
        assert result.errors == ["User Bob must be admin"]

    def test_vrule_run_still_works(self):
        @vrule(error="must be admin")
        def is_admin(u):
            return u.is_admin

        ok, _ctx = is_admin.run(User("Alice", is_admin=True))
        assert ok

        ok, _ctx = is_admin.run(User("Bob"))
        assert not ok

    def test_vrule_repr(self):
        @vrule(error="err")
        def check(u):
            return True

        assert repr(check) == "ValidatingPredicate(check)"


class TestVruleArgs:
    def test_vrule_args_basic(self):
        @vrule_args(error="credit must be above {min_score}")
        def credit_above(u, min_score):
            return u.credit_score > min_score

        rule_instance = credit_above(600)
        assert isinstance(rule_instance, ValidatingPredicate)

        result = rule_instance.validate(User("Alice", credit_score=700))
        assert result.ok

        result = rule_instance.validate(User("Bob", credit_score=500))
        assert not result.ok
        assert "credit must be above 600" in result.errors[0]

    def test_vrule_args_no_error(self):
        @vrule_args
        def older_than(u, age):
            return u.age > age

        result = older_than(30).validate(User("Bob", age=25))
        assert not result.ok
        assert "Check failed:" in result.errors[0]

    def test_vrule_args_callable_error(self):
        @vrule_args(error=lambda u, min_score: f"{u.name} needs {min_score}+")
        def credit_above(u, min_score):
            return u.credit_score > min_score

        result = credit_above(800).validate(User("Bob", credit_score=700))
        assert result.errors == ["Bob needs 800+"]


class TestValidatingComposition:
    def setup_method(self):
        @vrule(error="must be admin")
        def is_admin(u):
            return u.is_admin

        @vrule(error="must be active")
        def is_active(u):
            return u.is_active

        @vrule(error="must not be banned")
        def not_banned(u):
            return not u.is_banned

        self.is_admin = is_admin
        self.is_active = is_active
        self.not_banned = not_banned

    def test_and_collects_all_errors(self):
        combined = self.is_admin & self.is_active & self.not_banned
        assert isinstance(combined, ValidatingCombinator)

        user = User("Bob", is_admin=False, is_active=False, is_banned=True)
        result = combined.validate(user)
        assert not result.ok
        assert len(result.errors) == 3
        assert "must be admin" in result.errors
        assert "must be active" in result.errors
        assert "must not be banned" in result.errors

    def test_and_passes_when_all_pass(self):
        combined = self.is_admin & self.is_active
        user = User("Alice", is_admin=True, is_active=True)
        result = combined.validate(user)
        assert result.ok
        assert result.errors == []

    def test_or_passes_when_any_passes(self):
        combined = self.is_admin | self.is_active
        user = User("Alice", is_admin=False, is_active=True)
        result = combined.validate(user)
        assert result.ok

    def test_or_fails_when_all_fail(self):
        combined = self.is_admin | self.is_active
        user = User("Bob", is_admin=False, is_active=False)
        result = combined.validate(user)
        assert not result.ok

    def test_not_inverts(self):
        not_admin = ~self.is_admin
        assert isinstance(not_admin, ValidatingCombinator)

        result = not_admin.validate(User("Bob", is_admin=False))
        assert result.ok

        result = not_admin.validate(User("Alice", is_admin=True))
        assert not result.ok

    def test_mixed_with_regular_rule(self):
        """ValidatingCombinator & regular Combinator should still work."""

        @rule
        def has_good_credit(u):
            return u.credit_score > 600

        combined = self.is_admin & has_good_credit
        user = User("Bob", is_admin=False, credit_score=500)
        result = combined.validate(user)
        assert not result.ok
        assert len(result.errors) == 2

    def test_validation_result_bool(self):
        result = ValidationResult(ok=True, errors=[], ctx=None)
        assert bool(result) is True

        result = ValidationResult(ok=False, errors=["err"], ctx=None)
        assert bool(result) is False

    def test_validation_result_raise_if_invalid(self):
        result = ValidationResult(ok=False, errors=["a", "b"], ctx=None)
        with pytest.raises(ValueError, match="a; b"):
            result.raise_if_invalid()

    def test_validation_result_raise_custom_exception(self):
        result = ValidationResult(ok=False, errors=["oops"], ctx=None)
        with pytest.raises(TypeError, match="oops"):
            result.raise_if_invalid(TypeError)

    def test_validation_result_raise_noop_when_valid(self):
        result = ValidationResult(ok=True, errors=[], ctx=None)
        result.raise_if_invalid()  # Should not raise


# ---------------------------------------------------------------------------
# Async vrule
# ---------------------------------------------------------------------------


class TestAsyncVrule:
    def test_async_vrule_with_error(self):
        @async_vrule(error="must be admin")
        async def is_admin(u):
            return u.is_admin

        assert isinstance(is_admin, AsyncValidatingPredicate)

        result = asyncio.run(is_admin.validate(User("Alice", is_admin=True)))
        assert result.ok

        result = asyncio.run(is_admin.validate(User("Bob")))
        assert not result.ok
        assert result.errors == ["must be admin"]

    def test_async_vrule_without_error(self):
        @async_vrule
        async def is_active(u):
            return u.is_active

        result = asyncio.run(is_active.validate(User("Bob", is_active=False)))
        assert not result.ok
        assert "Check failed: is_active" in result.errors[0]

    def test_async_vrule_callable_error(self):
        @async_vrule(error=lambda u: f"{u.name} denied")  # type: ignore[reportAttributeAccessIssue]
        async def is_admin(u):
            return u.is_admin

        result = asyncio.run(is_admin.validate(User("Bob")))
        assert result.errors == ["Bob denied"]

    def test_async_vrule_run(self):
        @async_vrule(error="err")
        async def is_admin(u):
            return u.is_admin

        ok, _ = asyncio.run(is_admin.run(User("Alice", is_admin=True)))
        assert ok

    def test_async_vrule_repr(self):
        @async_vrule(error="err")
        async def check(u):
            return True

        assert repr(check) == "AsyncValidatingPredicate(check)"


class TestAsyncVruleArgs:
    def test_async_vrule_args_basic(self):
        @async_vrule_args(error="score below {min_score}")
        async def score_above(u, min_score):
            return u.credit_score > min_score

        instance = score_above(600)
        assert isinstance(instance, AsyncValidatingPredicate)

        result = asyncio.run(instance.validate(User("Alice", credit_score=700)))
        assert result.ok

        result = asyncio.run(instance.validate(User("Bob", credit_score=500)))
        assert not result.ok
        assert "score below 600" in result.errors[0]

    def test_async_vrule_args_no_error(self):
        @async_vrule_args
        async def older_than(u, age):
            return u.age > age

        result = asyncio.run(older_than(30).validate(User("Bob", age=25)))
        assert not result.ok

    def test_async_vrule_args_callable_error(self):
        @async_vrule_args(error=lambda u, age: f"{u.name} too young for {age}")
        async def older_than(u, age):
            return u.age > age

        result = asyncio.run(older_than(30).validate(User("Bob", age=25)))
        assert result.errors == ["Bob too young for 30"]


class TestAsyncValidatingComposition:
    def test_and_collects_all_errors(self):
        @async_vrule(error="not admin")
        async def is_admin(u):
            return u.is_admin

        @async_vrule(error="not active")
        async def is_active(u):
            return u.is_active

        combined = is_admin & is_active
        assert isinstance(combined, AsyncValidatingCombinator)

        result = asyncio.run(
            combined.validate(User("Bob", is_admin=False, is_active=False))
        )
        assert not result.ok
        assert len(result.errors) == 2

    def test_or_passes_when_any_passes(self):
        @async_vrule(error="not admin")
        async def is_admin(u):
            return u.is_admin

        @async_vrule(error="not active")
        async def is_active(u):
            return u.is_active

        combined = is_admin | is_active
        result = asyncio.run(
            combined.validate(User("Alice", is_admin=False, is_active=True))
        )
        assert result.ok

    def test_not_inverts(self):
        @async_vrule(error="is admin")
        async def is_admin(u):
            return u.is_admin

        not_admin = ~is_admin
        result = asyncio.run(not_admin.validate(User("Bob", is_admin=False)))
        assert result.ok

        result = asyncio.run(not_admin.validate(User("Alice", is_admin=True)))
        assert not result.ok
