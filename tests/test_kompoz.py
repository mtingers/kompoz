"""
Tests for Kompoz - Composable Predicate & Transform Combinators

Run with: pytest tests/test_kompoz.py -v
"""

from dataclasses import dataclass

import pytest

from kompoz import (
    Always,
    Debug,
    Never,
    Predicate,
    Registry,
    Transform,
    Try,
    predicate,
    transform,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class User:
    name: str
    is_admin: bool = False
    is_active: bool = True
    is_banned: bool = False
    account_age_days: int = 0
    credit_score: int = 500
    country: str = "US"
    has_override: bool = False


@dataclass
class Data:
    value: int


# =============================================================================
# Basic Predicate Tests
# =============================================================================


class TestPredicate:
    """Test basic predicate functionality."""

    def test_simple_predicate(self) -> None:
        is_positive: Predicate[int] = Predicate(lambda x: x > 0, "is_positive")
        assert is_positive.run(5) == (True, 5)
        assert is_positive.run(-5) == (False, -5)
        assert is_positive.run(0) == (False, 0)

    def test_predicate_decorator_simple(self) -> None:
        @predicate
        def is_even(x: int) -> bool:
            return x % 2 == 0

        assert is_even.run(4) == (True, 4)
        assert is_even.run(3) == (False, 3)

    def test_predicate_decorator_parameterized(self) -> None:
        @predicate
        def greater_than(x: int, threshold: int) -> bool:
            return x > threshold

        gt_10 = greater_than(10)
        assert gt_10.run(15) == (True, 15)
        assert gt_10.run(5) == (False, 5)
        assert gt_10.run(10) == (False, 10)

    def test_predicate_repr(self) -> None:
        @predicate
        def is_valid(x: int) -> bool:
            return True

        assert "is_valid" in repr(is_valid)


# =============================================================================
# Basic Transform Tests
# =============================================================================


class TestTransform:
    """Test basic transform functionality."""

    def test_simple_transform(self) -> None:
        double: Transform[int] = Transform(lambda x: x * 2, "double")
        assert double.run(5) == (True, 10)
        assert double.run(0) == (True, 0)

    def test_transform_decorator_simple(self) -> None:
        @transform
        def increment(x: int) -> int:
            return x + 1

        assert increment.run(5) == (True, 6)

    def test_transform_decorator_parameterized(self) -> None:
        @transform
        def add(x: int, n: int) -> int:
            return x + n

        add_10 = add(10)
        assert add_10.run(5) == (True, 15)

    def test_transform_exception_returns_false(self) -> None:
        @transform
        def divide_by_zero(x: int) -> int:
            return x // 0  # Will raise ZeroDivisionError

        ok, result = divide_by_zero.run(5)
        assert ok is False
        assert result == 5  # original value preserved

    def test_transform_repr(self) -> None:
        @transform
        def process(x: int) -> int:
            return x

        assert "process" in repr(process)


# =============================================================================
# Operator Tests
# =============================================================================


class TestOperators:
    """Test combinator operators."""

    def test_and_both_true(self) -> None:
        @predicate
        def is_positive(x: int) -> bool:
            return x > 0

        @predicate
        def is_even(x: int) -> bool:
            return x % 2 == 0

        rule = is_positive & is_even
        assert rule.run(4)[0] is True

    def test_and_first_false(self) -> None:
        @predicate
        def is_positive(x: int) -> bool:
            return x > 0

        @predicate
        def is_even(x: int) -> bool:
            return x % 2 == 0

        rule = is_positive & is_even
        assert rule.run(-4)[0] is False

    def test_and_second_false(self) -> None:
        @predicate
        def is_positive(x: int) -> bool:
            return x > 0

        @predicate
        def is_even(x: int) -> bool:
            return x % 2 == 0

        rule = is_positive & is_even
        assert rule.run(3)[0] is False

    def test_or_first_true(self) -> None:
        @predicate
        def is_positive(x: int) -> bool:
            return x > 0

        @predicate
        def is_zero(x: int) -> bool:
            return x == 0

        rule = is_positive | is_zero
        assert rule.run(5)[0] is True

    def test_or_second_true(self) -> None:
        @predicate
        def is_positive(x: int) -> bool:
            return x > 0

        @predicate
        def is_zero(x: int) -> bool:
            return x == 0

        rule = is_positive | is_zero
        assert rule.run(0)[0] is True

    def test_or_both_false(self) -> None:
        @predicate
        def is_positive(x: int) -> bool:
            return x > 0

        @predicate
        def is_zero(x: int) -> bool:
            return x == 0

        rule = is_positive | is_zero
        assert rule.run(-5)[0] is False

    def test_not(self) -> None:
        @predicate
        def is_banned(x: int) -> bool:
            return x < 0

        rule = ~is_banned
        assert rule.run(5)[0] is True
        assert rule.run(-5)[0] is False

    def test_then_operator(self) -> None:
        @transform
        def double(x: int) -> int:
            return x * 2

        @transform
        def add_one(x: int) -> int:
            return x + 1

        rule = double >> add_one
        ok, result = rule.run(5)
        assert ok is True
        assert result == 11  # (5 * 2) + 1

    def test_complex_combination(self) -> None:
        @predicate
        def is_positive(x: int) -> bool:
            return x > 0

        @predicate
        def is_even(x: int) -> bool:
            return x % 2 == 0

        @predicate
        def is_small(x: int) -> bool:
            return x < 100

        rule = is_positive & (is_even | is_small)

        assert rule.run(4)[0] is True  # positive and even
        assert rule.run(3)[0] is True  # positive and small
        assert rule.run(102)[0] is True  # positive and even (not small)
        assert rule.run(103)[0] is False  # positive but odd and not small
        assert rule.run(-4)[0] is False  # not positive


# =============================================================================
# Registry Tests
# =============================================================================


class TestRegistry:
    """Test registry and config loading."""

    @pytest.fixture
    def user_registry(self) -> Registry[User]:
        reg: Registry[User] = Registry()

        @reg.predicate
        def is_admin(u: User) -> bool:
            return u.is_admin

        @reg.predicate
        def is_active(u: User) -> bool:
            return u.is_active

        @reg.predicate
        def is_banned(u: User) -> bool:
            return u.is_banned

        @reg.predicate
        def has_override(u: User) -> bool:
            return u.has_override

        @reg.predicate
        def account_older_than(u: User, days: int) -> bool:
            return u.account_age_days > days

        @reg.predicate
        def credit_above(u: User, score: int) -> bool:
            return u.credit_score > score

        @reg.predicate
        def from_country(u: User, countries: list[str]) -> bool:
            return u.country in countries

        return reg

    def test_load_simple_predicate(self, user_registry: Registry[User]) -> None:
        rule = user_registry.load("is_admin")
        assert rule.run(User("Admin", is_admin=True))[0] is True
        assert rule.run(User("User", is_admin=False))[0] is False

    def test_load_parameterized_predicate(self, user_registry: Registry[User]) -> None:
        rule = user_registry.load({"account_older_than": [30]})
        assert rule.run(User("Old", account_age_days=60))[0] is True
        assert rule.run(User("New", account_age_days=10))[0] is False

    def test_load_and_config(self, user_registry: Registry[User]) -> None:
        rule = user_registry.load({"and": ["is_active", {"not": "is_banned"}]})
        assert rule.run(User("Good", is_active=True, is_banned=False))[0] is True
        assert rule.run(User("Bad", is_active=True, is_banned=True))[0] is False

    def test_load_or_config(self, user_registry: Registry[User]) -> None:
        rule = user_registry.load({"or": ["is_admin", "has_override"]})
        assert rule.run(User("Admin", is_admin=True))[0] is True
        assert rule.run(User("Override", has_override=True))[0] is True
        assert rule.run(User("Normal"))[0] is False

    def test_load_complex_config(self, user_registry: Registry[User]) -> None:
        config = {
            "or": [
                "is_admin",
                {
                    "and": [
                        "is_active",
                        {"not": "is_banned"},
                        {"account_older_than": [30]},
                        {"from_country": [["US", "NL", "BE"]]},
                        {"or": [{"credit_above": [650]}, "has_override"]},
                    ]
                },
            ]
        }
        rule = user_registry.load(config)

        # Should pass
        assert rule.run(User("Admin", is_admin=True))[0] is True
        assert rule.run(User("Good", account_age_days=60, credit_score=700))[0] is True
        assert (
            rule.run(User("Override", account_age_days=60, has_override=True))[0]
            is True
        )

        # Should fail
        assert (
            rule.run(
                User("Banned", is_banned=True, account_age_days=60, credit_score=700)
            )[0]
            is False
        )
        assert rule.run(User("New", account_age_days=10, credit_score=700))[0] is False
        assert (
            rule.run(User("LowCredit", account_age_days=60, credit_score=500))[0]
            is False
        )
        assert (
            rule.run(
                User(
                    "WrongCountry", account_age_days=60, credit_score=700, country="DE"
                )
            )[0]
            is False
        )

    def test_load_empty_and(self, user_registry: Registry[User]) -> None:
        rule = user_registry.load({"and": []})
        assert rule.run(User("Anyone"))[0] is True  # Always

    def test_unknown_predicate_raises(self, user_registry: Registry[User]) -> None:
        with pytest.raises(ValueError, match="Unknown predicate"):
            user_registry.load("nonexistent_predicate")

    def test_missing_args_raises(self, user_registry: Registry[User]) -> None:
        with pytest.raises(ValueError, match="requires arguments"):
            user_registry.load("account_older_than")


# =============================================================================
# Transform Pipeline Tests
# =============================================================================


class TestTransformPipeline:
    """Test data transformation pipelines."""

    @pytest.fixture
    def data_registry(self) -> Registry[Data]:
        reg: Registry[Data] = Registry()

        @reg.transform
        def double(d: Data) -> Data:
            return Data(d.value * 2)

        @reg.transform
        def add(d: Data, n: int) -> Data:
            return Data(d.value + n)

        @reg.predicate
        def is_positive(d: Data) -> bool:
            return d.value > 0

        return reg

    def test_simple_transform_chain(self, data_registry: Registry[Data]) -> None:
        rule = data_registry.load({"and": ["double", {"add": [10]}]})
        ok, result = rule.run(Data(5))
        assert ok is True
        assert result.value == 20  # (5 * 2) + 10

    def test_transform_with_predicate(self, data_registry: Registry[Data]) -> None:
        rule = data_registry.load({"and": ["is_positive", "double"]})

        ok, result = rule.run(Data(5))
        assert ok is True
        assert result.value == 10

        ok, result = rule.run(Data(-5))
        assert ok is False
        assert result.value == -5


# =============================================================================
# Utility Combinator Tests
# =============================================================================


class TestUtilityCombinators:
    """Test utility combinators."""

    def test_always(self) -> None:
        always: Always[str] = Always()
        assert always.run("anything")[0] is True
        assert always.run(None)[0] is True  # type: ignore[arg-type]

    def test_never(self) -> None:
        never: Never[str] = Never()
        assert never.run("anything")[0] is False

    def test_debug(self, capsys: pytest.CaptureFixture[str]) -> None:
        debug: Debug[str] = Debug("test")
        ok, result = debug.run("hello")
        assert ok is True
        assert result == "hello"
        captured = capsys.readouterr()
        assert "[test]" in captured.out
        assert "hello" in captured.out


# =============================================================================
# Try Combinator Tests
# =============================================================================


class TestTry:
    """Test Try combinator for exception handling."""

    def test_try_success(self) -> None:
        def safe_op(x: int) -> int:
            return x * 2

        t: Try[int] = Try(safe_op, "safe_op")
        ok, result = t.run(5)
        assert ok is True
        assert result == 10

    def test_try_failure(self) -> None:
        def unsafe_op(x: int) -> int:
            raise ValueError("oops")

        t: Try[int] = Try(unsafe_op, "unsafe_op")
        ok, result = t.run(5)
        assert ok is False
        assert result == 5  # original preserved

    def test_try_as_fallback(self) -> None:
        call_count = {"primary": 0, "backup": 0}

        def primary(x: int) -> int:
            call_count["primary"] += 1
            raise Exception("primary failed")

        def backup(x: int) -> int:
            call_count["backup"] += 1
            return x * 2

        pipeline = Try(primary, "primary") | Try(backup, "backup")
        ok, result = pipeline.run(5)

        assert ok is True
        assert result == 10
        assert call_count["primary"] == 1
        assert call_count["backup"] == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_callable_shorthand(self) -> None:
        @predicate
        def is_positive(x: int) -> bool:
            return x > 0

        # Can use () instead of .run()
        ok, result = is_positive(5)
        assert ok is True
        assert result == 5

    def test_context_preserved_on_failure(self) -> None:
        @predicate
        def always_false(x: dict[str, str]) -> bool:
            return False

        original = {"key": "value"}
        ok, result = always_false.run(original)
        assert ok is False
        assert result is original

    def test_chained_transforms_preserve_context(self) -> None:
        @transform
        def add_a(d: dict[str, int]) -> dict[str, int]:
            return {**d, "a": 1}

        @transform
        def add_b(d: dict[str, int]) -> dict[str, int]:
            return {**d, "b": 2}

        pipeline = add_a & add_b
        ok, result = pipeline.run({})
        assert ok is True
        assert result == {"a": 1, "b": 2}


# =============================================================================
# Parameterized Tests
# =============================================================================


class TestParameterized:
    """Parameterized tests for thorough coverage."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (10, True),
            (5, False),
            (0, False),
            (-5, False),
        ],
    )
    def test_greater_than_5(self, value: int, expected: bool) -> None:
        @predicate
        def gt_5(x: int) -> bool:
            return x > 5

        assert gt_5.run(value)[0] is expected

    @pytest.mark.parametrize(
        "account_age,credit,expected",
        [
            (60, 700, True),  # meets all criteria
            (60, 500, False),  # low credit
            (10, 700, False),  # new account
            (10, 500, False),  # both bad
        ],
    )
    def test_complex_rule(self, account_age: int, credit: int, expected: bool) -> None:
        @predicate
        def old_account(u: User) -> bool:
            return u.account_age_days > 30

        @predicate
        def good_credit(u: User) -> bool:
            return u.credit_score > 600

        rule = old_account & good_credit
        user = User("Test", account_age_days=account_age, credit_score=credit)
        assert rule.run(user)[0] is expected
