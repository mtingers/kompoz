"""Predicate combinator and decorator factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic

from kompoz._core import Combinator
from kompoz._types import T


class Predicate(Combinator[T]):
    """
    A combinator that checks a condition without modifying context.

    Example:
        is_valid: Predicate[int] = Predicate(lambda x: x > 0, "is_positive")
        ok, _ = is_valid.run(5)  # (True, 5)
    """

    def __init__(self, fn: Callable[[T], bool], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "predicate")

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return self.fn(ctx), ctx

    def __repr__(self) -> str:
        return f"Predicate({self.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Predicate):
            return NotImplemented
        return self.fn == other.fn and self.name == other.name

    def __hash__(self) -> int:
        return hash((id(self.fn), self.name))


class PredicateFactory(Generic[T]):
    """
    A factory that creates Predicates when called with arguments.

    Used for parameterized predicates like `older_than(30)`.
    """

    def __init__(self, fn: Callable[..., bool], name: str):
        self._fn = fn
        self._name = name
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> Predicate[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return Predicate(lambda ctx: self._fn(ctx, *args, **kwargs), name)

    def __repr__(self) -> str:
        return f"PredicateFactory({self._name})"


def rule(fn: Callable[[T], bool]) -> Predicate[T]:
    """
    Decorator to create a simple rule/predicate (single context argument).

    Example:
        @rule
        def is_admin(user: User) -> bool:
            return user.is_admin

        ok, _ = is_admin.run(user)

    For parameterized rules, use @rule_args instead.
    """
    return Predicate(fn, fn.__name__)


def rule_args(fn: Callable[..., bool]) -> PredicateFactory[Any]:
    """
    Decorator to create a parameterized rule factory.

    Example:
        @rule_args
        def older_than(user: User, days: int) -> bool:
            return user.account_age_days > days

        r = older_than(30)  # Returns Predicate
        ok, _ = r.run(user)

    For simple rules (single argument), use @rule instead.
    """
    return PredicateFactory(fn, fn.__name__)


# Aliases for backwards compatibility
predicate = rule
predicate_factory = rule_args
