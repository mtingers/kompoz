"""Utility combinators: Always, Never, Debug, Try."""

from __future__ import annotations

from collections.abc import Callable

from kompoz._core import Combinator
from kompoz._types import T


class Try(Combinator[T]):
    """
    A combinator that catches exceptions and converts them to failure.

    Useful for wrapping operations that might fail.

    Example:
        fetch = Try(fetch_from_api, "fetch_api")
        ok, result = fetch.run(request)
    """

    def __init__(self, fn: Callable[[T], T], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "try")

    def _execute(self, ctx: T) -> tuple[bool, T]:
        try:
            return True, self.fn(ctx)
        except Exception:
            return False, ctx

    def __repr__(self) -> str:
        return f"Try({self.name})"


class Always(Combinator[T]):
    """A combinator that always succeeds."""

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return True, ctx

    def __repr__(self) -> str:
        return "Always()"


class Never(Combinator[T]):
    """A combinator that always fails."""

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return False, ctx

    def __repr__(self) -> str:
        return "Never()"


class Debug(Combinator[T]):
    """A combinator that prints context and always succeeds."""

    def __init__(self, label: str = "debug"):
        self.label = label

    def _execute(self, ctx: T) -> tuple[bool, T]:
        print(f"[{self.label}] {ctx}")
        return True, ctx

    def __repr__(self) -> str:
        return f"Debug({self.label!r})"
