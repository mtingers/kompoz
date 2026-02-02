"""Transform combinator and decorator factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic

from kompoz._core import Combinator
from kompoz._types import T


class Transform(Combinator[T]):
    """
    A combinator that transforms the context.

    Succeeds unless an exception is raised.

    Example:
        double: Transform[int] = Transform(lambda x: x * 2, "double")
        ok, result = double.run(5)  # (True, 10)

    For thread-safe error access, use run_with_error() which returns
    both the result and any error that occurred, rather than storing
    the error in an instance variable.

    Attributes:
        last_error: The last exception that caused failure (if any).
                   Note: This is not thread-safe. For concurrent usage,
                   use run_with_error() instead.
    """

    def __init__(self, fn: Callable[[T], T], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "transform")
        self.last_error: Exception | None = None

    def run_with_error(self, ctx: T) -> tuple[bool, T, Exception | None]:
        """
        Execute the transform and return result with error information.

        This method is thread-safe as it returns the error rather than
        storing it in an instance variable.

        Returns:
            Tuple of (success, result_context, error_or_none)
        """
        try:
            result = self.fn(ctx)
            return True, result, None
        except Exception as e:
            return False, ctx, e

    def _execute(self, ctx: T) -> tuple[bool, T]:
        ok, result, error = self.run_with_error(ctx)
        # Update instance var for backwards compatibility (not thread-safe)
        self.last_error = error
        return ok, result

    def __repr__(self) -> str:
        return f"Transform({self.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return self.fn == other.fn and self.name == other.name

    def __hash__(self) -> int:
        return hash((id(self.fn), self.name))


class TransformFactory(Generic[T]):
    """
    A factory that creates Transforms when called with arguments.

    Used for parameterized transforms like `add(10)`.
    """

    def __init__(self, fn: Callable[..., T], name: str):
        self._fn = fn
        self._name = name
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> Transform[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return Transform(lambda ctx: self._fn(ctx, *args, **kwargs), name)

    def __repr__(self) -> str:
        return f"TransformFactory({self._name})"


def pipe(fn: Callable[[T], T]) -> Transform[T]:
    """
    Decorator to create a simple transform/pipe (single context argument).

    Example:
        @pipe
        def double(data: int) -> int:
            return data * 2

        ok, result = double.run(5)

    For parameterized transforms, use @pipe_args instead.
    """
    return Transform(fn, fn.__name__)


def pipe_args(fn: Callable[..., Any]) -> TransformFactory[Any]:
    """
    Decorator to create a parameterized transform factory.

    Example:
        @pipe_args
        def multiply(data: int, factor: int) -> int:
            return data * factor

        t = multiply(10)  # Returns Transform
        ok, result = t.run(5)

    For simple transforms (single argument), use @pipe instead.
    """
    return TransformFactory(fn, fn.__name__)


# Aliases for backwards compatibility
transform = pipe
transform_factory = pipe_args
