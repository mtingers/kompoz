"""
Kompoz - Composable Predicate & Transform Combinators

A Python library for building composable, declarative rule chains using
operator overloading. Supports boolean logic (AND, OR, NOT), data pipelines,
and config-driven rules via JSON/YAML.

Operators:
    &  = "and then" (sequence, short-circuits on failure)
    |  = "or else" (fallback, short-circuits on success)
    ~  = "not" / "inverse"
    >> = "then" (always runs both, keeps second result)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = [
    "Combinator",
    "Predicate",
    "Transform",
    "Try",
    "Always",
    "Never",
    "Debug",
    "Registry",
    "predicate",
    "transform",
]

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# =============================================================================
# Core Combinator Base
# =============================================================================


class Combinator(ABC, Generic[T]):
    """
    Base class for all combinators.

    A combinator takes a context and returns (success: bool, new_context).
    Combinators can be composed using operators:
        &  = and then (short-circuits on failure)
        |  = or else (short-circuits on success)
        ~  = not (inverts success/failure)
        >> = then (always runs both)
    """

    @abstractmethod
    def run(self, ctx: T) -> tuple[bool, T]:
        """Execute the combinator and return (success, new_context)."""
        ...

    def __and__(self, other: Combinator[T]) -> Combinator[T]:
        """a & b = run b only if a succeeds."""
        return _And(self, other)

    def __or__(self, other: Combinator[T]) -> Combinator[T]:
        """a | b = run b only if a fails."""
        return _Or(self, other)

    def __invert__(self) -> Combinator[T]:
        """~a = invert success/failure."""
        return _Not(self)

    def __rshift__(self, other: Combinator[T]) -> Combinator[T]:
        """a >> b = run b regardless of a's result (keep b's result)."""
        return _Then(self, other)

    def __call__(self, ctx: T) -> tuple[bool, T]:
        """Shorthand for run()."""
        return self.run(ctx)


@dataclass
class _And(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def run(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = self.left.run(ctx)
        if not ok:
            return False, ctx
        return self.right.run(ctx)


@dataclass
class _Or(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def run(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = self.left.run(ctx)
        if ok:
            return True, ctx
        return self.right.run(ctx)


@dataclass
class _Not(Combinator[T]):
    inner: Combinator[T]

    def run(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = self.inner.run(ctx)
        return not ok, ctx


@dataclass
class _Then(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def run(self, ctx: T) -> tuple[bool, T]:
        _, ctx = self.left.run(ctx)
        return self.right.run(ctx)


# =============================================================================
# Predicate Combinator
# =============================================================================


class Predicate(Combinator[T]):
    """
    A combinator that checks a condition without modifying context.

    Example:
        is_valid = Predicate(lambda x: x > 0, "is_positive")
        ok, _ = is_valid.run(5)  # (True, 5)
    """

    def __init__(self, fn: Callable[[T], bool], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "predicate")

    def run(self, ctx: T) -> tuple[bool, T]:
        return self.fn(ctx), ctx

    def __repr__(self) -> str:
        return f"Predicate({self.name})"


def predicate(fn: Callable[..., bool]) -> Predicate[T] | Callable[..., Predicate[T]]:
    """
    Decorator to create a predicate from a function.

    Simple predicate (single argument - the context):
        @predicate
        def is_admin(user):
            return user.is_admin

        # Usage: is_admin.run(user)

    Parameterized predicate (extra arguments):
        @predicate
        def older_than(user, days):
            return user.age > days

        # Usage: older_than(30).run(user)
    """
    params = list(inspect.signature(fn).parameters)

    if len(params) == 1:
        return Predicate(fn, fn.__name__)
    else:

        def factory(*args: Any, **kwargs: Any) -> Predicate[T]:
            name = f"{fn.__name__}({', '.join(map(repr, args))})"
            return Predicate(lambda ctx: fn(ctx, *args, **kwargs), name)

        factory.__name__ = fn.__name__
        return factory


# =============================================================================
# Transform Combinator
# =============================================================================


class Transform(Combinator[T]):
    """
    A combinator that transforms the context.

    Succeeds unless an exception is raised.

    Example:
        double = Transform(lambda x: x * 2, "double")
        ok, result = double.run(5)  # (True, 10)
    """

    def __init__(self, fn: Callable[[T], T], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "transform")

    def run(self, ctx: T) -> tuple[bool, T]:
        try:
            return True, self.fn(ctx)
        except Exception:
            return False, ctx

    def __repr__(self) -> str:
        return f"Transform({self.name})"


def transform(fn: Callable[..., T]) -> Transform[T] | Callable[..., Transform[T]]:
    """
    Decorator to create a transform from a function.

    Simple transform (single argument - the context):
        @transform
        def double(data):
            return data * 2

        # Usage: double.run(5)

    Parameterized transform (extra arguments):
        @transform
        def multiply(data, factor):
            return data * factor

        # Usage: multiply(10).run(5)
    """
    params = list(inspect.signature(fn).parameters)

    if len(params) == 1:
        return Transform(fn, fn.__name__)
    else:

        def factory(*args: Any, **kwargs: Any) -> Transform[T]:
            name = f"{fn.__name__}({', '.join(map(repr, args))})"
            return Transform(lambda ctx: fn(ctx, *args, **kwargs), name)

        factory.__name__ = fn.__name__
        return factory


# =============================================================================
# Try Combinator
# =============================================================================


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

    def run(self, ctx: T) -> tuple[bool, T]:
        try:
            return True, self.fn(ctx)
        except Exception:
            return False, ctx

    def __repr__(self) -> str:
        return f"Try({self.name})"


# =============================================================================
# Utility Combinators
# =============================================================================


class Always(Combinator[T]):
    """A combinator that always succeeds."""

    def run(self, ctx: T) -> tuple[bool, T]:
        return True, ctx

    def __repr__(self) -> str:
        return "Always()"


class Never(Combinator[T]):
    """A combinator that always fails."""

    def run(self, ctx: T) -> tuple[bool, T]:
        return False, ctx

    def __repr__(self) -> str:
        return "Never()"


class Debug(Combinator[T]):
    """A combinator that prints context and always succeeds."""

    def __init__(self, label: str = "debug"):
        self.label = label

    def run(self, ctx: T) -> tuple[bool, T]:
        print(f"[{self.label}] {ctx}")
        return True, ctx

    def __repr__(self) -> str:
        return f"Debug({self.label!r})"


# =============================================================================
# Registry & Config Loader
# =============================================================================


class Registry(Generic[T]):
    """
    Registry for named predicates and transforms.

    Allows loading combinator chains from JSON/YAML config files.

    Example:
        reg = Registry[User]()

        @reg.predicate
        def is_admin(u):
            return u.is_admin

        @reg.predicate
        def older_than(u, days):
            return u.age > days

        # Load from config
        rule = reg.load({"or": ["is_admin", {"older_than": [30]}]})
    """

    def __init__(self) -> None:
        self._predicates: dict[str, Predicate[T] | Callable[..., Predicate[T]]] = {}
        self._transforms: dict[str, Transform[T] | Callable[..., Transform[T]]] = {}

    def predicate(
        self, fn: Callable[..., bool]
    ) -> Predicate[T] | Callable[..., Predicate[T]]:
        """Decorator to register a predicate."""
        params = list(inspect.signature(fn).parameters)

        if len(params) == 1:
            p = Predicate(fn, fn.__name__)
            self._predicates[fn.__name__] = p
            return p
        else:

            def factory(*args: Any, **kwargs: Any) -> Predicate[T]:
                name = f"{fn.__name__}({', '.join(map(repr, args))})"
                return Predicate(lambda ctx: fn(ctx, *args, **kwargs), name)

            factory.__name__ = fn.__name__
            self._predicates[fn.__name__] = factory
            return factory

    def transform(
        self, fn: Callable[..., T]
    ) -> Transform[T] | Callable[..., Transform[T]]:
        """Decorator to register a transform."""
        params = list(inspect.signature(fn).parameters)

        if len(params) == 1:
            t = Transform(fn, fn.__name__)
            self._transforms[fn.__name__] = t
            return t
        else:

            def factory(*args: Any, **kwargs: Any) -> Transform[T]:
                name = f"{fn.__name__}({', '.join(map(repr, args))})"
                return Transform(lambda ctx: fn(ctx, *args, **kwargs), name)

            factory.__name__ = fn.__name__
            self._transforms[fn.__name__] = factory
            return factory

    def load(self, config: dict | list | str) -> Combinator[T]:
        """
        Load a combinator chain from a config structure.

        Config format:
            # Simple predicate (no args)
            "is_admin"

            # Parameterized predicate
            {"older_than": [30]}
            {"older_than": {"days": 30}}

            # AND (all must pass)
            {"and": ["is_active", "is_admin"]}

            # OR (any must pass)
            {"or": ["is_admin", {"and": ["is_active", {"older_than": [30]}]}]}

            # NOT
            {"not": "is_banned"}

            # Sequence (transform chain)
            {"seq": ["parse_int", "double", "stringify"]}

            # Fallback (try until one succeeds)
            {"fallback": ["fetch_primary", "fetch_cache"]}
        """
        return self._parse(config)

    def load_file(self, path: str) -> Combinator[T]:
        """Load combinator chain from a JSON or YAML file."""
        import json
        from pathlib import Path

        p = Path(path)
        content = p.read_text()

        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml

                config = yaml.safe_load(content)
            except ImportError as e:
                raise ImportError(
                    "PyYAML required for YAML files: pip install pyyaml"
                ) from e
        else:
            config = json.loads(content)

        return self.load(config)

    def _parse(self, node: dict | list | str) -> Combinator[T]:
        # String: simple predicate or transform reference
        if isinstance(node, str):
            return self._resolve(node)

        # List: implicit AND
        if isinstance(node, list):
            if not node:
                return Always()
            return self._combine_and([self._parse(item) for item in node])

        # Dict: operator or parameterized call
        if isinstance(node, dict):
            if len(node) != 1:
                raise ValueError(f"Config node must have exactly one key: {node}")

            key, value = next(iter(node.items()))

            # Operators
            if key == "and":
                items = [self._parse(item) for item in value]
                return self._combine_and(items)

            elif key == "or":
                items = [self._parse(item) for item in value]
                return self._combine_or(items)

            elif key == "not":
                return ~self._parse(value)

            elif key == "seq":
                items = [self._parse(item) for item in value]
                return self._combine_seq(items)

            elif key == "fallback":
                items = [self._parse(item) for item in value]
                return self._combine_or(items)

            # Parameterized predicate/transform
            else:
                return self._resolve(key, value)

        # programmer error?
        # raise ValueError(f"Invalid config node: {node}")

    def _resolve(self, name: str, args: Any = None) -> Combinator[T]:
        """Resolve a name to a predicate or transform, optionally with args."""
        # Check predicates first, then transforms
        if name in self._predicates:
            factory_or_pred = self._predicates[name]
            if args is None:
                if callable(factory_or_pred) and not isinstance(
                    factory_or_pred, Combinator
                ):
                    raise ValueError(f"Predicate '{name}' requires arguments")
                return factory_or_pred  # type: ignore
            else:
                if isinstance(args, list):
                    return factory_or_pred(*args)  # type: ignore
                elif isinstance(args, dict):
                    return factory_or_pred(**args)  # type: ignore
                else:
                    return factory_or_pred(args)  # type: ignore

        elif name in self._transforms:
            factory_or_trans = self._transforms[name]
            if args is None:
                if callable(factory_or_trans) and not isinstance(
                    factory_or_trans, Combinator
                ):
                    raise ValueError(f"Transform '{name}' requires arguments")
                return factory_or_trans  # type: ignore
            else:
                if isinstance(args, list):
                    return factory_or_trans(*args)  # type: ignore
                elif isinstance(args, dict):
                    return factory_or_trans(**args)  # type: ignore
                else:
                    return factory_or_trans(args)  # type: ignore

        raise ValueError(f"Unknown predicate or transform: '{name}'")

    def _combine_and(self, items: list[Combinator[T]]) -> Combinator[T]:
        if not items:
            return Always()
        result = items[0]
        for item in items[1:]:
            result = result & item
        return result

    def _combine_or(self, items: list[Combinator[T]]) -> Combinator[T]:
        if not items:
            return Never()
        result = items[0]
        for item in items[1:]:
            result = result | item
        return result

    def _combine_seq(self, items: list[Combinator[T]]) -> Combinator[T]:
        if not items:
            return Always()
        result = items[0]
        for item in items[1:]:
            result = result >> item
        return result
