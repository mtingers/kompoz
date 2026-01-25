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

Example:
    from kompoz import rule, rule_args

    @rule
    def is_admin(user):
        return user.is_admin

    @rule
    def is_active(user):
        return user.is_active

    @rule_args
    def account_older_than(user, days):
        return user.account_age_days > days

    # Combine with operators
    can_access = is_admin | (is_active & account_older_than(30))

    # Use it
    ok, _ = can_access.run(user)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Matth Ingersoll"
__all__ = [
    # Core
    "Combinator",
    "Predicate",
    "PredicateFactory",
    "Transform",
    "TransformFactory",
    "Try",
    "Always",
    "Never",
    "Debug",
    "Registry",
    # Expression parsing
    "parse_expression",
    "ExpressionParser",
    # Decorators
    "rule",
    "rule_args",
    "pipe",
    "pipe_args",
    # Tracing
    "TraceHook",
    "TraceConfig",
    "use_tracing",
    "run_traced",
    "PrintHook",
    "LoggingHook",
    "OpenTelemetryHook",
    # Explanation
    "explain",
    # Validation
    "ValidationResult",
    "ValidatingCombinator",
    "ValidatingPredicate",
    "vrule",
    "vrule_args",
    # Async
    "AsyncCombinator",
    "AsyncPredicate",
    "AsyncTransform",
    "async_rule",
    "async_rule_args",
    "async_pipe",
    "async_pipe_args",
    # Caching
    "CachedPredicate",
    "use_cache",
    "cached_rule",
    # Retry
    "Retry",
    "AsyncRetry",
    # Temporal
    "during_hours",
    "on_weekdays",
    "on_days",
    "after_date",
    "before_date",
    "between_dates",
    # Aliases for backwards compatibility
    "predicate",
    "predicate_factory",
    "transform",
    "transform_factory",
]

import asyncio
import inspect
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

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

    Tracing:
        Use `with use_tracing(hook):` to trace all run() calls within scope.
        Or use `run_traced(combinator, ctx, hook)` for explicit tracing.
    """

    @abstractmethod
    def _execute(self, ctx: T) -> tuple[bool, T]:
        """Internal execution - subclasses implement this."""
        ...

    def run(self, ctx: T) -> tuple[bool, T]:
        """
        Execute the combinator and return (success, new_context).

        If tracing is enabled via use_tracing(), this will automatically
        trace the execution.
        """
        # Check for global tracing context
        hook = _trace_hook.get()
        if hook is not None:
            config = _trace_config.get()
            return _traced_run(self, ctx, hook, config, depth=0)

        return self._execute(ctx)

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

    def _execute(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = self.left._execute(ctx)
        if not ok:
            return False, ctx
        return self.right._execute(ctx)


@dataclass
class _Or(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = self.left._execute(ctx)
        if ok:
            return True, ctx
        return self.right._execute(ctx)


@dataclass
class _Not(Combinator[T]):
    inner: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = self.inner._execute(ctx)
        return not ok, ctx


@dataclass
class _Then(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        _, ctx = self.left._execute(ctx)
        return self.right._execute(ctx)


# =============================================================================
# Predicate Combinator
# =============================================================================


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


# =============================================================================
# Transform Combinator
# =============================================================================


class Transform(Combinator[T]):
    """
    A combinator that transforms the context.

    Succeeds unless an exception is raised.

    Example:
        double: Transform[int] = Transform(lambda x: x * 2, "double")
        ok, result = double.run(5)  # (True, 10)
    """

    def __init__(self, fn: Callable[[T], T], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "transform")

    def _execute(self, ctx: T) -> tuple[bool, T]:
        try:
            return True, self.fn(ctx)
        except Exception:
            return False, ctx

    def __repr__(self) -> str:
        return f"Transform({self.name})"


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

    def _execute(self, ctx: T) -> tuple[bool, T]:
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


# =============================================================================
# Tracing & Hooks
# =============================================================================


@runtime_checkable
class TraceHook(Protocol):
    """
    Protocol for trace hooks.

    Implement this to integrate with logging, OpenTelemetry, or other
    tracing systems.

    Example:
        class MyHook:
            def on_enter(self, name, ctx, depth):
                print(f"{'  ' * depth}-> {name}")
                return None  # span token

            def on_exit(self, span, name, ok, duration_ms, depth):
                status = "✓" if ok else "✗"
                print(f"{'  ' * depth}<- {name} {status} ({duration_ms:.2f}ms)")

            def on_error(self, span, name, error, duration_ms, depth):
                print(f"{'  ' * depth}<- {name} ERROR: {error}")
    """

    def on_enter(self, name: str, ctx: Any, depth: int) -> Any:
        """
        Called before running a combinator.

        Args:
            name: Name/description of the combinator
            ctx: Current context being evaluated
            depth: Nesting depth (0 = root)

        Returns:
            Span token to pass to on_exit (can be None)
        """
        ...

    def on_exit(
        self, span: Any, name: str, ok: bool, duration_ms: float, depth: int
    ) -> None:
        """
        Called after a combinator completes.

        Args:
            span: Token returned from on_enter
            name: Name/description of the combinator
            ok: Whether the combinator succeeded
            duration_ms: Execution time in milliseconds
            depth: Nesting depth
        """
        ...

    def on_error(
        self, span: Any, name: str, error: Exception, duration_ms: float, depth: int
    ) -> None:
        """
        Optional: Called if a combinator raises an exception.

        Args:
            span: Token returned from on_enter
            name: Name/description of the combinator
            error: The exception that was raised
            duration_ms: Execution time in milliseconds
            depth: Nesting depth
        """
        ...


@dataclass
class TraceConfig:
    """
    Configuration for tracing behavior.

    Attributes:
        nested: If True, trace child combinators (AND, OR, NOT children)
        max_depth: Maximum depth to trace (None = unlimited)
        include_leaf_only: If True, only trace leaf combinators (Predicate, Transform)
    """

    nested: bool = True
    max_depth: int | None = None
    include_leaf_only: bool = False


# Context variables for global tracing
_trace_hook: ContextVar[TraceHook | None] = ContextVar("trace_hook", default=None)
_trace_config: ContextVar[TraceConfig] = ContextVar(
    "trace_config", default=TraceConfig()
)


@contextmanager
def use_tracing(hook: TraceHook, config: TraceConfig | None = None):
    """
    Context manager to enable tracing for all rule executions in scope.

    Args:
        hook: TraceHook implementation to receive trace events
        config: Optional TraceConfig to customize tracing behavior

    Example:
        with use_tracing(LoggingHook(logger)):
            rule.run(user)  # This will be traced

        # Or with custom config
        with use_tracing(PrintHook(), TraceConfig(max_depth=2)):
            complex_rule.run(data)
    """
    old_hook = _trace_hook.get()
    old_config = _trace_config.get()

    _trace_hook.set(hook)
    _trace_config.set(config or TraceConfig())

    try:
        yield
    finally:
        _trace_hook.set(old_hook)
        _trace_config.set(old_config)


def _get_combinator_name(combinator: Combinator) -> str:
    """Get a human-readable name for a combinator."""
    if isinstance(combinator, Predicate):
        return f"Predicate({combinator.name})"
    if isinstance(combinator, Transform):
        return f"Transform({combinator.name})"
    if isinstance(combinator, _And):
        return "AND"
    if isinstance(combinator, _Or):
        return "OR"
    if isinstance(combinator, _Not):
        return "NOT"
    if isinstance(combinator, _Then):
        return "THEN"
    if isinstance(combinator, Always):
        return "Always"
    if isinstance(combinator, Never):
        return "Never"
    if isinstance(combinator, Debug):
        return f"Debug({combinator.label})"
    if isinstance(combinator, Try):
        return f"Try({combinator.name})"
    return repr(combinator)


def _traced_run(
    combinator: Combinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig,
    depth: int = 0,
) -> tuple[bool, T]:
    """Execute a combinator with tracing."""

    # Check depth limit
    if config.max_depth is not None and depth > config.max_depth:
        return combinator._execute(ctx)

    name = _get_combinator_name(combinator)
    is_composite = isinstance(combinator, (_And, _Or, _Not, _Then))

    # Skip composite combinators if leaf_only mode
    if config.include_leaf_only and is_composite:
        # Still recurse into children with tracing
        if config.nested:
            return _traced_run_inner(combinator, ctx, hook, config, depth)
        return combinator._execute(ctx)

    # Call on_enter
    span = hook.on_enter(name, ctx, depth)
    start = time.perf_counter()

    try:
        if config.nested and is_composite:
            ok, result = _traced_run_inner(combinator, ctx, hook, config, depth)
        else:
            ok, result = combinator._execute(ctx)

        duration_ms = (time.perf_counter() - start) * 1000
        hook.on_exit(span, name, ok, duration_ms, depth)
        return ok, result

    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        # Call on_error - it's part of the TraceHook protocol
        hook.on_error(span, name, e, duration_ms, depth)
        raise


def _traced_run_inner(
    combinator: Combinator[T], ctx: T, hook: TraceHook, config: TraceConfig, depth: int
) -> tuple[bool, T]:
    """Handle tracing for composite combinators."""

    if isinstance(combinator, _And):
        ok, ctx = _traced_run(combinator.left, ctx, hook, config, depth + 1)
        if not ok:
            return False, ctx
        return _traced_run(combinator.right, ctx, hook, config, depth + 1)

    if isinstance(combinator, _Or):
        ok, ctx = _traced_run(combinator.left, ctx, hook, config, depth + 1)
        if ok:
            return True, ctx
        return _traced_run(combinator.right, ctx, hook, config, depth + 1)

    if isinstance(combinator, _Not):
        ok, result = _traced_run(combinator.inner, ctx, hook, config, depth + 1)
        return not ok, result

    if isinstance(combinator, _Then):
        _, ctx = _traced_run(combinator.left, ctx, hook, config, depth + 1)
        return _traced_run(combinator.right, ctx, hook, config, depth + 1)

    # Fallback for unknown composite types
    return combinator._execute(ctx)


def run_traced(
    combinator: Combinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig | None = None,
) -> tuple[bool, T]:
    """
    Run a combinator with explicit tracing.

    Args:
        combinator: The combinator to run
        ctx: Context to evaluate
        hook: TraceHook to receive events
        config: Optional TraceConfig

    Returns:
        Tuple of (success, result_context)

    Example:
        ok, result = run_traced(rule, user, PrintHook())
    """
    return _traced_run(combinator, ctx, hook, config or TraceConfig())


# =============================================================================
# Built-in Trace Hooks
# =============================================================================


class PrintHook:
    """
    Simple trace hook that prints to stdout.

    Example:
        with use_tracing(PrintHook()):
            rule.run(user)

        # Output:
        # -> Predicate(is_admin)
        # <- Predicate(is_admin) ✗ (0.02ms)
        # -> Predicate(is_active)
        # <- Predicate(is_active) ✓ (0.01ms)
    """

    def __init__(self, indent: str = "  ", show_ctx: bool = False):
        self.indent = indent
        self.show_ctx = show_ctx

    def on_enter(self, name: str, ctx: Any, depth: int) -> float:
        prefix = self.indent * depth
        if self.show_ctx:
            print(f"{prefix}-> {name} | ctx={ctx}")
        else:
            print(f"{prefix}-> {name}")
        return time.perf_counter()

    def on_exit(
        self, span: float, name: str, ok: bool, duration_ms: float, depth: int
    ) -> None:
        prefix = self.indent * depth
        status = "✓" if ok else "✗"
        print(f"{prefix}<- {name} {status} ({duration_ms:.2f}ms)")

    def on_error(
        self, span: float, name: str, error: Exception, duration_ms: float, depth: int
    ) -> None:
        prefix = self.indent * depth
        print(f"{prefix}<- {name} ERROR: {error} ({duration_ms:.2f}ms)")


class LoggingHook:
    """
    Trace hook that logs to a Python logger.

    Example:
        import logging
        logger = logging.getLogger("kompoz")

        with use_tracing(LoggingHook(logger)):
            rule.run(user)
    """

    def __init__(self, logger, level: int = 10):  # 10 = DEBUG
        self.logger = logger
        self.level = level

    def on_enter(self, name: str, ctx: Any, depth: int) -> dict:
        span = {"name": name, "depth": depth, "start": time.perf_counter()}
        self.logger.log(self.level, f"[ENTER] {name} (depth={depth})")
        return span

    def on_exit(
        self, span: dict, name: str, ok: bool, duration_ms: float, depth: int
    ) -> None:
        status = "OK" if ok else "FAIL"
        self.logger.log(self.level, f"[EXIT] {name} -> {status} ({duration_ms:.2f}ms)")

    def on_error(
        self, span: dict, name: str, error: Exception, duration_ms: float, depth: int
    ) -> None:
        self.logger.error(f"[ERROR] {name} -> {error} ({duration_ms:.2f}ms)")


class OpenTelemetryHook:
    """
    Trace hook for OpenTelemetry integration.

    Requires: pip install opentelemetry-api

    Example:
        from opentelemetry import trace
        tracer = trace.get_tracer("kompoz")

        with use_tracing(OpenTelemetryHook(tracer)):
            rule.run(user)
    """

    def __init__(self, tracer, span_prefix: str = "kompoz"):
        self.tracer = tracer
        self.span_prefix = span_prefix

    def on_enter(self, name: str, ctx: Any, depth: int) -> Any:
        span_name = f"{self.span_prefix}.{name}"
        span = self.tracer.start_span(span_name)
        span.set_attribute("kompoz.depth", depth)
        span.set_attribute("kompoz.combinator", name)
        return span

    def on_exit(
        self, span: Any, name: str, ok: bool, duration_ms: float, depth: int
    ) -> None:
        span.set_attribute("kompoz.success", ok)
        span.set_attribute("kompoz.duration_ms", duration_ms)
        if not ok:
            span.set_status(self.tracer.Status(self.tracer.StatusCode.ERROR))
        span.end()

    def on_error(
        self, span: Any, name: str, error: Exception, duration_ms: float, depth: int
    ) -> None:
        span.set_attribute("kompoz.success", False)
        span.set_attribute("kompoz.duration_ms", duration_ms)
        span.record_exception(error)
        span.set_status(self.tracer.Status(self.tracer.StatusCode.ERROR, str(error)))
        span.end()


# =============================================================================
# Explain Function
# =============================================================================


def explain(combinator: Combinator, verbose: bool = False) -> str:
    """
    Generate a plain English explanation of what a rule does.

    Args:
        combinator: The rule to explain
        verbose: If True, include more detail

    Returns:
        Human-readable explanation string

    Example:
        rule = is_admin | (is_active & ~is_banned)
        print(explain(rule))

        # Output:
        # Check passes if ANY of:
        #   • is_admin
        #   • ALL of:
        #     • is_active
        #     • NOT: is_banned
    """
    return _explain(combinator, depth=0, verbose=verbose)


def _explain(combinator: Combinator, depth: int, verbose: bool) -> str:
    """Recursive explain implementation."""
    indent = "  " * depth
    bullet = "• " if depth > 0 else ""

    if isinstance(combinator, Predicate):
        return f"{indent}{bullet}Check: {combinator.name}"

    if isinstance(combinator, Transform):
        return f"{indent}{bullet}Transform: {combinator.name}"

    if isinstance(combinator, _And):
        # Collect all AND children (flatten nested ANDs)
        children = _collect_chain(combinator, _And, "left", "right")
        child_explains = [_explain(c, depth + 1, verbose) for c in children]

        if depth == 0:
            header = "Check passes if ALL of:"
        else:
            header = f"{indent}{bullet}ALL of:"

        return header + "\n" + "\n".join(child_explains)

    if isinstance(combinator, _Or):
        # Collect all OR children (flatten nested ORs)
        children = _collect_chain(combinator, _Or, "left", "right")
        child_explains = [_explain(c, depth + 1, verbose) for c in children]

        if depth == 0:
            header = "Check passes if ANY of:"
        else:
            header = f"{indent}{bullet}ANY of:"

        return header + "\n" + "\n".join(child_explains)

    if isinstance(combinator, _Not):
        inner = _explain_inline(combinator.inner)
        return f"{indent}{bullet}NOT: {inner}"

    if isinstance(combinator, _Then):
        left = _explain(combinator.left, depth + 1, verbose)
        right = _explain(combinator.right, depth + 1, verbose)

        if depth == 0:
            header = "Execute in sequence:"
        else:
            header = f"{indent}{bullet}Execute in sequence:"

        return f"{header}\n{left}\n{indent}  THEN:\n{right}"

    if isinstance(combinator, Always):
        return f"{indent}{bullet}Always pass"

    if isinstance(combinator, Never):
        return f"{indent}{bullet}Always fail"

    if isinstance(combinator, Debug):
        return f"{indent}{bullet}Debug: {combinator.label}"

    if isinstance(combinator, Try):
        return f"{indent}{bullet}Try: {combinator.name} (catch errors)"

    # Fallback
    return f"{indent}{bullet}{repr(combinator)}"


def _explain_inline(combinator: Combinator) -> str:
    """Get a short inline explanation for NOT children."""
    if isinstance(combinator, Predicate):
        return combinator.name
    if isinstance(combinator, Transform):
        return combinator.name
    if isinstance(combinator, _And):
        children = _collect_chain(combinator, _And, "left", "right")
        parts = [_explain_inline(c) for c in children]
        return f"({' & '.join(parts)})"
    if isinstance(combinator, _Or):
        children = _collect_chain(combinator, _Or, "left", "right")
        parts = [_explain_inline(c) for c in children]
        return f"({' | '.join(parts)})"
    if isinstance(combinator, _Not):
        return f"~{_explain_inline(combinator.inner)}"
    return repr(combinator)


def _collect_chain(
    combinator: Combinator, cls: type, left_attr: str, right_attr: str
) -> list:
    """Collect chained combinators of the same type."""
    result = []

    def collect(c):
        if isinstance(c, cls):
            collect(getattr(c, left_attr))
            collect(getattr(c, right_attr))
        else:
            result.append(c)

    collect(combinator)
    return result


# =============================================================================
# Validation with Error Messages
# =============================================================================


@dataclass
class ValidationResult:
    """
    Result of validation with error messages.

    Attributes:
        ok: Whether all checks passed
        errors: List of error messages from failed checks
        ctx: The (possibly transformed) context
    """

    ok: bool
    errors: list[str]
    ctx: Any

    def __bool__(self) -> bool:
        return self.ok

    def raise_if_invalid(self, exception_class: type = ValueError) -> None:
        """Raise an exception if validation failed."""
        if not self.ok:
            raise exception_class("; ".join(self.errors))


class ValidatingCombinator(Combinator[T]):
    """
    Base class for combinators that support validation with error messages.

    Subclasses must implement the validate() method.
    """

    @abstractmethod
    def validate(self, ctx: T) -> ValidationResult:
        """Run validation and return result with errors."""
        ...

    def __and__(self, other: Combinator[T]) -> ValidatingCombinator[T]:
        """Override & to create validating AND."""
        return _ValidatingAnd(self, other)

    def __or__(self, other: Combinator[T]) -> ValidatingCombinator[T]:
        """Override | to create validating OR."""
        return _ValidatingOr(self, other)


class ValidatingPredicate(ValidatingCombinator[T]):
    """
    A predicate that provides an error message on failure.

    Example:
        @vrule(error="User must be an admin")
        def is_admin(user):
            return user.is_admin

        result = is_admin.validate(user)
        if not result.ok:
            print(result.errors)  # ["User must be an admin"]
    """

    def __init__(
        self,
        fn: Callable[[T], bool],
        name: str | None = None,
        error: str | Callable[[T], str] | None = None,
    ):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "predicate")
        self._error = error

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return self.fn(ctx), ctx

    def get_error(self, ctx: T) -> str:
        """Get the error message for this predicate."""
        if self._error is None:
            return f"Check failed: {self.name}"
        if callable(self._error):
            return self._error(ctx)
        # String interpolation with ctx
        try:
            return self._error.format(ctx=ctx)
        except (KeyError, AttributeError):
            return self._error

    def validate(self, ctx: T) -> ValidationResult:
        """Run validation and return result with errors."""
        ok, result = self._execute(ctx)
        errors = [] if ok else [self.get_error(ctx)]
        return ValidationResult(ok=ok, errors=errors, ctx=result)

    def __repr__(self) -> str:
        return f"ValidatingPredicate({self.name})"


class _ValidatingAnd(ValidatingCombinator[T]):
    """AND combinator that collects all validation errors."""

    def __init__(self, left: Combinator[T], right: Combinator[T]):
        self.left = left
        self.right = right

    def _execute(self, ctx: T) -> tuple[bool, T]:
        ok1, ctx = self.left._execute(ctx)
        if not ok1:
            return False, ctx
        return self.right._execute(ctx)

    def validate(self, ctx: T) -> ValidationResult:
        """Validate both sides and collect all errors."""
        errors: list[str] = []

        # Validate left
        if isinstance(self.left, ValidatingCombinator):
            left_result = self.left.validate(ctx)
            errors.extend(left_result.errors)
            ctx = left_result.ctx
        else:
            ok, ctx = self.left._execute(ctx)
            if not ok:
                errors.append(f"Check failed: {_get_combinator_name(self.left)}")

        # Validate right (even if left failed, to collect all errors)
        if isinstance(self.right, ValidatingCombinator):
            right_result = self.right.validate(ctx)
            errors.extend(right_result.errors)
            ctx = right_result.ctx
        else:
            ok, ctx = self.right._execute(ctx)
            if not ok:
                errors.append(f"Check failed: {_get_combinator_name(self.right)}")

        return ValidationResult(ok=len(errors) == 0, errors=errors, ctx=ctx)


class _ValidatingOr(ValidatingCombinator[T]):
    """OR combinator for validation - passes if any succeeds."""

    def __init__(self, left: Combinator[T], right: Combinator[T]):
        self.left = left
        self.right = right

    def _execute(self, ctx: T) -> tuple[bool, T]:
        ok1, result1 = self.left._execute(ctx)
        if ok1:
            return True, result1
        return self.right._execute(ctx)

    def validate(self, ctx: T) -> ValidationResult:
        """Validate - passes if either side passes."""
        # Try left first
        if isinstance(self.left, ValidatingCombinator):
            left_result = self.left.validate(ctx)
            if left_result.ok:
                return left_result
        else:
            ok, result = self.left._execute(ctx)
            if ok:
                return ValidationResult(ok=True, errors=[], ctx=result)

        # Left failed, try right
        if isinstance(self.right, ValidatingCombinator):
            return self.right.validate(ctx)
        else:
            ok, result = self.right._execute(ctx)
            if ok:
                return ValidationResult(ok=True, errors=[], ctx=result)
            return ValidationResult(
                ok=False,
                errors=[f"Check failed: {_get_combinator_name(self.right)}"],
                ctx=result,
            )


@overload
def vrule(
    fn: Callable[[T], bool], *, error: str | Callable[[T], str] | None = None
) -> ValidatingPredicate[T]: ...


@overload
def vrule(
    fn: None = None, *, error: str | Callable[[T], str] | None = None
) -> Callable[[Callable[[T], bool]], ValidatingPredicate[T]]: ...


def vrule(
    fn: Callable[[T], bool] | None = None,
    *,
    error: str | Callable[[T], str] | None = None,
) -> ValidatingPredicate[T] | Callable[[Callable[[T], bool]], ValidatingPredicate[T]]:
    """
    Decorator to create a validating rule with an error message.

    Example:
        @vrule(error="User {ctx.name} must be an admin")
        def is_admin(user):
            return user.is_admin

        @vrule(error=lambda u: f"{u.name} is banned!")
        def is_not_banned(user):
            return not user.is_banned

        result = is_admin.validate(user)
        result = (is_admin & is_not_banned).validate(user)  # Collects all errors
    """

    def decorator(f: Callable[[T], bool]) -> ValidatingPredicate[T]:
        return ValidatingPredicate(f, f.__name__, error)

    if fn is not None:
        return decorator(fn)
    return decorator


@overload
def vrule_args(
    fn: Callable[..., bool], *, error: str | Callable[..., str] | None = None
) -> Callable[..., ValidatingPredicate[Any]]: ...


@overload
def vrule_args(
    fn: None = None, *, error: str | Callable[..., str] | None = None
) -> Callable[[Callable[..., bool]], Callable[..., ValidatingPredicate[Any]]]: ...


def vrule_args(
    fn: Callable[..., bool] | None = None,
    *,
    error: str | Callable[..., str] | None = None,
) -> (
    Callable[..., ValidatingPredicate[Any]]
    | Callable[[Callable[..., bool]], Callable[..., ValidatingPredicate[Any]]]
):
    """
    Decorator to create a parameterized validating rule factory.

    Example:
        @vrule_args(error="Account must be older than {arg0} days")
        def account_older_than(user, days):
            return user.account_age_days > days

        result = account_older_than(30).validate(user)
    """

    def decorator(f: Callable[..., bool]) -> Callable[..., ValidatingPredicate[Any]]:
        def factory(*args: Any, **kwargs: Any) -> ValidatingPredicate[Any]:
            name = f"{f.__name__}({', '.join(map(repr, args))})"

            # Create error message with args available
            err_msg: str | Callable[[Any], str] | None
            if error is None:
                err_msg = None
            elif callable(error):
                # Capture the callable in a local variable for type narrowing
                error_fn: Callable[..., str] = error

                def make_error_fn(ctx: Any) -> str:
                    return error_fn(ctx, *args, **kwargs)

                err_msg = make_error_fn
            else:
                # Format the error string with args
                format_kwargs = {f"arg{i}": v for i, v in enumerate(args)}
                format_kwargs.update(kwargs)
                try:
                    err_msg = error.format(**format_kwargs)
                except (KeyError, IndexError):
                    err_msg = error

            def predicate_fn(ctx: Any) -> bool:
                return f(ctx, *args, **kwargs)

            return ValidatingPredicate(predicate_fn, name, err_msg)

        factory.__name__ = f.__name__
        return factory

    if fn is not None:
        return decorator(fn)
    return decorator


# =============================================================================
# Async Support
# =============================================================================


class AsyncCombinator(ABC, Generic[T]):
    """
    Base class for async combinators.

    Similar to Combinator but uses async/await.
    """

    @abstractmethod
    async def run(self, ctx: T) -> tuple[bool, T]:
        """Execute the combinator asynchronously."""
        ...

    def __and__(self, other: AsyncCombinator[T]) -> AsyncCombinator[T]:
        return _AsyncAnd(self, other)

    def __or__(self, other: AsyncCombinator[T]) -> AsyncCombinator[T]:
        return _AsyncOr(self, other)

    def __invert__(self) -> AsyncCombinator[T]:
        return _AsyncNot(self)

    def __rshift__(self, other: AsyncCombinator[T]) -> AsyncCombinator[T]:
        return _AsyncThen(self, other)

    async def __call__(self, ctx: T) -> tuple[bool, T]:
        return await self.run(ctx)


@dataclass
class _AsyncAnd(AsyncCombinator[T]):
    left: AsyncCombinator[T]
    right: AsyncCombinator[T]

    async def run(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = await self.left.run(ctx)
        if not ok:
            return False, ctx
        return await self.right.run(ctx)


@dataclass
class _AsyncOr(AsyncCombinator[T]):
    left: AsyncCombinator[T]
    right: AsyncCombinator[T]

    async def run(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = await self.left.run(ctx)
        if ok:
            return True, ctx
        return await self.right.run(ctx)


@dataclass
class _AsyncNot(AsyncCombinator[T]):
    inner: AsyncCombinator[T]

    async def run(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = await self.inner.run(ctx)
        return not ok, ctx


@dataclass
class _AsyncThen(AsyncCombinator[T]):
    left: AsyncCombinator[T]
    right: AsyncCombinator[T]

    async def run(self, ctx: T) -> tuple[bool, T]:
        _, ctx = await self.left.run(ctx)
        return await self.right.run(ctx)


class AsyncPredicate(AsyncCombinator[T]):
    """
    An async predicate that checks a condition.

    Example:
        @async_rule
        async def has_permission(user):
            return await db.check_permission(user.id)
    """

    def __init__(self, fn: Callable[[T], Any], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "async_predicate")

    async def run(self, ctx: T) -> tuple[bool, T]:
        result = await self.fn(ctx)
        return bool(result), ctx

    def __repr__(self) -> str:
        return f"AsyncPredicate({self.name})"


class AsyncPredicateFactory(Generic[T]):
    """Factory for parameterized async predicates."""

    def __init__(self, fn: Callable[..., Any], name: str):
        self._fn = fn
        self._name = name
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> AsyncPredicate[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return AsyncPredicate(lambda ctx: self._fn(ctx, *args, **kwargs), name)

    def __repr__(self) -> str:
        return f"AsyncPredicateFactory({self._name})"


class AsyncTransform(AsyncCombinator[T]):
    """
    An async transform that modifies context.

    Example:
        @async_pipe
        async def fetch_profile(user):
            user.profile = await api.get_profile(user.id)
            return user
    """

    def __init__(self, fn: Callable[[T], Any], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "async_transform")

    async def run(self, ctx: T) -> tuple[bool, T]:
        try:
            result = await self.fn(ctx)
            return True, result
        except Exception:
            return False, ctx

    def __repr__(self) -> str:
        return f"AsyncTransform({self.name})"


class AsyncTransformFactory(Generic[T]):
    """Factory for parameterized async transforms."""

    def __init__(self, fn: Callable[..., Any], name: str):
        self._fn = fn
        self._name = name
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> AsyncTransform[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return AsyncTransform(lambda ctx: self._fn(ctx, *args, **kwargs), name)

    def __repr__(self) -> str:
        return f"AsyncTransformFactory({self._name})"


def async_rule(fn: Callable[[T], Any]) -> AsyncPredicate[T]:
    """
    Decorator to create an async predicate.

    Example:
        @async_rule
        async def has_permission(user):
            return await db.check_permission(user.id)

        ok, _ = await has_permission.run(user)
    """
    return AsyncPredicate(fn, fn.__name__)


def async_rule_args(fn: Callable[..., Any]) -> AsyncPredicateFactory[Any]:
    """
    Decorator to create a parameterized async predicate factory.

    Example:
        @async_rule_args
        async def has_role(user, role):
            return await db.check_role(user.id, role)

        ok, _ = await has_role("admin").run(user)
    """
    return AsyncPredicateFactory(fn, fn.__name__)


def async_pipe(fn: Callable[[T], Any]) -> AsyncTransform[T]:
    """
    Decorator to create an async transform.

    Example:
        @async_pipe
        async def enrich_user(user):
            user.profile = await api.get_profile(user.id)
            return user
    """
    return AsyncTransform(fn, fn.__name__)


def async_pipe_args(fn: Callable[..., Any]) -> AsyncTransformFactory[Any]:
    """
    Decorator to create a parameterized async transform factory.

    Example:
        @async_pipe_args
        async def fetch_data(ctx, endpoint):
            ctx.data = await api.get(endpoint)
            return ctx
    """
    return AsyncTransformFactory(fn, fn.__name__)


# =============================================================================
# Caching / Memoization
# =============================================================================


# Context variable for caching scope
_cache_store: ContextVar[dict[str, tuple[bool, Any]] | None] = ContextVar(
    "cache_store", default=None
)


@contextmanager
def use_cache():
    """
    Context manager to enable caching for all cached rules in scope.

    Example:
        with use_cache():
            # Same predicate called multiple times will only execute once
            rule.run(user)
            rule.run(user)  # Uses cached result
    """
    old_cache = _cache_store.get()
    _cache_store.set({})
    try:
        yield
    finally:
        _cache_store.set(old_cache)


class CachedPredicate(Combinator[T]):
    """
    A predicate that caches its result within a use_cache() scope.

    The cache key is based on the predicate name and the context's id or hash.
    """

    def __init__(
        self,
        fn: Callable[[T], bool],
        name: str | None = None,
        key_fn: Callable[[T], Any] | None = None,
    ):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "cached_predicate")
        self.key_fn = key_fn or id  # Default to using object id

    def _get_cache_key(self, ctx: T) -> str:
        """Generate a cache key for this context."""
        ctx_key = self.key_fn(ctx)
        return f"{self.name}:{ctx_key}"

    def _execute(self, ctx: T) -> tuple[bool, T]:
        cache = _cache_store.get()

        if cache is not None:
            key = self._get_cache_key(ctx)
            if key in cache:
                return cache[key]

            result = self.fn(ctx), ctx
            cache[key] = result
            return result

        return self.fn(ctx), ctx

    def __repr__(self) -> str:
        return f"CachedPredicate({self.name})"


class CachedPredicateFactory(Generic[T]):
    """Factory for parameterized cached predicates."""

    def __init__(
        self,
        fn: Callable[..., bool],
        name: str,
        key_fn: Callable[[T], Any] | None = None,
    ):
        self._fn = fn
        self._name = name
        self._key_fn = key_fn
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> CachedPredicate[T]:
        name = f"{self._name}({', '.join(map(repr, args))})"
        return CachedPredicate(
            lambda ctx: self._fn(ctx, *args, **kwargs), name, self._key_fn
        )

    def __repr__(self) -> str:
        return f"CachedPredicateFactory({self._name})"


@overload
def cached_rule(
    fn: Callable[[T], bool], *, key: Callable[[T], Any] | None = None
) -> CachedPredicate[T]: ...


@overload
def cached_rule(
    fn: None = None, *, key: Callable[[T], Any] | None = None
) -> Callable[[Callable[[T], bool]], CachedPredicate[T]]: ...


def cached_rule(
    fn: Callable[[T], bool] | None = None, *, key: Callable[[T], Any] | None = None
) -> CachedPredicate[T] | Callable[[Callable[[T], bool]], CachedPredicate[T]]:
    """
    Decorator to create a cached rule.

    Results are cached within a use_cache() scope.

    Example:
        @cached_rule
        def expensive_check(user):
            # This will only run once per user within use_cache()
            return slow_database_query(user.id)

        @cached_rule(key=lambda u: u.id)
        def check_by_id(user):
            return api_call(user.id)

        with use_cache():
            rule.run(user)
            rule.run(user)  # Uses cached result
    """

    def decorator(f: Callable[[T], bool]) -> CachedPredicate[T]:
        return CachedPredicate(f, f.__name__, key)

    if fn is not None:
        return decorator(fn)
    return decorator


# =============================================================================
# Retry Logic
# =============================================================================


class Retry(Combinator[T]):
    """
    A combinator that retries on failure with configurable backoff.

    Example:
        # Retry up to 3 times with exponential backoff
        fetch = Retry(fetch_from_api, max_attempts=3, backoff=1.0, exponential=True)

        # Retry with jitter to avoid thundering herd
        fetch = Retry(fetch_from_api, max_attempts=5, backoff=0.5, jitter=0.1)
    """

    def __init__(
        self,
        inner: Combinator[T] | Callable[[T], T],
        max_attempts: int = 3,
        backoff: float = 0.0,
        exponential: bool = False,
        jitter: float = 0.0,
        name: str | None = None,
    ):
        if isinstance(inner, Combinator):
            self.inner = inner
            self.name = name or repr(inner)
        else:
            self.inner = Transform(inner, getattr(inner, "__name__", "retry_fn"))
            self.name = name or getattr(inner, "__name__", "retry")

        self.max_attempts = max_attempts
        self.backoff = backoff
        self.exponential = exponential
        self.jitter = jitter

    def _get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff <= 0:
            return 0

        if self.exponential:
            delay = self.backoff * (2**attempt)
        else:
            delay = self.backoff

        if self.jitter > 0:
            delay += random.uniform(0, self.jitter)

        return delay

    def _execute(self, ctx: T) -> tuple[bool, T]:
        last_ctx = ctx

        for attempt in range(self.max_attempts):
            try:
                ok, result = self.inner._execute(last_ctx)
                if ok:
                    return True, result
                last_ctx = result
            except Exception:
                # Continue to retry on exception
                pass

            # Don't sleep after last attempt
            if attempt < self.max_attempts - 1:
                delay = self._get_delay(attempt)
                if delay > 0:
                    time.sleep(delay)

        return False, last_ctx

    def __repr__(self) -> str:
        return f"Retry({self.name}, max_attempts={self.max_attempts})"


class AsyncRetry(AsyncCombinator[T]):
    """
    Async version of Retry combinator.

    Example:
        fetch = AsyncRetry(fetch_from_api, max_attempts=3, backoff=1.0)
        ok, result = await fetch.run(request)
    """

    def __init__(
        self,
        inner: AsyncCombinator[T] | Callable[[T], Any],
        max_attempts: int = 3,
        backoff: float = 0.0,
        exponential: bool = False,
        jitter: float = 0.0,
        name: str | None = None,
    ):
        if isinstance(inner, AsyncCombinator):
            self.inner = inner
            self.name = name or repr(inner)
        else:
            self.inner = AsyncTransform(inner, getattr(inner, "__name__", "retry_fn"))
            self.name = name or getattr(inner, "__name__", "retry")

        self.max_attempts = max_attempts
        self.backoff = backoff
        self.exponential = exponential
        self.jitter = jitter

    def _get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.backoff <= 0:
            return 0

        if self.exponential:
            delay = self.backoff * (2**attempt)
        else:
            delay = self.backoff

        if self.jitter > 0:
            delay += random.uniform(0, self.jitter)

        return delay

    async def run(self, ctx: T) -> tuple[bool, T]:
        last_ctx = ctx
        last_error = None

        for attempt in range(self.max_attempts):
            try:
                ok, result = await self.inner.run(last_ctx)
                if ok:
                    return True, result
                last_ctx = result
            except Exception as e:
                last_error = e
                # Continue to retry

            # Don't sleep after last attempt
            if attempt < self.max_attempts - 1:
                delay = self._get_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

        return False, last_ctx

    def __repr__(self) -> str:
        return f"AsyncRetry({self.name}, max_attempts={self.max_attempts})"


# =============================================================================
# Time-Based / Temporal Predicates
# =============================================================================


class during_hours(Combinator[T]):
    """
    Predicate that passes only during specified hours.

    Example:
        # Only allow during business hours (9 AM to 5 PM)
        business_hours = during_hours(9, 17)

        # With timezone (requires datetime context or system time)
        trading_hours = during_hours(9, 16, tz="America/New_York")
    """

    def __init__(self, start_hour: int, end_hour: int, tz: str | None = None):
        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
            raise ValueError("Hours must be 0-23")

        self.start_hour = start_hour
        self.end_hour = end_hour
        self.tz = tz

    def _get_current_hour(self) -> int:
        """Get current hour, optionally in specified timezone."""
        if self.tz:
            try:
                from zoneinfo import ZoneInfo

                now = datetime.now(ZoneInfo(self.tz))
            except ImportError:
                # Fallback for Python < 3.9
                now = datetime.now()
        else:
            now = datetime.now()
        return now.hour

    def _execute(self, ctx: T) -> tuple[bool, T]:
        hour = self._get_current_hour()

        if self.start_hour <= self.end_hour:
            # Normal range (e.g., 9 to 17)
            ok = self.start_hour <= hour < self.end_hour
        else:
            # Overnight range (e.g., 22 to 6)
            ok = hour >= self.start_hour or hour < self.end_hour

        return ok, ctx

    def __repr__(self) -> str:
        return f"during_hours({self.start_hour}, {self.end_hour})"


class on_weekdays(Combinator[T]):
    """
    Predicate that passes only on weekdays (Monday-Friday).

    Example:
        weekday_only = on_weekdays()
        can_trade = is_active & on_weekdays() & during_hours(9, 16)
    """

    def _execute(self, ctx: T) -> tuple[bool, T]:
        # Monday = 0, Sunday = 6
        weekday = datetime.now().weekday()
        return weekday < 5, ctx

    def __repr__(self) -> str:
        return "on_weekdays()"


class on_days(Combinator[T]):
    """
    Predicate that passes only on specified days of the week.

    Example:
        # Monday, Wednesday, Friday
        mwf = on_days(0, 2, 4)

        # Weekends only
        weekends = on_days(5, 6)
    """

    def __init__(self, *days: int):
        """
        Args:
            days: Day numbers where Monday=0, Sunday=6
        """
        for d in days:
            if not 0 <= d <= 6:
                raise ValueError("Days must be 0-6 (Monday=0, Sunday=6)")
        self.days = set(days)

    def _execute(self, ctx: T) -> tuple[bool, T]:
        weekday = datetime.now().weekday()
        return weekday in self.days, ctx

    def __repr__(self) -> str:
        return f"on_days({', '.join(map(str, sorted(self.days)))})"


class after_date(Combinator[T]):
    """
    Predicate that passes only after a specified date.

    Example:
        # Feature available after launch
        post_launch = after_date(2024, 6, 1)

        # Using date object
        from datetime import date
        post_launch = after_date(date(2024, 6, 1))
    """

    def __init__(
        self, year_or_date: int | date, month: int | None = None, day: int | None = None
    ):
        if isinstance(year_or_date, date):
            self.date = year_or_date
        else:
            if month is None or day is None:
                raise ValueError("Must provide month and day with year")
            self.date = date(year_or_date, month, day)

    def _execute(self, ctx: T) -> tuple[bool, T]:
        today = date.today()
        return today > self.date, ctx

    def __repr__(self) -> str:
        return f"after_date({self.date})"


class before_date(Combinator[T]):
    """
    Predicate that passes only before a specified date.

    Example:
        # Promo ends on specific date
        promo_active = before_date(2024, 12, 31)
    """

    def __init__(
        self, year_or_date: int | date, month: int | None = None, day: int | None = None
    ):
        if isinstance(year_or_date, date):
            self.date = year_or_date
        else:
            if month is None or day is None:
                raise ValueError("Must provide month and day with year")
            self.date = date(year_or_date, month, day)

    def _execute(self, ctx: T) -> tuple[bool, T]:
        today = date.today()
        return today < self.date, ctx

    def __repr__(self) -> str:
        return f"before_date({self.date})"


class between_dates(Combinator[T]):
    """
    Predicate that passes only between two dates (inclusive).

    Example:
        # Holiday promotion
        holiday_promo = between_dates(date(2024, 12, 20), date(2024, 12, 31))

        # Q1 only
        q1 = between_dates(2024, 1, 1, 2024, 3, 31)
    """

    start_date: date
    end_date: date

    def __init__(
        self,
        start: date | int,
        end_or_start_month: date | int,
        start_day: int | None = None,
        end_year: int | None = None,
        end_month: int | None = None,
        end_day: int | None = None,
    ):
        if isinstance(start, date) and isinstance(end_or_start_month, date):
            self.start_date = start
            self.end_date = end_or_start_month
        elif isinstance(start, int) and isinstance(end_or_start_month, int):
            # Constructor: between_dates(y1, m1, d1, y2, m2, d2)
            if (
                start_day is None
                or end_year is None
                or end_month is None
                or end_day is None
            ):
                raise ValueError("Must provide all 6 arguments for date range")
            self.start_date = date(start, end_or_start_month, start_day)
            self.end_date = date(end_year, end_month, end_day)
        else:
            raise TypeError("Arguments must be either two date objects or six integers")

    def _execute(self, ctx: T) -> tuple[bool, T]:
        today = date.today()
        return self.start_date <= today <= self.end_date, ctx

    def __repr__(self) -> str:
        return f"between_dates({self.start_date}, {self.end_date})"


# =============================================================================
# Registry & Expression Parser
# =============================================================================


class ExpressionParser:
    """
    Parser for human-readable rule expressions.

    Supports two equivalent syntaxes:
        Symbol style:  is_admin & ~is_banned & account_older_than(30)
        Word style:    is_admin AND NOT is_banned AND account_older_than(30)

    Operators (by precedence, lowest to highest):
        |, OR       - Any must pass (lowest precedence)
        &, AND      - All must pass
        ~, NOT, !   - Invert result (highest precedence)

    Grouping:
        ( )         - Override precedence

    Rules:
        rule_name                   - Simple rule
        rule_name(arg)              - Rule with one argument
        rule_name(arg1, arg2)       - Rule with multiple arguments

    Multi-line expressions are supported (newlines are ignored).

    Examples:
        is_admin & is_active
        is_admin AND is_active
        is_admin | is_premium
        is_admin OR is_premium
        ~is_banned
        NOT is_banned
        is_admin & (is_active | is_premium)
        account_older_than(30) & credit_above(700)

        # Multi-line
        is_admin
        & ~is_banned
        & account_older_than(30)
    """

    # Token types
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    EOF = "EOF"

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: list[tuple[str, Any]] = []
        self.token_pos = 0
        self._tokenize()

    def _tokenize(self) -> None:
        """Convert text into tokens."""
        while self.pos < len(self.text):
            ch = self.text[self.pos]

            # Skip whitespace and newlines
            if ch in " \t\n\r":
                self.pos += 1
                continue

            # Skip comments
            if ch == "#":
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
                continue

            # Operators
            if ch == "&":
                self.tokens.append((self.AND, "&"))
                self.pos += 1
            elif ch == "|":
                self.tokens.append((self.OR, "|"))
                self.pos += 1
            elif ch in "~!":
                self.tokens.append((self.NOT, ch))
                self.pos += 1
            elif ch == "(":
                self.tokens.append((self.LPAREN, "("))
                self.pos += 1
            elif ch == ")":
                self.tokens.append((self.RPAREN, ")"))
                self.pos += 1
            elif ch == ",":
                self.tokens.append((self.COMMA, ","))
                self.pos += 1

            # Strings
            elif ch in "\"'":
                self.tokens.append((self.STRING, self._read_string(ch)))

            # Numbers
            elif ch.isdigit() or (
                ch == "-"
                and self.pos + 1 < len(self.text)
                and self.text[self.pos + 1].isdigit()
            ):
                self.tokens.append((self.NUMBER, self._read_number()))

            # Identifiers and keywords
            elif ch.isalpha() or ch == "_":
                ident = self._read_ident()
                upper = ident.upper()
                if upper == "AND":
                    self.tokens.append((self.AND, ident))
                elif upper == "OR":
                    self.tokens.append((self.OR, ident))
                elif upper == "NOT":
                    self.tokens.append((self.NOT, ident))
                else:
                    self.tokens.append((self.IDENT, ident))

            else:
                raise ValueError(f"Unexpected character: {ch!r} at position {self.pos}")

        self.tokens.append((self.EOF, None))

    def _read_string(self, quote: str) -> str:
        """Read a quoted string."""
        self.pos += 1  # skip opening quote
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != quote:
            if self.text[self.pos] == "\\":
                self.pos += 2  # skip escape
            else:
                self.pos += 1
        result = self.text[start : self.pos]
        self.pos += 1  # skip closing quote
        return result

    def _read_number(self) -> int | float:
        """Read a number."""
        start = self.pos
        if self.text[self.pos] == "-":
            self.pos += 1
        while self.pos < len(self.text) and (
            self.text[self.pos].isdigit() or self.text[self.pos] == "."
        ):
            self.pos += 1
        text = self.text[start : self.pos]
        return float(text) if "." in text else int(text)

    def _read_ident(self) -> str:
        """Read an identifier."""
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self.pos += 1
        return self.text[start : self.pos]

    def _peek(self) -> tuple[str, Any]:
        """Look at current token without consuming."""
        return self.tokens[self.token_pos]

    def _consume(self) -> tuple[str, Any]:
        """Consume and return current token."""
        token = self.tokens[self.token_pos]
        self.token_pos += 1
        return token

    def _expect(self, token_type: str) -> tuple[str, Any]:
        """Consume token and verify its type."""
        token = self._consume()
        if token[0] != token_type:
            raise ValueError(f"Expected {token_type}, got {token[0]}")
        return token

    def parse(self) -> dict | str:
        """
        Parse expression and return config dict.

        Grammar:
            expr     = or_expr
            or_expr  = and_expr (('|' | 'OR') and_expr)*
            and_expr = not_expr (('&' | 'AND') not_expr)*
            not_expr = ('~' | 'NOT' | '!')? primary
            primary  = IDENT args? | '(' expr ')'
            args     = '(' arg_list? ')'
            arg_list = arg (',' arg)*
            arg      = NUMBER | STRING | IDENT
        """
        result = self._parse_or()
        if self._peek()[0] != self.EOF:
            raise ValueError(f"Unexpected token: {self._peek()}")
        return result

    def _parse_or(self) -> dict | str:
        """Parse OR expression (lowest precedence)."""
        left = self._parse_and()

        items = [left]
        while self._peek()[0] == self.OR:
            self._consume()
            items.append(self._parse_and())

        if len(items) == 1:
            return items[0]
        return {"or": items}

    def _parse_and(self) -> dict | str:
        """Parse AND expression."""
        left = self._parse_not()

        items = [left]
        while self._peek()[0] == self.AND:
            self._consume()
            items.append(self._parse_not())

        if len(items) == 1:
            return items[0]
        return {"and": items}

    def _parse_not(self) -> dict | str:
        """Parse NOT expression (highest precedence)."""
        if self._peek()[0] == self.NOT:
            self._consume()
            inner = self._parse_not()  # Allow chained NOT
            return {"not": inner}
        return self._parse_primary()

    def _parse_primary(self) -> dict | str:
        """Parse primary expression (identifier or grouped expr)."""
        token = self._peek()

        if token[0] == self.LPAREN:
            self._consume()
            expr = self._parse_or()
            self._expect(self.RPAREN)
            return expr

        if token[0] == self.IDENT:
            name = self._consume()[1]

            # Check for arguments
            if self._peek()[0] == self.LPAREN:
                self._consume()  # (
                args = self._parse_args()
                self._expect(self.RPAREN)
                return {name: args}

            return name

        raise ValueError(f"Unexpected token: {token}")

    def _parse_args(self) -> list:
        """Parse argument list."""
        args = []

        if self._peek()[0] == self.RPAREN:
            return args

        args.append(self._parse_arg())

        while self._peek()[0] == self.COMMA:
            self._consume()
            args.append(self._parse_arg())

        return args

    def _parse_arg(self) -> Any:
        """Parse single argument."""
        token = self._peek()

        if token[0] == self.NUMBER:
            return self._consume()[1]
        if token[0] == self.STRING:
            return self._consume()[1]
        if token[0] == self.IDENT:
            # Treat bare identifiers as strings
            return self._consume()[1]

        raise ValueError(f"Invalid argument: {token}")


def parse_expression(text: str) -> dict | str:
    """
    Parse a rule expression into a config structure.

    Args:
        text: Expression string like "is_admin & ~is_banned"

    Returns:
        Config dict compatible with Registry.load()

    Example:
        >>> parse_expression("is_admin & ~is_banned")
        {'and': ['is_admin', {'not': 'is_banned'}]}
    """
    return ExpressionParser(text).parse()


class Registry(Generic[T]):
    """
    Registry for named predicates and transforms.

    Allows loading combinator chains from human-readable expressions.

    Example:
        reg: Registry[User] = Registry()

        @reg.predicate
        def is_admin(u: User) -> bool:
            return u.is_admin

        @reg.predicate
        def older_than(u: User, days: int) -> bool:
            return u.age > days

        # Load from expression
        loaded = reg.load("is_admin & older_than(30)")

        # Also supports word syntax
        loaded = reg.load("is_admin AND older_than(30)")
    """

    def __init__(self) -> None:
        self._predicates: dict[str, Predicate[T] | PredicateFactory[T]] = {}
        self._transforms: dict[str, Transform[T] | TransformFactory[T]] = {}

    def predicate(self, fn: Callable[..., bool]) -> Predicate[T] | PredicateFactory[T]:
        """
        Decorator to register a predicate.

        For simple predicates (single arg), returns Predicate[T].
        For parameterized predicates (multiple args), returns PredicateFactory[T].
        """
        params = list(inspect.signature(fn).parameters)

        if len(params) == 1:
            p: Predicate[T] = Predicate(fn, fn.__name__)
            self._predicates[fn.__name__] = p
            return p
        else:
            factory: PredicateFactory[T] = PredicateFactory(fn, fn.__name__)
            self._predicates[fn.__name__] = factory
            return factory

    def transform(self, fn: Callable[..., T]) -> Transform[T] | TransformFactory[T]:
        """
        Decorator to register a transform.

        For simple transforms (single arg), returns Transform[T].
        For parameterized transforms (multiple args), returns TransformFactory[T].
        """
        params = list(inspect.signature(fn).parameters)

        if len(params) == 1:
            t: Transform[T] = Transform(fn, fn.__name__)
            self._transforms[fn.__name__] = t
            return t
        else:
            factory: TransformFactory[T] = TransformFactory(fn, fn.__name__)
            self._transforms[fn.__name__] = factory
            return factory

    def load(self, expr: str) -> Combinator[T]:
        """
        Load a combinator chain from a human-readable expression.

        Expression format:
            # Simple rules
            is_admin
            is_active

            # Operators (symbols or words)
            is_admin & is_active          # AND
            is_admin AND is_active        # AND (same as &)
            is_admin | is_premium         # OR
            is_admin OR is_premium        # OR (same as |)
            ~is_banned                    # NOT
            NOT is_banned                 # NOT (same as ~)
            !is_banned                    # NOT (same as ~)

            # Grouping with parentheses
            is_admin | (is_active & ~is_banned)

            # Parameterized rules
            account_older_than(30)
            credit_above(700)
            in_role("admin", "moderator")

            # Multi-line (newlines are ignored)
            is_admin
            & is_active
            & ~is_banned
            & account_older_than(30)

            # Comments
            is_admin  # must be admin
            & ~is_banned  # and not banned

        Operator precedence (lowest to highest):
            OR  (|)  - evaluated last
            AND (&)  - evaluated second
            NOT (~)  - evaluated first
        """
        config = parse_expression(expr)
        return self._build(config)

    def load_file(self, path: str) -> Combinator[T]:
        """Load combinator chain from an expression file."""
        from pathlib import Path

        content = Path(path).read_text()
        return self.load(content)

    def _build(self, node: dict | str) -> Combinator[T]:
        """Build a combinator from a parsed expression config."""
        # String: simple predicate or transform reference
        if isinstance(node, str):
            return self._resolve(node)

        # Dict: operator or parameterized call
        if isinstance(node, dict):
            if len(node) != 1:
                raise ValueError(f"Config node must have exactly one key: {node}")

            key, value = next(iter(node.items()))

            # Operators
            if key == "and":
                items = [self._build(item) for item in value]
                return self._combine_and(items)

            elif key == "or":
                items = [self._build(item) for item in value]
                return self._combine_or(items)

            elif key == "not":
                return ~self._build(value)

            # Parameterized predicate/transform
            else:
                return self._resolve(key, value)

        raise ValueError(f"Invalid config node: {node}")

    def _resolve(self, name: str, args: Any = None) -> Combinator[T]:
        """Resolve a name to a predicate or transform, optionally with args."""
        # Check predicates first, then transforms
        if name in self._predicates:
            pred_or_factory = self._predicates[name]
            if args is None:
                if isinstance(pred_or_factory, PredicateFactory):
                    raise ValueError(f"Predicate '{name}' requires arguments")
                return pred_or_factory
            else:
                if isinstance(pred_or_factory, Predicate):
                    raise ValueError(f"Predicate '{name}' does not take arguments")
                if isinstance(args, list):
                    return pred_or_factory(*args)
                elif isinstance(args, dict):
                    return pred_or_factory(**args)
                else:
                    return pred_or_factory(args)

        elif name in self._transforms:
            trans_or_factory = self._transforms[name]
            if args is None:
                if isinstance(trans_or_factory, TransformFactory):
                    raise ValueError(f"Transform '{name}' requires arguments")
                return trans_or_factory
            else:
                if isinstance(trans_or_factory, Transform):
                    raise ValueError(f"Transform '{name}' does not take arguments")
                if isinstance(args, list):
                    return trans_or_factory(*args)
                elif isinstance(args, dict):
                    return trans_or_factory(**args)
                else:
                    return trans_or_factory(args)

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
