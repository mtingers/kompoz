"""
Kompoz - Composable Predicate & Transform Combinators

A Python library for building composable, declarative rule chains using
operator overloading. Supports boolean logic (AND, OR, NOT), data pipelines,
and config-driven rules via a human-readable expression DSL.

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
    "if_then_else",
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
    "run_async_traced",
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
    "AsyncPredicateFactory",
    "AsyncTransform",
    "AsyncTransformFactory",
    "async_rule",
    "async_rule_args",
    "async_pipe",
    "async_pipe_args",
    "async_if_then_else",
    # Async Validation
    "AsyncValidatingCombinator",
    "AsyncValidatingPredicate",
    "async_vrule",
    "async_vrule_args",
    # Caching
    "CachedPredicate",
    "CachedPredicateFactory",
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

# Optional OpenTelemetry imports - only needed if using OpenTelemetryHook
try:
    from opentelemetry.trace import (
        Link as _Link,
    )
    from opentelemetry.trace import (
        Status as _Status,
    )
    from opentelemetry.trace import (
        StatusCode as _StatusCode,
    )
    from opentelemetry.trace import (
        set_span_in_context as _set_span_in_context,
    )

    _HAS_OPENTELEMETRY = True
except ImportError:
    _HAS_OPENTELEMETRY = False
    _Link = None
    _Status = None
    _StatusCode = None
    _set_span_in_context = None

T = TypeVar("T")

# Type aliases for callbacks
RetryCallback = Callable[[int, Exception | None, float], None]
"""Callback signature for retry hooks: (attempt, error, delay) -> None"""

AsyncRetryCallback = Callable[[int, Exception | None, float], Any]
"""Callback signature for async retry hooks: (attempt, error, delay) -> None or Coroutine"""


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

    Conditional branching:
        condition.if_else(then_branch, else_branch)
        if_then_else(condition, then_branch, else_branch)

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

    def if_else(
        self, then_branch: Combinator[T], else_branch: Combinator[T]
    ) -> Combinator[T]:
        """
        Create a conditional: if self succeeds, run then_branch; else run else_branch.

        Example:
            is_premium.if_else(apply_discount, charge_full_price)

        This is equivalent to:
            IF is_premium THEN apply_discount ELSE charge_full_price

        Unlike OR (|), this always executes exactly one branch based on the condition.
        """
        return _IfThenElse(self, then_branch, else_branch)

    def __call__(self, ctx: T) -> tuple[bool, T]:
        """Shorthand for run()."""
        return self.run(ctx)


def _is_composite(combinator: Combinator) -> bool:
    """Check if combinator is a composite type that needs stack-based execution."""
    return isinstance(combinator, (_And, _Or, _Not, _Then, _IfThenElse))


def _execute_iterative(root: Combinator[T], ctx: T) -> tuple[bool, T]:
    """
    Execute a combinator tree iteratively using an explicit stack.

    This avoids deep recursion for complex nested combinators.
    Uses continuation-passing style with a work stack.
    """
    # Stack items: (combinator, context, continuation_type, continuation_data)
    # continuation_type: None (fresh), 'and_left', 'and_right', 'or_left', 'or_right',
    #                    'not', 'then_left', 'then_right'

    # Result stack for passing results back up
    result_stack: list[tuple[bool, T]] = []

    # Work stack: (combinator, ctx, phase)
    # phase: 0 = initial, 1 = after left, 2 = done
    work_stack: list[tuple[Combinator[T], T, int, Any]] = [(root, ctx, 0, None)]

    while work_stack:
        combinator, current_ctx, phase, saved_data = work_stack.pop()

        # Handle _And
        if isinstance(combinator, _And):
            if phase == 0:
                # First, execute left child
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.left):
                    work_stack.append((combinator.left, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.left._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                # Left done, check result
                ok, new_ctx = result_stack.pop()
                if not ok:
                    result_stack.append((False, new_ctx))
                else:
                    # Execute right child
                    work_stack.append((combinator, new_ctx, 2, None))
                    if _is_composite(combinator.right):
                        work_stack.append((combinator.right, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.right._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:  # phase == 2
                # Right done, result is already on stack
                pass

        # Handle _Or
        elif isinstance(combinator, _Or):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.left):
                    work_stack.append((combinator.left, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.left._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                ok, new_ctx = result_stack.pop()
                if ok:
                    result_stack.append((True, new_ctx))
                else:
                    work_stack.append((combinator, new_ctx, 2, None))
                    if _is_composite(combinator.right):
                        work_stack.append((combinator.right, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.right._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:
                pass

        # Handle _Not
        elif isinstance(combinator, _Not):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.inner):
                    work_stack.append((combinator.inner, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.inner._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            else:
                ok, new_ctx = result_stack.pop()
                result_stack.append((not ok, new_ctx))

        # Handle _Then
        elif isinstance(combinator, _Then):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.left):
                    work_stack.append((combinator.left, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.left._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                _, new_ctx = result_stack.pop()  # Ignore left result
                work_stack.append((combinator, new_ctx, 2, None))
                if _is_composite(combinator.right):
                    work_stack.append((combinator.right, new_ctx, 0, None))
                else:
                    ok2, new_ctx2 = combinator.right._execute(new_ctx)
                    result_stack.append((ok2, new_ctx2))
            else:
                pass

        # Handle _IfThenElse
        elif isinstance(combinator, _IfThenElse):
            if phase == 0:
                # First, evaluate the condition
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_composite(combinator.condition):
                    work_stack.append((combinator.condition, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.condition._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                # Condition evaluated, choose branch
                cond_ok, new_ctx = result_stack.pop()
                work_stack.append((combinator, new_ctx, 2, None))
                if cond_ok:
                    # Execute then_branch
                    if _is_composite(combinator.then_branch):
                        work_stack.append((combinator.then_branch, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.then_branch._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
                else:
                    # Execute else_branch
                    if _is_composite(combinator.else_branch):
                        work_stack.append((combinator.else_branch, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.else_branch._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:  # phase == 2
                # Branch result is on stack
                pass

        # Handle leaf combinators (shouldn't normally get here from root)
        else:
            ok, new_ctx = combinator._execute(current_ctx)
            result_stack.append((ok, new_ctx))

    return result_stack[-1] if result_stack else (False, ctx)


@dataclass
class _And(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


@dataclass
class _Or(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


@dataclass
class _Not(Combinator[T]):
    inner: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


@dataclass
class _Then(Combinator[T]):
    left: Combinator[T]
    right: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


@dataclass
class _IfThenElse(Combinator[T]):
    """
    Conditional combinator: if condition succeeds, run then_branch; else run else_branch.

    Unlike OR which short-circuits on success, this explicitly branches:
    - condition ? then_branch : else_branch
    - IF condition THEN then_branch ELSE else_branch
    """

    condition: Combinator[T]
    then_branch: Combinator[T]
    else_branch: Combinator[T]

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_iterative(self, ctx)


def if_then_else(
    condition: Combinator[T], then_branch: Combinator[T], else_branch: Combinator[T]
) -> Combinator[T]:
    """
    Create a conditional combinator: if condition succeeds, run then_branch; else run else_branch.

    Example:
        from kompoz import if_then_else, rule

        @rule
        def is_premium(user):
            return user.is_premium

        @rule
        def apply_discount(user):
            user.discount = 0.2
            return True

        @rule
        def charge_full_price(user):
            user.discount = 0
            return True

        pricing = if_then_else(is_premium, apply_discount, charge_full_price)
        ok, user = pricing.run(user)

    DSL equivalent:
        IF is_premium THEN apply_discount ELSE charge_full_price
        is_premium ? apply_discount : charge_full_price

    Unlike OR (|) which is a fallback (try a, if fail try b), if_then_else
    explicitly branches based on the condition result.
    """
    return _IfThenElse(condition, then_branch, else_branch)


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

    Attributes:
        last_error: The last exception that caused failure (if any)
    """

    def __init__(self, fn: Callable[[T], T], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "transform")
        self.last_error: Exception | None = None

    def _execute(self, ctx: T) -> tuple[bool, T]:
        try:
            result = self.fn(ctx)
            self.last_error = None
            return True, result
        except Exception as e:
            self.last_error = e
            return False, ctx

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
                status = "✔" if ok else "✗"
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
    # Check specific types first (before base classes)
    # Validating combinators (check before Predicate since ValidatingPredicate inherits Combinator)
    if isinstance(combinator, ValidatingPredicate):
        return f"ValidatingPredicate({combinator.name})"
    if isinstance(combinator, _ValidatingAnd):
        return "ValidatingAND"
    if isinstance(combinator, _ValidatingOr):
        return "ValidatingOR"
    if isinstance(combinator, _ValidatingNot):
        return "ValidatingNOT"
    # Cached predicate (check before Predicate)
    if isinstance(combinator, CachedPredicate):
        return f"CachedPredicate({combinator.name})"
    # Base predicates and transforms
    if isinstance(combinator, Predicate):
        return f"Predicate({combinator.name})"
    if isinstance(combinator, Transform):
        return f"Transform({combinator.name})"
    # Composite combinators
    if isinstance(combinator, _And):
        return "AND"
    if isinstance(combinator, _Or):
        return "OR"
    if isinstance(combinator, _Not):
        return "NOT"
    if isinstance(combinator, _Then):
        return "THEN"
    if isinstance(combinator, _IfThenElse):
        return "IF_THEN_ELSE"
    # Utility combinators
    if isinstance(combinator, Always):
        return "Always"
    if isinstance(combinator, Never):
        return "Never"
    if isinstance(combinator, Debug):
        return f"Debug({combinator.label})"
    if isinstance(combinator, Try):
        return f"Try({combinator.name})"
    # Retry
    if isinstance(combinator, Retry):
        return f"Retry({combinator.name})"
    # Temporal combinators
    if isinstance(combinator, during_hours):
        return repr(combinator)
    if isinstance(combinator, on_weekdays):
        return "on_weekdays()"
    if isinstance(combinator, on_days):
        return repr(combinator)
    if isinstance(combinator, after_date):
        return repr(combinator)
    if isinstance(combinator, before_date):
        return repr(combinator)
    if isinstance(combinator, between_dates):
        return repr(combinator)
    return repr(combinator)


def _traced_run(
    combinator: Combinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig,
    depth: int = 0,
) -> tuple[bool, T]:
    """Execute a combinator with tracing (using flattening to avoid deep recursion)."""
    return _traced_run_impl(combinator, ctx, hook, config, depth)


def _is_traced_composite(combinator: Combinator) -> bool:
    """Check if combinator is a composite type for tracing."""
    return isinstance(
        combinator,
        (_And, _Or, _Not, _Then, _ValidatingAnd, _ValidatingOr, _ValidatingNot),
    )


def _flatten_and_chain(combinator: Combinator[T]) -> list[Combinator[T]]:
    """Flatten nested AND combinators into a list (iteratively)."""
    result: list[Combinator[T]] = []
    stack: list[Combinator[T]] = [combinator]
    while stack:
        current = stack.pop()
        if isinstance(current, (_And, _ValidatingAnd)):
            stack.append(current.right)
            stack.append(current.left)
        else:
            result.append(current)
    return result


def _flatten_or_chain(combinator: Combinator[T]) -> list[Combinator[T]]:
    """Flatten nested OR combinators into a list (iteratively)."""
    result: list[Combinator[T]] = []
    stack: list[Combinator[T]] = [combinator]
    while stack:
        current = stack.pop()
        if isinstance(current, (_Or, _ValidatingOr)):
            stack.append(current.right)
            stack.append(current.left)
        else:
            result.append(current)
    return result


def _flatten_then_chain(combinator: Combinator[T]) -> list[Combinator[T]]:
    """Flatten nested THEN combinators into a list (iteratively)."""
    result: list[Combinator[T]] = []
    stack: list[Combinator[T]] = [combinator]
    while stack:
        current = stack.pop()
        if isinstance(current, _Then):
            stack.append(current.right)
            stack.append(current.left)
        else:
            result.append(current)
    return result


def _unwrap_not(combinator: Combinator[T]) -> tuple[Combinator[T], int]:
    """Unwrap chained NOTs and return (inner, count)."""
    count = 0
    current = combinator
    while isinstance(current, (_Not, _ValidatingNot)):
        count += 1
        current = current.inner
    return current, count


def _traced_run_impl(
    combinator: Combinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig,
    depth: int,
) -> tuple[bool, T]:
    """
    Execute a combinator with tracing.

    Uses flattening for chains to avoid deep recursion, while still
    providing proper trace events for each node.
    """
    # Check depth limit
    if config.max_depth is not None and depth > config.max_depth:
        return combinator._execute(ctx)

    name = _get_combinator_name(combinator)
    is_composite = _is_traced_composite(combinator)

    # Skip composite combinators if leaf_only mode
    if config.include_leaf_only and is_composite:
        if config.nested:
            return _traced_composite_no_span(combinator, ctx, hook, config, depth)
        return combinator._execute(ctx)

    # Call on_enter
    span = hook.on_enter(name, ctx, depth)
    start = time.perf_counter()

    try:
        if config.nested and is_composite:
            ok, result = _traced_composite(
                combinator, ctx, hook, config, depth, span, name, start
            )
        else:
            ok, result = combinator._execute(ctx)
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, ok, duration_ms, depth)

        return ok, result

    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        hook.on_error(span, name, e, duration_ms, depth)
        raise


def _traced_composite(
    combinator: Combinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig,
    depth: int,
    span: Any,
    name: str,
    start: float,
) -> tuple[bool, T]:
    """Handle tracing for composite combinators with proper span management."""

    # Handle AND chains
    if isinstance(combinator, (_And, _ValidatingAnd)):
        children = _flatten_and_chain(combinator)
        current_ctx = ctx
        for child in children:
            ok, current_ctx = _traced_run_impl(
                child, current_ctx, hook, config, depth + 1
            )
            if not ok:
                duration_ms = (time.perf_counter() - start) * 1000
                hook.on_exit(span, name, False, duration_ms, depth)
                return False, current_ctx
        duration_ms = (time.perf_counter() - start) * 1000
        hook.on_exit(span, name, True, duration_ms, depth)
        return True, current_ctx

    # Handle OR chains
    if isinstance(combinator, (_Or, _ValidatingOr)):
        children = _flatten_or_chain(combinator)
        current_ctx = ctx
        for child in children:
            ok, current_ctx = _traced_run_impl(
                child, current_ctx, hook, config, depth + 1
            )
            if ok:
                duration_ms = (time.perf_counter() - start) * 1000
                hook.on_exit(span, name, True, duration_ms, depth)
                return True, current_ctx
        duration_ms = (time.perf_counter() - start) * 1000
        hook.on_exit(span, name, False, duration_ms, depth)
        return False, current_ctx

    # Handle NOT (with chained NOT unwrapping)
    if isinstance(combinator, (_Not, _ValidatingNot)):
        inner, invert_count = _unwrap_not(combinator)
        ok, result = _traced_run_impl(inner, ctx, hook, config, depth + 1)
        if invert_count % 2 == 1:
            ok = not ok
        duration_ms = (time.perf_counter() - start) * 1000
        hook.on_exit(span, name, ok, duration_ms, depth)
        return ok, result

    # Handle THEN chains
    if isinstance(combinator, _Then):
        children = _flatten_then_chain(combinator)
        current_ctx = ctx
        for child in children[:-1]:
            _, current_ctx = _traced_run_impl(
                child, current_ctx, hook, config, depth + 1
            )
        ok, current_ctx = _traced_run_impl(
            children[-1], current_ctx, hook, config, depth + 1
        )
        duration_ms = (time.perf_counter() - start) * 1000
        hook.on_exit(span, name, ok, duration_ms, depth)
        return ok, current_ctx

    # Fallback - shouldn't reach here for known composites
    ok, result = combinator._execute(ctx)
    duration_ms = (time.perf_counter() - start) * 1000
    hook.on_exit(span, name, ok, duration_ms, depth)
    return ok, result


def _traced_composite_no_span(
    combinator: Combinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig,
    depth: int,
) -> tuple[bool, T]:
    """Handle composite tracing in leaf_only mode (no span for this node)."""

    # Handle AND chains
    if isinstance(combinator, (_And, _ValidatingAnd)):
        children = _flatten_and_chain(combinator)
        current_ctx = ctx
        for child in children:
            ok, current_ctx = _traced_run_impl(
                child, current_ctx, hook, config, depth + 1
            )
            if not ok:
                return False, current_ctx
        return True, current_ctx

    # Handle OR chains
    if isinstance(combinator, (_Or, _ValidatingOr)):
        children = _flatten_or_chain(combinator)
        current_ctx = ctx
        for child in children:
            ok, current_ctx = _traced_run_impl(
                child, current_ctx, hook, config, depth + 1
            )
            if ok:
                return True, current_ctx
        return False, current_ctx

    # Handle NOT
    if isinstance(combinator, (_Not, _ValidatingNot)):
        inner, invert_count = _unwrap_not(combinator)
        ok, result = _traced_run_impl(inner, ctx, hook, config, depth + 1)
        if invert_count % 2 == 1:
            ok = not ok
        return ok, result

    # Handle THEN chains
    if isinstance(combinator, _Then):
        children = _flatten_then_chain(combinator)
        current_ctx = ctx
        for child in children[:-1]:
            _, current_ctx = _traced_run_impl(
                child, current_ctx, hook, config, depth + 1
            )
        return _traced_run_impl(children[-1], current_ctx, hook, config, depth + 1)

    # Handle IF/THEN/ELSE
    if isinstance(combinator, _IfThenElse):
        cond_ok, new_ctx = _traced_run_impl(
            combinator.condition, ctx, hook, config, depth + 1
        )
        if cond_ok:
            return _traced_run_impl(
                combinator.then_branch, new_ctx, hook, config, depth + 1
            )
        else:
            return _traced_run_impl(
                combinator.else_branch, new_ctx, hook, config, depth + 1
            )

    # Fallback
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


async def run_async_traced(
    combinator: AsyncCombinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig | None = None,
) -> tuple[bool, T]:
    """
    Run an async combinator with explicit tracing.

    Args:
        combinator: The async combinator to run
        ctx: Context to evaluate
        hook: TraceHook to receive events
        config: Optional TraceConfig

    Returns:
        Tuple of (success, result_context)

    Example:
        ok, result = await run_async_traced(async_rule, user, PrintHook())
    """
    return await _async_traced_run(combinator, ctx, hook, config or TraceConfig())


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
        # <- Predicate(is_active) ✔ (0.01ms)
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
        status = "✔" if ok else "✗"
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
    Full-featured OpenTelemetry trace hook for Kompoz with:

    - Correct parent/child span hierarchy
    - Explicit context management (async-safe)
    - Depth-based span suppression
    - Automatic span collapsing for single-child logical nodes
    - Predicate-as-event optimization
    - Logical operator semantics (AND / OR / NOT)
    - Short-circuit evaluation tagging
    - Optional sibling span linking
    - Trace-derived metric attributes

    Requires: pip install opentelemetry-api
    """

    def __init__(
        self,
        tracer,
        *,
        max_span_depth: int | None = None,
        link_sibling_spans: bool = True,
        collapse_single_child_operators: bool = True,
        predicates_as_events: bool = False,
    ):
        if not _HAS_OPENTELEMETRY:
            raise ImportError(
                "OpenTelemetry is not installed. "
                "Install it with: pip install opentelemetry-api"
            )
        self.tracer = tracer
        self.max_span_depth = max_span_depth
        self.link_sibling_spans = link_sibling_spans
        self.collapse_single_child_operators = collapse_single_child_operators
        self.predicates_as_events = predicates_as_events

        self._span_stack: list[Any] = []
        self._last_span_at_depth: dict[int, Any] = {}
        self._child_count: dict[Any, int] = {}

    # -------------------------------------------------
    # Span lifecycle
    # -------------------------------------------------

    def on_enter(self, name: str, ctx: Any, depth: int) -> Any:
        # These are guaranteed non-None because __init__ checks _HAS_OPENTELEMETRY
        assert _set_span_in_context is not None
        assert _Link is not None

        # Depth-based suppression
        if self.max_span_depth is not None and depth > self.max_span_depth:
            return None

        parent = self._span_stack[-1] if self._span_stack else None
        parent_ctx = _set_span_in_context(parent) if parent else None

        # Predicate-as-event optimization
        if self.predicates_as_events and parent and self._is_predicate(name):
            parent.add_event(
                "predicate.evaluate",
                {
                    "kompoz.predicate": name,
                    "kompoz.depth": depth,
                },
            )
            self._increment_child(parent)
            return None

        links = []
        if self.link_sibling_spans and depth in self._last_span_at_depth:
            links.append(_Link(self._last_span_at_depth[depth].get_span_context()))

        span = self.tracer.start_span(
            name,
            context=parent_ctx,
            links=links or None,
        )

        self._annotate_span(span, name, depth)

        if parent:
            self._increment_child(parent)

        self._span_stack.append(span)
        self._last_span_at_depth[depth] = span
        self._child_count[span] = 0
        return span

    def on_exit(
        self,
        span: Any,
        name: str,
        ok: bool,
        duration_ms: float,
        depth: int,
    ) -> None:
        if span is None:
            return

        # These are guaranteed non-None because __init__ checks _HAS_OPENTELEMETRY
        assert _Status is not None
        assert _StatusCode is not None

        span.set_attribute("kompoz.success", ok)
        span.set_attribute("kompoz.duration_ms", duration_ms)
        span.set_attribute("kompoz.depth", depth)

        if not ok:
            span.set_status(_Status(_StatusCode.ERROR))

        # Collapse single-child logical operators
        if (
            self.collapse_single_child_operators
            and span.attributes.get("kompoz.node_type") == "logical"
            and self._child_count.get(span, 0) == 1
        ):
            span.set_attribute("kompoz.collapsed", True)

        span.end()
        self._span_stack.pop()

    def on_error(
        self,
        span: Any,
        name: str,
        error: Exception,
        duration_ms: float,
        depth: int,
    ) -> None:
        if span is None:
            return

        # These are guaranteed non-None because __init__ checks _HAS_OPENTELEMETRY
        assert _Status is not None
        assert _StatusCode is not None

        span.set_attribute("kompoz.success", False)
        span.set_attribute("kompoz.duration_ms", duration_ms)
        span.set_attribute("kompoz.depth", depth)
        span.record_exception(error)
        span.set_status(_Status(_StatusCode.ERROR, str(error)))

        span.end()
        self._span_stack.pop()

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _increment_child(self, span: Any) -> None:
        self._child_count[span] = self._child_count.get(span, 0) + 1

    def _is_predicate(self, name: str) -> bool:
        return name.upper().startswith("PREDICATE")

    def _annotate_span(self, span: Any, name: str, depth: int) -> None:
        upper = name.upper()

        if upper in {"AND", "OR", "NOT"}:
            span.set_attribute("kompoz.operator", upper)
            span.set_attribute("kompoz.node_type", "logical")
            span.set_attribute("kompoz.short_circuit", False)
        else:
            span.set_attribute("kompoz.node_type", "execution")

        span.set_attribute("kompoz.name", name)
        span.set_attribute("kompoz.depth", depth)


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
    return _explain_iterative(combinator, verbose=verbose)


def _explain_iterative(combinator: Combinator, verbose: bool) -> str:
    """Iterative explain implementation using a stack."""
    # Stack items: (combinator, depth, output_index)
    # output_index is where to insert this node's output in the results list

    # We'll build a list of (depth, text) tuples, then join them
    output_lines: list[tuple[int, str]] = []

    # Stack: (combinator, depth)
    # We process in reverse order so output is in correct order
    stack: list[tuple[Combinator, int]] = [(combinator, 0)]

    while stack:
        comb, depth = stack.pop()
        indent = "  " * depth
        bullet = "• " if depth > 0 else ""

        # Check specific types first (before base classes)
        # Validating combinators
        if isinstance(comb, ValidatingPredicate):
            output_lines.append((depth, f"{indent}{bullet}Validate: {comb.name}"))

        elif isinstance(comb, _ValidatingAnd):
            children = _collect_chain(comb, _ValidatingAnd, "left", "right")
            if depth == 0:
                header = "Validate ALL of (collect errors):"
            else:
                header = f"{indent}{bullet}ALL of (collect errors):"
            output_lines.append((depth, header))
            # Add children in reverse order so they appear in correct order
            for child in reversed(children):
                stack.append((child, depth + 1))

        elif isinstance(comb, _ValidatingOr):
            children = _collect_chain(comb, _ValidatingOr, "left", "right")
            if depth == 0:
                header = "Validate ANY of:"
            else:
                header = f"{indent}{bullet}ANY of:"
            output_lines.append((depth, header))
            for child in reversed(children):
                stack.append((child, depth + 1))

        elif isinstance(comb, _ValidatingNot):
            inner = _explain_inline_iterative(comb.inner)
            output_lines.append((depth, f"{indent}{bullet}NOT (validating): {inner}"))

        # Cached predicate
        elif isinstance(comb, CachedPredicate):
            output_lines.append((depth, f"{indent}{bullet}Cached check: {comb.name}"))

        # Base predicate and transform
        elif isinstance(comb, Predicate):
            output_lines.append((depth, f"{indent}{bullet}Check: {comb.name}"))

        elif isinstance(comb, Transform):
            output_lines.append((depth, f"{indent}{bullet}Transform: {comb.name}"))

        # Standard composite combinators
        elif isinstance(comb, _And):
            children = _collect_chain(comb, _And, "left", "right")
            if depth == 0:
                header = "Check passes if ALL of:"
            else:
                header = f"{indent}{bullet}ALL of:"
            output_lines.append((depth, header))
            for child in reversed(children):
                stack.append((child, depth + 1))

        elif isinstance(comb, _Or):
            children = _collect_chain(comb, _Or, "left", "right")
            if depth == 0:
                header = "Check passes if ANY of:"
            else:
                header = f"{indent}{bullet}ANY of:"
            output_lines.append((depth, header))
            for child in reversed(children):
                stack.append((child, depth + 1))

        elif isinstance(comb, _Not):
            inner = _explain_inline_iterative(comb.inner)
            output_lines.append((depth, f"{indent}{bullet}NOT: {inner}"))

        elif isinstance(comb, _Then):
            children = _collect_chain(comb, _Then, "left", "right")
            if depth == 0:
                header = "Execute in sequence (always run all):"
            else:
                header = f"{indent}{bullet}Sequence:"
            output_lines.append((depth, header))
            for child in reversed(children):
                stack.append((child, depth + 1))

        elif isinstance(comb, _IfThenElse):
            cond_str = _explain_inline_iterative(comb.condition)
            then_str = _explain_inline_iterative(comb.then_branch)
            else_str = _explain_inline_iterative(comb.else_branch)
            if depth == 0:
                output_lines.append((depth, f"IF {cond_str}"))
                output_lines.append((depth, f"THEN {then_str}"))
                output_lines.append((depth, f"ELSE {else_str}"))
            else:
                output_lines.append(
                    (
                        depth,
                        f"{indent}{bullet}IF {cond_str} THEN {then_str} ELSE {else_str}",
                    )
                )

        # Utility combinators
        elif isinstance(comb, Always):
            output_lines.append((depth, f"{indent}{bullet}Always pass"))

        elif isinstance(comb, Never):
            output_lines.append((depth, f"{indent}{bullet}Always fail"))

        elif isinstance(comb, Debug):
            output_lines.append((depth, f"{indent}{bullet}Debug: {comb.label}"))

        elif isinstance(comb, Try):
            output_lines.append(
                (depth, f"{indent}{bullet}Try: {comb.name} (catch errors)")
            )

        # Retry
        elif isinstance(comb, Retry):
            inner_explain = _explain_inline_iterative(comb.inner)
            output_lines.append(
                (
                    depth,
                    f"{indent}{bullet}Retry up to {comb.max_attempts}x: {inner_explain}",
                )
            )

        # Temporal combinators
        elif isinstance(comb, during_hours):
            end_type = (
                "inclusive" if getattr(comb, "inclusive_end", False) else "exclusive"
            )
            output_lines.append(
                (
                    depth,
                    f"{indent}{bullet}During hours {comb.start_hour}:00-{comb.end_hour}:00 ({end_type})",
                )
            )

        elif isinstance(comb, on_weekdays):
            output_lines.append((depth, f"{indent}{bullet}On weekdays (Mon-Fri)"))

        elif isinstance(comb, on_days):
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            days_str = ", ".join(day_names[d] for d in sorted(comb.days))
            output_lines.append((depth, f"{indent}{bullet}On days: {days_str}"))

        elif isinstance(comb, after_date):
            output_lines.append((depth, f"{indent}{bullet}After {comb.date}"))

        elif isinstance(comb, before_date):
            output_lines.append((depth, f"{indent}{bullet}Before {comb.date}"))

        elif isinstance(comb, between_dates):
            output_lines.append(
                (
                    depth,
                    f"{indent}{bullet}Between {comb.start_date} and {comb.end_date}",
                )
            )

        # Fallback
        else:
            output_lines.append((depth, f"{indent}{bullet}{repr(comb)}"))

    return "\n".join(line for _, line in output_lines)


def _explain_inline_iterative(combinator: Combinator) -> str:
    """Get a short inline explanation for NOT children (iteratively)."""
    # For inline, we build a string representation
    # Use a stack with instructions

    result_parts: list[str] = []
    # Stack: (combinator, instruction)
    # instruction: 'process' or 'join_and' or 'join_or' or 'join_then'
    stack: list[tuple[Any, str]] = [(combinator, "process")]

    while stack:
        item, instruction = stack.pop()

        if instruction == "process":
            comb = item

            # Validating combinators
            if isinstance(comb, ValidatingPredicate):
                result_parts.append(comb.name)
            elif isinstance(comb, _ValidatingAnd):
                children = _collect_chain(comb, _ValidatingAnd, "left", "right")
                stack.append((len(children), "join_and"))
                for child in reversed(children):
                    stack.append((child, "process"))
            elif isinstance(comb, _ValidatingOr):
                children = _collect_chain(comb, _ValidatingOr, "left", "right")
                stack.append((len(children), "join_or"))
                for child in reversed(children):
                    stack.append((child, "process"))
            elif isinstance(comb, _ValidatingNot):
                stack.append((None, "prefix_not"))
                stack.append((comb.inner, "process"))
            # Cached predicate
            elif isinstance(comb, CachedPredicate):
                result_parts.append(comb.name)
            # Base types
            elif isinstance(comb, Predicate):
                result_parts.append(comb.name)
            elif isinstance(comb, Transform):
                result_parts.append(comb.name)
            elif isinstance(comb, _And):
                children = _collect_chain(comb, _And, "left", "right")
                stack.append((len(children), "join_and"))
                for child in reversed(children):
                    stack.append((child, "process"))
            elif isinstance(comb, _Or):
                children = _collect_chain(comb, _Or, "left", "right")
                stack.append((len(children), "join_or"))
                for child in reversed(children):
                    stack.append((child, "process"))
            elif isinstance(comb, _Not):
                stack.append((None, "prefix_not"))
                stack.append((comb.inner, "process"))
            elif isinstance(comb, _Then):
                children = _collect_chain(comb, _Then, "left", "right")
                stack.append((len(children), "join_then"))
                for child in reversed(children):
                    stack.append((child, "process"))
            elif isinstance(comb, _IfThenElse):
                stack.append((None, "join_if"))
                stack.append((comb.else_branch, "process"))
                stack.append((comb.then_branch, "process"))
                stack.append((comb.condition, "process"))
            # Retry
            elif isinstance(comb, Retry):
                stack.append((None, "wrap_retry"))
                stack.append((comb.inner, "process"))
            # Temporal
            elif isinstance(comb, during_hours):
                result_parts.append(f"during_hours({comb.start_hour}, {comb.end_hour})")
            elif isinstance(comb, on_weekdays):
                result_parts.append("on_weekdays()")
            elif isinstance(comb, on_days):
                result_parts.append(repr(comb))
            elif isinstance(comb, after_date):
                result_parts.append(repr(comb))
            elif isinstance(comb, before_date):
                result_parts.append(repr(comb))
            elif isinstance(comb, between_dates):
                result_parts.append(repr(comb))
            else:
                result_parts.append(repr(comb))

        elif instruction == "join_and":
            count = item
            parts = [result_parts.pop() for _ in range(count)]
            parts.reverse()
            result_parts.append(f"({' & '.join(parts)})")

        elif instruction == "join_or":
            count = item
            parts = [result_parts.pop() for _ in range(count)]
            parts.reverse()
            result_parts.append(f"({' | '.join(parts)})")

        elif instruction == "join_then":
            count = item
            parts = [result_parts.pop() for _ in range(count)]
            parts.reverse()
            result_parts.append(f"({' >> '.join(parts)})")

        elif instruction == "prefix_not":
            inner = result_parts.pop()
            result_parts.append(f"~{inner}")

        elif instruction == "wrap_retry":
            inner = result_parts.pop()
            result_parts.append(f"Retry({inner})")

        elif instruction == "join_if":
            cond = result_parts.pop()
            then_part = result_parts.pop()
            else_part = result_parts.pop()
            result_parts.append(f"({cond} ? {then_part} : {else_part})")

    return result_parts[0] if result_parts else ""


def _collect_chain(
    combinator: Combinator, cls: type, left_attr: str, right_attr: str
) -> list:
    """Collect chained combinators of the same type (iteratively)."""
    result: list[Combinator] = []
    stack: list[Combinator] = [combinator]

    while stack:
        c = stack.pop()
        if isinstance(c, cls):
            # Push right first so left is processed first
            stack.append(getattr(c, right_attr))
            stack.append(getattr(c, left_attr))
        else:
            result.append(c)

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

    def __invert__(self) -> ValidatingCombinator[T]:
        """Override ~ to create validating NOT."""
        return _ValidatingNot(self)


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
        except (KeyError, AttributeError, IndexError):
            return self._error

    def validate(self, ctx: T) -> ValidationResult:
        """Run validation and return result with errors."""
        ok, result = self._execute(ctx)
        errors = [] if ok else [self.get_error(ctx)]
        return ValidationResult(ok=ok, errors=errors, ctx=result)

    def __repr__(self) -> str:
        return f"ValidatingPredicate({self.name})"


def _is_validating_composite(combinator: Combinator) -> bool:
    """Check if combinator is a validating composite type."""
    return isinstance(combinator, (_ValidatingAnd, _ValidatingOr, _ValidatingNot))


def _execute_validating_iterative(root: Combinator[T], ctx: T) -> tuple[bool, T]:
    """
    Execute a validating combinator tree iteratively using an explicit stack.
    """
    result_stack: list[tuple[bool, T]] = []
    work_stack: list[tuple[Combinator[T], T, int, Any]] = [(root, ctx, 0, None)]

    while work_stack:
        combinator, current_ctx, phase, saved_data = work_stack.pop()

        if isinstance(combinator, _ValidatingAnd):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_validating_composite(combinator.left) or _is_composite(
                    combinator.left
                ):
                    work_stack.append((combinator.left, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.left._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                ok, new_ctx = result_stack.pop()
                if not ok:
                    result_stack.append((False, new_ctx))
                else:
                    work_stack.append((combinator, new_ctx, 2, None))
                    if _is_validating_composite(combinator.right) or _is_composite(
                        combinator.right
                    ):
                        work_stack.append((combinator.right, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.right._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:
                pass

        elif isinstance(combinator, _ValidatingOr):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_validating_composite(combinator.left) or _is_composite(
                    combinator.left
                ):
                    work_stack.append((combinator.left, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.left._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            elif phase == 1:
                ok, new_ctx = result_stack.pop()
                if ok:
                    result_stack.append((True, new_ctx))
                else:
                    work_stack.append((combinator, new_ctx, 2, None))
                    if _is_validating_composite(combinator.right) or _is_composite(
                        combinator.right
                    ):
                        work_stack.append((combinator.right, new_ctx, 0, None))
                    else:
                        ok2, new_ctx2 = combinator.right._execute(new_ctx)
                        result_stack.append((ok2, new_ctx2))
            else:
                pass

        elif isinstance(combinator, _ValidatingNot):
            if phase == 0:
                work_stack.append((combinator, current_ctx, 1, None))
                if _is_validating_composite(combinator.inner) or _is_composite(
                    combinator.inner
                ):
                    work_stack.append((combinator.inner, current_ctx, 0, None))
                else:
                    ok, new_ctx = combinator.inner._execute(current_ctx)
                    result_stack.append((ok, new_ctx))
            else:
                ok, new_ctx = result_stack.pop()
                result_stack.append((not ok, new_ctx))

        # Handle standard composite combinators
        elif _is_composite(combinator):
            ok, new_ctx = _execute_iterative(combinator, current_ctx)
            result_stack.append((ok, new_ctx))

        else:
            ok, new_ctx = combinator._execute(current_ctx)
            result_stack.append((ok, new_ctx))

    return result_stack[-1] if result_stack else (False, ctx)


class _ValidatingAnd(ValidatingCombinator[T]):
    """AND combinator that collects all validation errors."""

    def __init__(self, left: Combinator[T], right: Combinator[T]):
        self.left = left
        self.right = right

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_validating_iterative(self, ctx)

    def validate(self, ctx: T) -> ValidationResult:
        """Validate both sides and collect all errors (iteratively)."""
        errors: list[str] = []

        # Flatten the AND chain iteratively
        to_validate: list[Combinator[T]] = []
        stack: list[Combinator[T]] = [self]
        while stack:
            current = stack.pop()
            if isinstance(current, _ValidatingAnd):
                stack.append(current.right)
                stack.append(current.left)
            else:
                to_validate.append(current)

        # Validate each item
        for item in to_validate:
            if isinstance(item, ValidatingCombinator):
                result = item.validate(ctx)
                errors.extend(result.errors)
                ctx = result.ctx
            else:
                ok, ctx = item._execute(ctx)
                if not ok:
                    errors.append(f"Check failed: {_get_combinator_name(item)}")

        return ValidationResult(ok=len(errors) == 0, errors=errors, ctx=ctx)


class _ValidatingOr(ValidatingCombinator[T]):
    """OR combinator for validation - passes if any succeeds."""

    def __init__(self, left: Combinator[T], right: Combinator[T]):
        self.left = left
        self.right = right

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_validating_iterative(self, ctx)

    def validate(self, ctx: T) -> ValidationResult:
        """Validate - passes if any in the chain passes (iteratively)."""
        # Flatten the OR chain iteratively
        to_validate: list[Combinator[T]] = []
        stack: list[Combinator[T]] = [self]
        while stack:
            current = stack.pop()
            if isinstance(current, _ValidatingOr):
                stack.append(current.right)
                stack.append(current.left)
            else:
                to_validate.append(current)

        # Try each item until one passes
        last_result: ValidationResult | None = None
        for item in to_validate:
            if isinstance(item, ValidatingCombinator):
                result = item.validate(ctx)
                if result.ok:
                    return result
                last_result = result
            else:
                ok, result_ctx = item._execute(ctx)
                if ok:
                    return ValidationResult(ok=True, errors=[], ctx=result_ctx)
                last_result = ValidationResult(
                    ok=False,
                    errors=[f"Check failed: {_get_combinator_name(item)}"],
                    ctx=result_ctx,
                )

        return last_result or ValidationResult(
            ok=False, errors=["No conditions to check"], ctx=ctx
        )


class _ValidatingNot(ValidatingCombinator[T]):
    """NOT combinator for validation - inverts the result."""

    def __init__(self, inner: Combinator[T], error: str | None = None):
        self.inner = inner
        self._error = error

    def _execute(self, ctx: T) -> tuple[bool, T]:
        return _execute_validating_iterative(self, ctx)

    def validate(self, ctx: T) -> ValidationResult:
        """Validate - inverts the inner result."""
        if isinstance(self.inner, ValidatingCombinator):
            inner_result = self.inner.validate(ctx)
            if inner_result.ok:
                error_msg = self._error or "NOT condition failed (inner passed)"
                return ValidationResult(
                    ok=False, errors=[error_msg], ctx=inner_result.ctx
                )
            else:
                return ValidationResult(ok=True, errors=[], ctx=inner_result.ctx)
        else:
            ok, result = self.inner._execute(ctx)
            if ok:
                error_msg = (
                    self._error or f"NOT {_get_combinator_name(self.inner)} failed"
                )
                return ValidationResult(ok=False, errors=[error_msg], ctx=result)
            else:
                return ValidationResult(ok=True, errors=[], ctx=result)


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
) -> Callable[..., ValidatingPredicate]: ...


@overload
def vrule_args(
    fn: None = None, *, error: str | Callable[..., str] | None = None
) -> Callable[[Callable[..., bool]], Callable[..., ValidatingPredicate]]: ...


def vrule_args(
    fn: Callable[..., bool] | None = None,
    *,
    error: str | Callable[..., str] | None = None,
) -> (
    Callable[..., ValidatingPredicate]
    | Callable[[Callable[..., bool]], Callable[..., ValidatingPredicate]]
):
    def decorator(f: Callable[..., bool]) -> Callable[..., ValidatingPredicate]:
        # 1. Inspect the function signature to enable param name
        sig = inspect.signature(f)

        def factory(*args: Any, **kwargs: Any) -> ValidatingPredicate:
            name = f"{f.__name__}({', '.join(map(repr, args))})"

            # 2. Helper to resolve parameter names from args
            def get_bound_params():
                # We assume the first argument of 'f' is 'ctx', which isn't in *args here.
                # We bind a dummy value for the first argument to align *args correctly.
                try:
                    bound = sig.bind_partial(None, *args, **kwargs)
                    bound.apply_defaults()
                    # Remove the first argument (the context placeholder)
                    params = dict(bound.arguments)
                    first_param_name = list(sig.parameters.keys())[0]
                    if first_param_name in params:
                        del params[first_param_name]
                    return params
                except TypeError:
                    # Fallback if binding fails
                    return kwargs

            err_msg: str | Callable[[Any], str] | None

            if error is None:
                err_msg = None

            # CASE A: Error is a callable (custom function)
            elif callable(error):
                error_fn: Callable[..., str] = error

                def make_error_fn(ctx: Any) -> str:
                    return error_fn(ctx, *args, **kwargs)

                err_msg = make_error_fn

            # CASE B: Error is a string (template)
            else:
                # This tells the type checker: "Inside this block, we are 100% sure this is a string."
                template_str: str = error

                def make_formatted_error(ctx: Any) -> str:
                    # 1. Standard {arg0}, {arg1} support
                    format_context = {f"arg{i}": v for i, v in enumerate(args)}

                    # 2. Parameter name support ({score})
                    format_context.update(get_bound_params())

                    # 3. Context support ({ctx})
                    format_context["ctx"] = ctx

                    try:
                        # Now we use 'template_str' instead of 'error'
                        return template_str.format(**format_context)
                    except (KeyError, IndexError, AttributeError):
                        return template_str

                err_msg = make_formatted_error

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
    Supports tracing via use_tracing() context manager.
    """

    @abstractmethod
    async def _execute(self, ctx: T) -> tuple[bool, T]:
        """Internal execution - subclasses implement this."""
        ...

    async def run(self, ctx: T) -> tuple[bool, T]:
        """
        Execute the combinator asynchronously.

        If tracing is enabled via use_tracing(), this will automatically
        trace the execution.
        """
        hook = _trace_hook.get()
        if hook is not None:
            config = _trace_config.get()
            return await _async_traced_run(self, ctx, hook, config, depth=0)

        return await self._execute(ctx)

    def __and__(self, other: AsyncCombinator[T]) -> AsyncCombinator[T]:
        return _AsyncAnd(self, other)

    def __or__(self, other: AsyncCombinator[T]) -> AsyncCombinator[T]:
        return _AsyncOr(self, other)

    def __invert__(self) -> AsyncCombinator[T]:
        return _AsyncNot(self)

    def __rshift__(self, other: AsyncCombinator[T]) -> AsyncCombinator[T]:
        return _AsyncThen(self, other)

    def if_else(
        self, then_branch: AsyncCombinator[T], else_branch: AsyncCombinator[T]
    ) -> AsyncCombinator[T]:
        """
        Create a conditional: if self succeeds, run then_branch; else run else_branch.

        Example:
            await is_premium.if_else(apply_discount, charge_full_price).run(user)

        Unlike OR (|), this always executes exactly one branch based on the condition.
        """
        return _AsyncIfThenElse(self, then_branch, else_branch)

    async def __call__(self, ctx: T) -> tuple[bool, T]:
        return await self.run(ctx)


def _get_async_combinator_name(combinator: AsyncCombinator) -> str:
    """Get a human-readable name for an async combinator."""
    # Check validating types before base types
    if isinstance(combinator, AsyncValidatingPredicate):
        return f"AsyncValidatingPredicate({combinator.name})"
    if isinstance(combinator, _AsyncValidatingAnd):
        return "AsyncValidatingAND"
    if isinstance(combinator, _AsyncValidatingOr):
        return "AsyncValidatingOR"
    if isinstance(combinator, _AsyncValidatingNot):
        return "AsyncValidatingNOT"
    if isinstance(combinator, AsyncPredicate):
        return f"AsyncPredicate({combinator.name})"
    if isinstance(combinator, AsyncTransform):
        return f"AsyncTransform({combinator.name})"
    if isinstance(combinator, _AsyncAnd):
        return "AsyncAND"
    if isinstance(combinator, _AsyncOr):
        return "AsyncOR"
    if isinstance(combinator, _AsyncNot):
        return "AsyncNOT"
    if isinstance(combinator, _AsyncThen):
        return "AsyncTHEN"
    if isinstance(combinator, _AsyncIfThenElse):
        return "AsyncIF_THEN_ELSE"
    if isinstance(combinator, AsyncRetry):
        return f"AsyncRetry({combinator.name})"
    return repr(combinator)


async def _async_traced_run(
    combinator: AsyncCombinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig,
    depth: int = 0,
) -> tuple[bool, T]:
    """Execute an async combinator with tracing (iteratively where possible)."""
    return await _async_traced_run_iterative(combinator, ctx, hook, config, depth)


def _is_async_composite(combinator: AsyncCombinator) -> bool:
    """Check if combinator is a composite type for async tracing."""
    return isinstance(
        combinator, (_AsyncAnd, _AsyncOr, _AsyncNot, _AsyncThen, _AsyncIfThenElse)
    )


async def _async_traced_run_iterative(
    root: AsyncCombinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig,
    initial_depth: int = 0,
) -> tuple[bool, T]:
    """Execute async combinator tree with tracing using explicit work list."""

    # For async, we use a work list approach but still need to await
    # We process the tree by flattening chains where possible

    async def process_node(
        combinator: AsyncCombinator[T],
        current_ctx: T,
        depth: int,
    ) -> tuple[bool, T]:
        """Process a single node with tracing."""

        # Check depth limit
        if config.max_depth is not None and depth > config.max_depth:
            return await combinator._execute(current_ctx)

        name = _get_async_combinator_name(combinator)
        is_composite = _is_async_composite(combinator)

        # Skip composite combinators if leaf_only mode
        if config.include_leaf_only and is_composite:
            if config.nested:
                return await process_composite_no_span(combinator, current_ctx, depth)
            return await combinator._execute(current_ctx)

        # Call on_enter
        span = hook.on_enter(name, current_ctx, depth)
        start = time.perf_counter()

        try:
            if config.nested and is_composite:
                ok, result = await process_composite(
                    combinator, current_ctx, depth, span, name, start
                )
            else:
                ok, result = await combinator._execute(current_ctx)
                duration_ms = (time.perf_counter() - start) * 1000
                hook.on_exit(span, name, ok, duration_ms, depth)

            return ok, result

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_error(span, name, e, duration_ms, depth)
            raise

    async def process_composite(
        combinator: AsyncCombinator[T],
        current_ctx: T,
        depth: int,
        span: Any,
        name: str,
        start: float,
    ) -> tuple[bool, T]:
        """Process composite combinator with proper span handling."""

        if isinstance(combinator, _AsyncAnd):
            # Flatten AND chain
            children = _flatten_async_and(combinator)
            ctx = current_ctx
            for child in children:
                ok, ctx = await process_node(child, ctx, depth + 1)
                if not ok:
                    duration_ms = (time.perf_counter() - start) * 1000
                    hook.on_exit(span, name, False, duration_ms, depth)
                    return False, ctx
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, True, duration_ms, depth)
            return True, ctx

        elif isinstance(combinator, _AsyncOr):
            # Flatten OR chain
            children = _flatten_async_or(combinator)
            ctx = current_ctx
            for child in children:
                ok, ctx = await process_node(child, ctx, depth + 1)
                if ok:
                    duration_ms = (time.perf_counter() - start) * 1000
                    hook.on_exit(span, name, True, duration_ms, depth)
                    return True, ctx
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, False, duration_ms, depth)
            return False, ctx

        elif isinstance(combinator, _AsyncNot):
            # Handle chained NOTs iteratively
            current: AsyncCombinator[T] = combinator
            invert_count = 0
            while isinstance(current, _AsyncNot):
                invert_count += 1
                current = current.inner

            ok, result = await process_node(current, current_ctx, depth + 1)
            if invert_count % 2 == 1:
                ok = not ok

            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, ok, duration_ms, depth)
            return ok, result

        elif isinstance(combinator, _AsyncThen):
            # Flatten THEN chain
            children = _flatten_async_then(combinator)
            ctx = current_ctx
            for child in children[:-1]:
                _, ctx = await process_node(child, ctx, depth + 1)
            ok, ctx = await process_node(children[-1], ctx, depth + 1)
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, ok, duration_ms, depth)
            return ok, ctx

        elif isinstance(combinator, _AsyncIfThenElse):
            cond_ok, new_ctx = await process_node(
                combinator.condition, current_ctx, depth + 1
            )
            if cond_ok:
                ok, result = await process_node(
                    combinator.then_branch, new_ctx, depth + 1
                )
            else:
                ok, result = await process_node(
                    combinator.else_branch, new_ctx, depth + 1
                )
            duration_ms = (time.perf_counter() - start) * 1000
            hook.on_exit(span, name, ok, duration_ms, depth)
            return ok, result

        # Fallback
        ok, result = await combinator._execute(current_ctx)
        duration_ms = (time.perf_counter() - start) * 1000
        hook.on_exit(span, name, ok, duration_ms, depth)
        return ok, result

    async def process_composite_no_span(
        combinator: AsyncCombinator[T],
        current_ctx: T,
        depth: int,
    ) -> tuple[bool, T]:
        """Process composite without creating span (leaf_only mode)."""

        if isinstance(combinator, _AsyncAnd):
            children = _flatten_async_and(combinator)
            ctx = current_ctx
            for child in children:
                ok, ctx = await process_node(child, ctx, depth + 1)
                if not ok:
                    return False, ctx
            return True, ctx

        elif isinstance(combinator, _AsyncOr):
            children = _flatten_async_or(combinator)
            ctx = current_ctx
            for child in children:
                ok, ctx = await process_node(child, ctx, depth + 1)
                if ok:
                    return True, ctx
            return False, ctx

        elif isinstance(combinator, _AsyncNot):
            current: AsyncCombinator[T] = combinator
            invert_count = 0
            while isinstance(current, _AsyncNot):
                invert_count += 1
                current = current.inner

            ok, result = await process_node(current, current_ctx, depth + 1)
            if invert_count % 2 == 1:
                ok = not ok
            return ok, result

        elif isinstance(combinator, _AsyncThen):
            children = _flatten_async_then(combinator)
            ctx = current_ctx
            for child in children[:-1]:
                _, ctx = await process_node(child, ctx, depth + 1)
            return await process_node(children[-1], ctx, depth + 1)

        elif isinstance(combinator, _AsyncIfThenElse):
            cond_ok, new_ctx = await process_node(
                combinator.condition, current_ctx, depth + 1
            )
            if cond_ok:
                return await process_node(combinator.then_branch, new_ctx, depth + 1)
            else:
                return await process_node(combinator.else_branch, new_ctx, depth + 1)

        return await combinator._execute(current_ctx)

    return await process_node(root, ctx, initial_depth)


def _flatten_async_and(combinator: AsyncCombinator[T]) -> list[AsyncCombinator[T]]:
    """Flatten nested _AsyncAnd into a list (iterative to avoid recursion)."""
    result: list[AsyncCombinator[T]] = []
    stack: list[AsyncCombinator[T]] = [combinator]
    while stack:
        current = stack.pop()
        if isinstance(current, _AsyncAnd):
            stack.append(current.right)
            stack.append(current.left)
        else:
            result.append(current)
    return result


def _flatten_async_or(combinator: AsyncCombinator[T]) -> list[AsyncCombinator[T]]:
    """Flatten nested _AsyncOr into a list (iterative to avoid recursion)."""
    result: list[AsyncCombinator[T]] = []
    stack: list[AsyncCombinator[T]] = [combinator]
    while stack:
        current = stack.pop()
        if isinstance(current, _AsyncOr):
            stack.append(current.right)
            stack.append(current.left)
        else:
            result.append(current)
    return result


def _flatten_async_then(combinator: AsyncCombinator[T]) -> list[AsyncCombinator[T]]:
    """Flatten nested _AsyncThen into a list (iterative to avoid recursion)."""
    result: list[AsyncCombinator[T]] = []
    stack: list[AsyncCombinator[T]] = [combinator]
    while stack:
        current = stack.pop()
        if isinstance(current, _AsyncThen):
            stack.append(current.right)
            stack.append(current.left)
        else:
            result.append(current)
    return result


@dataclass
class _AsyncAnd(AsyncCombinator[T]):
    left: AsyncCombinator[T]
    right: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        # Flatten chain and iterate to avoid deep recursion
        for combinator in _flatten_async_and(self):
            ok, ctx = await combinator._execute(ctx)
            if not ok:
                return False, ctx
        return True, ctx


@dataclass
class _AsyncOr(AsyncCombinator[T]):
    left: AsyncCombinator[T]
    right: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        # Flatten chain and iterate to avoid deep recursion
        for combinator in _flatten_async_or(self):
            ok, ctx = await combinator._execute(ctx)
            if ok:
                return True, ctx
        return False, ctx


@dataclass
class _AsyncNot(AsyncCombinator[T]):
    inner: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        # Handle chained NOT (e.g., ~~~a) iteratively
        current: AsyncCombinator[T] = self
        invert_count = 0
        while isinstance(current, _AsyncNot):
            invert_count += 1
            current = current.inner
        ok, ctx = await current._execute(ctx)
        # Odd number of inversions flips the result
        if invert_count % 2 == 1:
            ok = not ok
        return ok, ctx


@dataclass
class _AsyncThen(AsyncCombinator[T]):
    left: AsyncCombinator[T]
    right: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        # Flatten chain and iterate to avoid deep recursion
        combinators = _flatten_async_then(self)
        for combinator in combinators[:-1]:
            _, ctx = await combinator._execute(ctx)
        # Return the result of the last combinator
        return await combinators[-1]._execute(ctx)


@dataclass
class _AsyncIfThenElse(AsyncCombinator[T]):
    """
    Async conditional combinator: if condition succeeds, run then_branch; else run else_branch.

    Unlike OR which short-circuits on success, this explicitly branches:
    - condition ? then_branch : else_branch
    - IF condition THEN then_branch ELSE else_branch
    """

    condition: AsyncCombinator[T]
    then_branch: AsyncCombinator[T]
    else_branch: AsyncCombinator[T]

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        cond_ok, new_ctx = await self.condition._execute(ctx)
        if cond_ok:
            return await self.then_branch._execute(new_ctx)
        else:
            return await self.else_branch._execute(new_ctx)


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

    async def _execute(self, ctx: T) -> tuple[bool, T]:
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

    Attributes:
        last_error: The last exception that caused failure (if any)
    """

    def __init__(self, fn: Callable[[T], Any], name: str | None = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "async_transform")
        self.last_error: Exception | None = None

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        try:
            result = await self.fn(ctx)
            self.last_error = None
            return True, result
        except Exception as e:
            self.last_error = e
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


def async_if_then_else(
    condition: AsyncCombinator[T],
    then_branch: AsyncCombinator[T],
    else_branch: AsyncCombinator[T],
) -> AsyncCombinator[T]:
    """
    Create an async conditional combinator: if condition succeeds, run then_branch; else run else_branch.

    Example:
        from kompoz import async_if_then_else, async_rule, async_pipe

        @async_rule
        async def is_premium(user):
            return await db.is_premium(user.id)

        @async_pipe
        async def apply_discount(user):
            user.discount = 0.2
            return user

        @async_pipe
        async def charge_full_price(user):
            user.discount = 0
            return user

        pricing = async_if_then_else(is_premium, apply_discount, charge_full_price)
        ok, user = await pricing.run(user)

    Unlike OR (|) which is a fallback (try a, if fail try b), async_if_then_else
    explicitly branches based on the condition result.
    """
    return _AsyncIfThenElse(condition, then_branch, else_branch)


# =============================================================================
# Async Validation Support
# =============================================================================


class AsyncValidatingCombinator(AsyncCombinator[T]):
    """
    Base class for async combinators that support validation with error messages.

    Subclasses must implement the validate() method.
    """

    @abstractmethod
    async def validate(self, ctx: T) -> ValidationResult:
        """Run async validation and return result with errors."""
        ...

    def __and__(self, other: AsyncCombinator[T]) -> AsyncValidatingCombinator[T]:
        """Override & to create async validating AND."""
        return _AsyncValidatingAnd(self, other)

    def __or__(self, other: AsyncCombinator[T]) -> AsyncValidatingCombinator[T]:
        """Override | to create async validating OR."""
        return _AsyncValidatingOr(self, other)

    def __invert__(self) -> AsyncValidatingCombinator[T]:
        """Override ~ to create async validating NOT."""
        return _AsyncValidatingNot(self)


class AsyncValidatingPredicate(AsyncValidatingCombinator[T]):
    """
    An async predicate that provides an error message on failure.

    Example:
        @async_vrule(error="User must be an admin")
        async def is_admin(user):
            return await db.check_admin(user.id)

        result = await is_admin.validate(user)
        if not result.ok:
            print(result.errors)  # ["User must be an admin"]
    """

    def __init__(
        self,
        fn: Callable[[T], Any],
        name: str | None = None,
        error: str | Callable[[T], str] | None = None,
    ):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "async_predicate")
        self._error = error

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        result = await self.fn(ctx)
        return bool(result), ctx

    def get_error(self, ctx: T) -> str:
        """Get the error message for this predicate."""
        if self._error is None:
            return f"Check failed: {self.name}"
        if callable(self._error):
            return self._error(ctx)
        try:
            return self._error.format(ctx=ctx)
        except (KeyError, AttributeError, IndexError):
            return self._error

    async def validate(self, ctx: T) -> ValidationResult:
        """Run async validation and return result with errors."""
        ok, result = await self._execute(ctx)
        errors = [] if ok else [self.get_error(ctx)]
        return ValidationResult(ok=ok, errors=errors, ctx=result)

    def __repr__(self) -> str:
        return f"AsyncValidatingPredicate({self.name})"


class _AsyncValidatingAnd(AsyncValidatingCombinator[T]):
    """Async AND combinator that collects all validation errors."""

    def __init__(self, left: AsyncCombinator[T], right: AsyncCombinator[T]):
        self.left = left
        self.right = right

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        ok1, ctx = await self.left._execute(ctx)
        if not ok1:
            return False, ctx
        return await self.right._execute(ctx)

    async def validate(self, ctx: T) -> ValidationResult:
        """Validate both sides and collect all errors."""
        errors: list[str] = []

        # Flatten the AND chain iteratively
        to_validate: list[AsyncCombinator[T]] = []
        stack: list[AsyncCombinator[T]] = [self]
        while stack:
            current = stack.pop()
            if isinstance(current, _AsyncValidatingAnd):
                stack.append(current.right)
                stack.append(current.left)
            else:
                to_validate.append(current)

        for item in to_validate:
            if isinstance(item, AsyncValidatingCombinator):
                result = await item.validate(ctx)
                errors.extend(result.errors)
                ctx = result.ctx
            else:
                ok, ctx = await item._execute(ctx)
                if not ok:
                    errors.append(f"Check failed: {_get_async_combinator_name(item)}")

        return ValidationResult(ok=len(errors) == 0, errors=errors, ctx=ctx)


class _AsyncValidatingOr(AsyncValidatingCombinator[T]):
    """Async OR combinator for validation - passes if any succeeds."""

    def __init__(self, left: AsyncCombinator[T], right: AsyncCombinator[T]):
        self.left = left
        self.right = right

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        ok1, ctx = await self.left._execute(ctx)
        if ok1:
            return True, ctx
        return await self.right._execute(ctx)

    async def validate(self, ctx: T) -> ValidationResult:
        """Validate - passes if any in the chain passes."""
        to_validate: list[AsyncCombinator[T]] = []
        stack: list[AsyncCombinator[T]] = [self]
        while stack:
            current = stack.pop()
            if isinstance(current, _AsyncValidatingOr):
                stack.append(current.right)
                stack.append(current.left)
            else:
                to_validate.append(current)

        last_result: ValidationResult | None = None
        for item in to_validate:
            if isinstance(item, AsyncValidatingCombinator):
                result = await item.validate(ctx)
                if result.ok:
                    return result
                last_result = result
            else:
                ok, result_ctx = await item._execute(ctx)
                if ok:
                    return ValidationResult(ok=True, errors=[], ctx=result_ctx)
                last_result = ValidationResult(
                    ok=False,
                    errors=[f"Check failed: {_get_async_combinator_name(item)}"],
                    ctx=result_ctx,
                )

        return last_result or ValidationResult(
            ok=False, errors=["No conditions to check"], ctx=ctx
        )


class _AsyncValidatingNot(AsyncValidatingCombinator[T]):
    """Async NOT combinator for validation - inverts the result."""

    def __init__(self, inner: AsyncCombinator[T], error: str | None = None):
        self.inner = inner
        self._error = error

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        ok, ctx = await self.inner._execute(ctx)
        return not ok, ctx

    async def validate(self, ctx: T) -> ValidationResult:
        """Validate - inverts the inner result."""
        if isinstance(self.inner, AsyncValidatingCombinator):
            inner_result = await self.inner.validate(ctx)
            if inner_result.ok:
                error_msg = self._error or "NOT condition failed (inner passed)"
                return ValidationResult(
                    ok=False, errors=[error_msg], ctx=inner_result.ctx
                )
            else:
                return ValidationResult(ok=True, errors=[], ctx=inner_result.ctx)
        else:
            ok, result = await self.inner._execute(ctx)
            if ok:
                error_msg = (
                    self._error
                    or f"NOT {_get_async_combinator_name(self.inner)} failed"
                )
                return ValidationResult(ok=False, errors=[error_msg], ctx=result)
            else:
                return ValidationResult(ok=True, errors=[], ctx=result)


@overload
def async_vrule(
    fn: Callable[[T], Any], *, error: str | Callable[[T], str] | None = None
) -> AsyncValidatingPredicate[T]: ...


@overload
def async_vrule(
    fn: None = None, *, error: str | Callable[[T], str] | None = None
) -> Callable[[Callable[[T], Any]], AsyncValidatingPredicate[T]]: ...


def async_vrule(
    fn: Callable[[T], Any] | None = None,
    *,
    error: str | Callable[[T], str] | None = None,
) -> (
    AsyncValidatingPredicate[T]
    | Callable[[Callable[[T], Any]], AsyncValidatingPredicate[T]]
):
    """
    Decorator to create an async validating rule with an error message.

    Example:
        @async_vrule(error="User {ctx.name} must be an admin")
        async def is_admin(user):
            return await db.check_admin(user.id)

        @async_vrule(error=lambda u: f"{u.name} is banned!")
        async def is_not_banned(user):
            return not await db.is_banned(user.id)

        result = await is_admin.validate(user)
        result = await (is_admin & is_not_banned).validate(user)  # Collects all errors
    """

    def decorator(f: Callable[[T], Any]) -> AsyncValidatingPredicate[T]:
        return AsyncValidatingPredicate(f, f.__name__, error)

    if fn is not None:
        return decorator(fn)
    return decorator


@overload
def async_vrule_args(
    fn: Callable[..., Any], *, error: str | Callable[..., str] | None = None
) -> Callable[..., AsyncValidatingPredicate]: ...


@overload
def async_vrule_args(
    fn: None = None, *, error: str | Callable[..., str] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., AsyncValidatingPredicate]]: ...


def async_vrule_args(
    fn: Callable[..., Any] | None = None,
    *,
    error: str | Callable[..., str] | None = None,
) -> (
    Callable[..., AsyncValidatingPredicate]
    | Callable[[Callable[..., Any]], Callable[..., AsyncValidatingPredicate]]
):
    """
    Decorator to create a parameterized async validating rule factory.

    Example:
        @async_vrule_args(error="Score {score} is below minimum {min_score}")
        async def score_above(user, min_score):
            score = await db.get_score(user.id)
            return score >= min_score

        result = await score_above(700).validate(user)
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., AsyncValidatingPredicate]:
        sig = inspect.signature(f)

        def factory(*args: Any, **kwargs: Any) -> AsyncValidatingPredicate:
            name = f"{f.__name__}({', '.join(map(repr, args))})"

            def get_bound_params():
                try:
                    bound = sig.bind_partial(None, *args, **kwargs)
                    bound.apply_defaults()
                    params = dict(bound.arguments)
                    first_param_name = list(sig.parameters.keys())[0]
                    if first_param_name in params:
                        del params[first_param_name]
                    return params
                except TypeError:
                    return kwargs

            err_msg: str | Callable[[Any], str] | None

            if error is None:
                err_msg = None
            elif callable(error):
                error_fn: Callable[..., str] = error

                def make_error_fn(ctx: Any) -> str:
                    return error_fn(ctx, *args, **kwargs)

                err_msg = make_error_fn
            else:
                template_str: str = error

                def make_formatted_error(ctx: Any) -> str:
                    format_context = {f"arg{i}": v for i, v in enumerate(args)}
                    format_context.update(get_bound_params())
                    format_context["ctx"] = ctx
                    try:
                        return template_str.format(**format_context)
                    except (KeyError, IndexError, AttributeError):
                        return template_str

                err_msg = make_formatted_error

            async def predicate_fn(ctx: Any) -> bool:
                return await f(ctx, *args, **kwargs)

            return AsyncValidatingPredicate(predicate_fn, name, err_msg)

        factory.__name__ = f.__name__
        return factory

    if fn is not None:
        return decorator(fn)
    return decorator


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

        # With observability hook
        def on_retry(attempt, error, delay):
            print(f"Retry {attempt}: {error}, waiting {delay}s")

        fetch = Retry(fetch_from_api, max_attempts=3, on_retry=on_retry)

    Args:
        inner: The combinator or callable to retry
        max_attempts: Maximum number of attempts (default: 3)
        backoff: Base delay between retries in seconds (default: 0)
        exponential: Use exponential backoff (default: False)
        jitter: Random jitter to add to delay (default: 0)
        name: Optional name for debugging
        on_retry: Optional callback(attempt, error, delay) called before each retry
    """

    def __init__(
        self,
        inner: Combinator[T] | Callable[[T], T],
        max_attempts: int = 3,
        backoff: float = 0.0,
        exponential: bool = False,
        jitter: float = 0.0,
        name: str | None = None,
        on_retry: Callable[[int, Exception | None, float], None] | None = None,
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
        self.on_retry = on_retry
        self.last_error: Exception | None = None
        self.attempts_made: int = 0

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
        self.last_error = None
        self.attempts_made = 0

        for attempt in range(self.max_attempts):
            self.attempts_made = attempt + 1
            try:
                ok, result = self.inner._execute(last_ctx)
                if ok:
                    return True, result
                last_ctx = result
                self.last_error = None
            except Exception as e:
                self.last_error = e
                # Continue to retry on exception

            # Don't sleep after last attempt
            if attempt < self.max_attempts - 1:
                delay = self._get_delay(attempt)

                # Call the retry hook if provided
                if self.on_retry is not None:
                    self.on_retry(attempt + 1, self.last_error, delay)

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

        # With observability hook
        async def on_retry(attempt, error, delay):
            print(f"Retry {attempt}: {error}, waiting {delay}s")

        fetch = AsyncRetry(fetch_from_api, max_attempts=3, on_retry=on_retry)

    Args:
        inner: The async combinator or callable to retry
        max_attempts: Maximum number of attempts (default: 3)
        backoff: Base delay between retries in seconds (default: 0)
        exponential: Use exponential backoff (default: False)
        jitter: Random jitter to add to delay (default: 0)
        name: Optional name for debugging
        on_retry: Optional async callback(attempt, error, delay) called before each retry
    """

    def __init__(
        self,
        inner: AsyncCombinator[T] | Callable[[T], Any],
        max_attempts: int = 3,
        backoff: float = 0.0,
        exponential: bool = False,
        jitter: float = 0.0,
        name: str | None = None,
        on_retry: Callable[[int, Exception | None, float], Any] | None = None,
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
        self.on_retry = on_retry
        self.last_error: Exception | None = None
        self.attempts_made: int = 0

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

    async def _execute(self, ctx: T) -> tuple[bool, T]:
        last_ctx = ctx
        self.last_error = None
        self.attempts_made = 0

        for attempt in range(self.max_attempts):
            self.attempts_made = attempt + 1
            try:
                ok, result = await self.inner._execute(last_ctx)
                if ok:
                    return True, result
                last_ctx = result
                self.last_error = None
            except Exception as e:
                self.last_error = e
                # Continue to retry

            # Don't sleep after last attempt
            if attempt < self.max_attempts - 1:
                delay = self._get_delay(attempt)

                # Call the retry hook if provided (supports both sync and async)
                if self.on_retry is not None:
                    result = self.on_retry(attempt + 1, self.last_error, delay)
                    if asyncio.iscoroutine(result):
                        await result

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

    By default, the end hour is exclusive (e.g., during_hours(9, 17) means
    9:00-16:59). Set inclusive_end=True to include the end hour.

    Example:
        # 9:00 AM to 4:59 PM (end exclusive, default)
        business_hours = during_hours(9, 17)

        # 9:00 AM to 5:59 PM (end inclusive)
        business_hours = during_hours(9, 17, inclusive_end=True)

        # Overnight: 10:00 PM to 5:59 AM
        night_shift = during_hours(22, 6)

        # With timezone
        trading_hours = during_hours(9, 16, tz="America/New_York")

    Args:
        start_hour: Start hour (0-23), inclusive
        end_hour: End hour (0-23), exclusive by default
        tz: Optional timezone name (e.g., "America/New_York")
        inclusive_end: If True, include the end hour (default: False)
    """

    def __init__(
        self,
        start_hour: int,
        end_hour: int,
        tz: str | None = None,
        inclusive_end: bool = False,
    ):
        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
            raise ValueError("Hours must be 0-23")

        self.start_hour = start_hour
        self.end_hour = end_hour
        self.tz = tz
        self.inclusive_end = inclusive_end

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
            if self.inclusive_end:
                ok = self.start_hour <= hour <= self.end_hour
            else:
                ok = self.start_hour <= hour < self.end_hour
        else:
            # Overnight range (e.g., 22 to 6)
            if self.inclusive_end:
                ok = hour >= self.start_hour or hour <= self.end_hour
            else:
                ok = hour >= self.start_hour or hour < self.end_hour

        return ok, ctx

    def __repr__(self) -> str:
        if self.inclusive_end:
            return (
                f"during_hours({self.start_hour}, {self.end_hour}, inclusive_end=True)"
            )
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
        IF/THEN/ELSE, ? : - Conditional branching (lowest precedence)
        |, OR       - Any must pass
        >>, THEN    - Sequence (run both)
        &, AND      - All must pass
        ~, NOT, !   - Invert result
        :modifier   - Apply modifier (highest precedence)

    Grouping:
        ( )         - Override precedence

    Rules:
        rule_name                   - Simple rule
        rule_name(arg)              - Rule with one argument
        rule_name(arg1, arg2)       - Rule with multiple arguments

    Conditional (if/else):
        IF condition THEN action ELSE alternative
        condition ? action : alternative

    Modifiers (postfix syntax):
        rule:retry(n)               - Retry up to n times
        rule:retry(n, backoff)      - With backoff delay in seconds
        rule:retry(n, backoff, true)  - Exponential backoff
        rule:retry(n, backoff, true, jitter)  - With jitter
        rule:cached                 - Cache results within use_cache() scope
        (expr):modifier             - Apply to grouped expression
        rule:mod1:mod2              - Chain multiple modifiers

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

        # Conditional
        IF is_premium THEN apply_discount ELSE charge_full
        is_premium ? apply_discount : charge_full

        # Modifiers
        fetch_data:retry(3)
        fetch_data:retry(5, 1.0, true)
        expensive_check:cached
        (primary | fallback):retry(3)

        # Multi-line
        is_admin
        & ~is_banned
        & account_older_than(30)
    """

    # Token types
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    THEN = "THEN"  # >> operator and THEN keyword
    IF = "IF"
    ELSE = "ELSE"
    QUESTION = "QUESTION"  # ?
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    COLON = "COLON"  # For modifier syntax (:retry, :cached) and ternary
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOL = "BOOL"  # For true/false literals
    EOF = "EOF"

    # Reserved modifier keywords
    MODIFIERS = {"retry", "cached"}

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
            elif (
                ch == ">"
                and self.pos + 1 < len(self.text)
                and self.text[self.pos + 1] == ">"
            ):
                self.tokens.append((self.THEN, ">>"))
                self.pos += 2
            elif ch in "~!":
                self.tokens.append((self.NOT, ch))
                self.pos += 1
            elif ch == "?":
                self.tokens.append((self.QUESTION, "?"))
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
            elif ch == ":":
                self.tokens.append((self.COLON, ":"))
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
                lower = ident.lower()
                if upper == "AND":
                    self.tokens.append((self.AND, ident))
                elif upper == "OR":
                    self.tokens.append((self.OR, ident))
                elif upper == "NOT":
                    self.tokens.append((self.NOT, ident))
                elif upper == "IF":
                    self.tokens.append((self.IF, ident))
                elif upper == "THEN":
                    # Same token as >> operator - context determines meaning
                    self.tokens.append((self.THEN, ident))
                elif upper == "ELSE":
                    self.tokens.append((self.ELSE, ident))
                elif lower in ("true", "false"):
                    self.tokens.append((self.BOOL, lower == "true"))
                else:
                    self.tokens.append((self.IDENT, ident))

            else:
                raise ValueError(f"Unexpected character: {ch!r} at position {self.pos}")

        self.tokens.append((self.EOF, None))

    def _read_string(self, quote: str) -> str:
        """Read a quoted string with escape sequence processing."""
        self.pos += 1  # skip opening quote
        result = []
        while self.pos < len(self.text) and self.text[self.pos] != quote:
            if self.text[self.pos] == "\\" and self.pos + 1 < len(self.text):
                next_ch = self.text[self.pos + 1]
                # Handle common escape sequences
                escape_map = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}
                if next_ch in escape_map:
                    result.append(escape_map[next_ch])
                else:
                    # For \' or \" or any other, just use the escaped char
                    result.append(next_ch)
                self.pos += 2
            else:
                result.append(self.text[self.pos])
                self.pos += 1
        if self.pos >= len(self.text):
            raise ValueError("Unterminated string literal")
        self.pos += 1  # skip closing quote
        return "".join(result)

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
            expr      = if_expr
            if_expr   = 'IF' or_expr 'THEN' or_expr 'ELSE' or_expr
                      | or_expr ('?' or_expr ':' or_expr)?
            or_expr   = then_expr (('|' | 'OR') then_expr)*
            then_expr = and_expr (('>>' | 'THEN') and_expr)*
            and_expr  = not_expr (('&' | 'AND') not_expr)*
            not_expr  = ('~' | 'NOT' | '!')? postfix
            postfix   = primary (':' MODIFIER args?)*
            primary   = IDENT args? | '(' expr ')'
            args      = '(' arg_list? ')'
            arg_list  = arg (',' arg)*
            arg       = NUMBER | STRING | IDENT | BOOL

        Conditionals:
            IF condition THEN then_branch ELSE else_branch
            condition ? then_branch : else_branch

        Modifiers:
            :retry(max_attempts, [backoff], [exponential], [jitter])
            :cached
        """
        result = self._parse_if()
        if self._peek()[0] != self.EOF:
            raise ValueError(f"Unexpected token: {self._peek()}")
        return result

    def _parse_if(self) -> dict | str:
        """Parse IF/THEN/ELSE or ternary expression (lowest precedence)."""
        # Check for keyword IF
        if self._peek()[0] == self.IF:
            self._consume()  # consume IF
            # Parse condition allowing OR but not THEN sequences
            # This means IF a >> b THEN c ELSE d needs parentheses: IF (a >> b) THEN c ELSE d
            condition = self._parse_or_no_then()
            self._expect(self.THEN)  # expect THEN keyword
            then_branch = self._parse_if()  # Allow nested if/ternary in branches
            self._expect(self.ELSE)  # expect ELSE
            else_branch = self._parse_if()  # Allow nested if/ternary in branches
            return {"if": {"cond": condition, "then": then_branch, "else": else_branch}}

        # Otherwise parse or_expr and check for ternary ?:
        condition = self._parse_or()

        if self._peek()[0] == self.QUESTION:
            self._consume()  # consume ?
            then_branch = self._parse_if()  # Allow nested if/ternary
            self._expect(self.COLON)  # expect :
            else_branch = self._parse_if()  # Allow nested if/ternary
            return {"if": {"cond": condition, "then": then_branch, "else": else_branch}}

        return condition

    def _parse_or_no_then(self) -> dict | str:
        """Parse OR expression without THEN sequences (for IF conditions)."""
        left = self._parse_and()

        items = [left]
        while self._peek()[0] == self.OR:
            self._consume()
            items.append(self._parse_and())

        if len(items) == 1:
            return items[0]
        return {"or": items}

    def _parse_or(self) -> dict | str:
        """Parse OR expression."""
        left = self._parse_then()

        items = [left]
        while self._peek()[0] == self.OR:
            self._consume()
            items.append(self._parse_then())

        if len(items) == 1:
            return items[0]
        return {"or": items}

    def _parse_then(self) -> dict | str:
        """Parse THEN expression (sequence, runs both)."""
        left = self._parse_and()

        items = [left]
        while self._peek()[0] == self.THEN:
            self._consume()
            items.append(self._parse_and())

        if len(items) == 1:
            return items[0]
        return {"then": items}

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
        return self._parse_postfix()

    def _parse_postfix(self) -> dict | str:
        """Parse primary with optional :modifier suffixes."""
        result = self._parse_primary()

        # Check for chained modifiers like :retry(3):cached
        while self._peek()[0] == self.COLON:
            # Peek ahead - if this is part of ternary ?: we should stop
            # Save position in case we need to backtrack
            saved_pos = self.token_pos
            self._consume()  # consume ':'

            # Check if this looks like a modifier (IDENT)
            if self._peek()[0] != self.IDENT:
                # Not a modifier, backtrack (this is probably ternary :)
                self.token_pos = saved_pos
                break

            modifier_name = self._peek()[1]
            modifier = modifier_name.lower()

            # Peek at what comes after the identifier
            next_pos = self.token_pos + 1
            next_token = (
                self.tokens[next_pos][0] if next_pos < len(self.tokens) else self.EOF
            )

            if modifier not in self.MODIFIERS:
                # Only raise error if it looks like a modifier call (has parens)
                # This allows ternary `a ? b : c` to work (c is not followed by parens)
                if next_token == self.LPAREN:
                    raise ValueError(
                        f"Unknown modifier '{modifier_name}'. Valid modifiers: {', '.join(sorted(self.MODIFIERS))}"
                    )
                else:
                    # Could be ternary colon, backtrack
                    self.token_pos = saved_pos
                    break

            # It's a modifier, consume and process
            self._consume()  # consume modifier name

            # Check for optional arguments
            args: list[Any] = []
            if self._peek()[0] == self.LPAREN:
                self._consume()  # (
                args = self._parse_args()
                self._expect(self.RPAREN)  # )

            # Wrap result with modifier
            if modifier == "retry":
                result = {"retry": {"inner": result, "args": args}}
            elif modifier == "cached":
                result = {"cached": result}

        return result

    def _parse_primary(self) -> dict | str:
        """Parse primary expression (identifier or grouped expr)."""
        token = self._peek()

        if token[0] == self.LPAREN:
            self._consume()
            expr = self._parse_if()  # Allow full expressions including if/then/else
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
        if token[0] == self.BOOL:
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


class _CachedCombinatorWrapper(Combinator[T]):
    """
    Internal wrapper to cache any combinator's result.

    Used by Registry when :cached modifier is applied to non-Predicate combinators.
    The cache is keyed by object id and is shared across all instances.
    """

    _cache: dict[int, tuple[bool, Any]] = {}

    def __init__(self, inner: Combinator[T]):
        self.inner = inner

    def _execute(self, ctx: T) -> tuple[bool, T]:
        # Check if caching is enabled via use_cache()
        cache = _cache_store.get()
        if cache is not None:
            key = f"_wrapped:{id(self.inner)}:{id(ctx)}"
            if key in cache:
                return cache[key]
            result = self.inner._execute(ctx)
            cache[key] = result
            return result

        # No cache scope, just execute
        return self.inner._execute(ctx)

    def __repr__(self) -> str:
        return f"Cached({self.inner!r})"


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

            # Conditional (if/then/else)
            IF is_premium THEN apply_discount ELSE charge_full
            is_premium ? apply_discount : charge_full

            # Modifiers (postfix syntax)
            fetch_data:retry(3)                   # Retry up to 3 times
            fetch_data:retry(3, 1.0)              # With 1s backoff
            fetch_data:retry(3, 1.0, true)        # Exponential backoff
            fetch_data:retry(3, 1.0, true, 0.1)   # With jitter
            expensive_check:cached                # Cache results
            (fetch_a | fetch_b):retry(5)          # Retry grouped expr
            slow_query:cached:retry(3)            # Chain modifiers

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
            Modifiers (:) - highest, binds to immediate left
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

            elif key == "then":
                items = [self._build(item) for item in value]
                return self._combine_seq(items)

            elif key == "if":
                # If/then/else conditional
                condition = self._build(value["cond"])
                then_branch = self._build(value["then"])
                else_branch = self._build(value["else"])
                return _IfThenElse(condition, then_branch, else_branch)

            # Modifier: retry
            elif key == "retry":
                inner = self._build(value["inner"])
                args = value.get("args", [])

                # Parse retry args: max_attempts, [backoff], [exponential], [jitter]
                max_attempts = int(args[0]) if len(args) > 0 else 3
                backoff = float(args[1]) if len(args) > 1 else 0.0
                exponential = bool(args[2]) if len(args) > 2 else False
                jitter = float(args[3]) if len(args) > 3 else 0.0

                return Retry(
                    inner,
                    max_attempts=max_attempts,
                    backoff=backoff,
                    exponential=exponential,
                    jitter=jitter,
                )

            # Modifier: cached
            elif key == "cached":
                inner = self._build(value)
                # Wrap in CachedPredicate if it's a predicate
                if isinstance(inner, Predicate):
                    return CachedPredicate(inner.fn, inner.name)
                else:
                    # For non-predicates, wrap with a generic caching combinator
                    return _CachedCombinatorWrapper(inner)

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
