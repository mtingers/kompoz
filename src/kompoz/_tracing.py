"""Tracing, hooks, and explain functionality."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from kompoz._async import AsyncCombinator, _async_traced_run
from kompoz._caching import CachedPredicate
from kompoz._core import Combinator, _And, _Or, _Not, _Then, _IfThenElse, _is_composite
from kompoz._predicate import Predicate
from kompoz._retry import Retry
from kompoz._temporal import (
    after_date,
    before_date,
    between_dates,
    during_hours,
    on_days,
    on_weekdays,
)
from kompoz._transform import Transform
from kompoz._types import T, _trace_config, _trace_hook
from kompoz._utility import Always, Debug, Never, Try
from kompoz._validation import (
    ValidatingPredicate,
    _ValidatingAnd,
    _ValidatingNot,
    _ValidatingOr,
    _is_validating_composite,
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


