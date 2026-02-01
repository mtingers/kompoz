"""Shared types, TypeVars, and ContextVars used across all modules."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class TraceHook(Protocol):
    """
    Protocol for trace hooks.

    Implement this to integrate with logging, OpenTelemetry, or other
    tracing systems.
    """

    def on_enter(self, name: str, ctx: Any, depth: int) -> Any: ...

    def on_exit(self, span: Any, name: str, ok: bool, duration_ms: float, depth: int) -> None: ...

    def on_error(
        self, span: Any, name: str, error: Exception, duration_ms: float, depth: int
    ) -> None: ...


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


# Type aliases for callbacks
RetryCallback = Callable[[int, Exception | None, float], None]
"""Callback signature for retry hooks: (attempt, error, delay) -> None"""

AsyncRetryCallback = Callable[[int, Exception | None, float], Any]
"""Callback signature for async retry hooks: (attempt, error, delay) -> None or Coroutine"""

# Context variables for global tracing
_trace_hook: ContextVar[TraceHook | None] = ContextVar("trace_hook", default=None)
_trace_config: ContextVar[TraceConfig] = ContextVar("trace_config", default=TraceConfig())

# Context variable for caching scope
_cache_store: ContextVar[dict[str, tuple[bool, Any]] | None] = ContextVar(
    "cache_store", default=None
)
