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
    # Parallel Async
    "parallel_and",
    "parallel_or",
    # Caching
    "CachedPredicate",
    "CachedPredicateFactory",
    "use_cache",
    "use_cache_shared",
    "cached_rule",
    "AsyncCachedPredicate",
    "AsyncCachedPredicateFactory",
    "async_cached_rule",
    # Retry
    "Retry",
    "AsyncRetry",
    "RetryResult",
    # Concurrency utilities
    "AsyncTimeout",
    "with_timeout",
    "AsyncLimited",
    "limited",
    "AsyncCircuitBreaker",
    "circuit_breaker",
    "CircuitState",
    "CircuitBreakerStats",
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

# Core
# Async
from kompoz._async import (
    AsyncCombinator,
    AsyncPredicate,
    AsyncPredicateFactory,
    AsyncTransform,
    AsyncTransformFactory,
    async_if_then_else,
    async_pipe,
    async_pipe_args,
    async_rule,
    async_rule_args,
)

# Async validation
from kompoz._async_validation import (
    AsyncValidatingCombinator,
    AsyncValidatingPredicate,
    async_vrule,
    async_vrule_args,
    parallel_and,
    parallel_or,
)

# Caching
from kompoz._caching import (
    AsyncCachedPredicate,
    AsyncCachedPredicateFactory,
    CachedPredicate,
    CachedPredicateFactory,
    async_cached_rule,
    cached_rule,
    use_cache,
    use_cache_shared,
)

# Concurrency utilities
from kompoz._concurrency import (
    AsyncCircuitBreaker,
    AsyncLimited,
    AsyncTimeout,
    CircuitBreakerStats,
    CircuitState,
    circuit_breaker,
    limited,
    with_timeout,
)
from kompoz._core import Combinator, if_then_else

# Predicates
from kompoz._predicate import (
    Predicate,
    PredicateFactory,
    predicate,
    predicate_factory,
    rule,
    rule_args,
)

# Registry and expression parsing
from kompoz._registry import ExpressionParser, Registry, parse_expression

# Retry
from kompoz._retry import AsyncRetry, Retry, RetryResult

# Temporal predicates
from kompoz._temporal import (
    after_date,
    before_date,
    between_dates,
    during_hours,
    on_days,
    on_weekdays,
)

# Tracing
from kompoz._tracing import (
    LoggingHook,
    OpenTelemetryHook,
    PrintHook,
    TraceConfig,
    TraceHook,
    explain,
    run_async_traced,
    run_traced,
    use_tracing,
)

# Transforms
from kompoz._transform import (
    Transform,
    TransformFactory,
    pipe,
    pipe_args,
    transform,
    transform_factory,
)

# Utility combinators
from kompoz._utility import Always, Debug, Never, Try

# Validation
from kompoz._validation import (
    ValidatingCombinator,
    ValidatingPredicate,
    ValidationResult,
    vrule,
    vrule_args,
)
