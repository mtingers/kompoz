# API Reference

Quick-lookup table of all public symbols in `kompoz`.

## Core

| Symbol | Kind | Description |
| --- | --- | --- |
| [`Combinator`](core.md#combinator) | class | Abstract base for all combinators |
| [`Predicate`](core.md#predicate) | class | Checks a condition, doesn't modify context |
| [`PredicateFactory`](core.md#predicatefactory) | class | Factory for parameterized predicates |
| [`Transform`](core.md#transform) | class | Transforms context, fails on exception |
| [`TransformFactory`](core.md#transformfactory) | class | Factory for parameterized transforms |
| [`Try`](core.md#try) | class | Wraps a function, converts exceptions to failure |
| [`Registry`](core.md#registry) | class | Register and load rules from expressions |
| [`ExpressionParser`](core.md#expressionparser) | class | Parser for rule expressions |

## Decorators

| Symbol | Kind | Description |
| --- | --- | --- |
| [`@rule`](decorators.md#rule) | decorator | Create a simple predicate |
| [`@rule_args`](decorators.md#rule_args) | decorator | Create a parameterized predicate factory |
| [`@pipe`](decorators.md#pipe) | decorator | Create a simple transform |
| [`@pipe_args`](decorators.md#pipe_args) | decorator | Create a parameterized transform factory |
| [`@vrule`](decorators.md#vrule) | decorator | Create a validating rule with error message |
| [`@vrule_args`](decorators.md#vrule_args) | decorator | Create a parameterized validating rule |
| [`@async_rule`](decorators.md#async_rule) | decorator | Create an async predicate |
| [`@async_rule_args`](decorators.md#async_rule_args) | decorator | Create a parameterized async predicate |
| [`@async_pipe`](decorators.md#async_pipe) | decorator | Create an async transform |
| [`@async_pipe_args`](decorators.md#async_pipe_args) | decorator | Create a parameterized async transform |
| [`@async_vrule`](decorators.md#async_vrule) | decorator | Create an async validating rule |
| [`@async_vrule_args`](decorators.md#async_vrule_args) | decorator | Create a parameterized async validating rule |
| [`@cached_rule`](decorators.md#cached_rule) | decorator | Create a rule with result caching |
| [`@async_cached_rule`](decorators.md#async_cached_rule) | decorator | Create an async rule with result caching |

## Functions

| Symbol | Kind | Description |
| --- | --- | --- |
| [`explain`](functions.md#explain) | function | Generate plain English explanation of a rule |
| [`if_then_else`](functions.md#if_then_else) | function | Create conditional combinator |
| [`async_if_then_else`](functions.md#async_if_then_else) | function | Create async conditional combinator |
| [`parse_expression`](functions.md#parse_expression) | function | Parse expression string into config dict |
| [`use_tracing`](functions.md#use_tracing) | function | Context manager to enable tracing |
| [`run_traced`](functions.md#run_traced) | function | Run with explicit tracing |
| [`run_async_traced`](functions.md#run_async_traced) | function | Run async combinator with tracing |
| [`use_cache`](functions.md#use_cache) | function | Context manager to enable caching |
| [`parallel_and`](functions.md#parallel_and) | function | Async AND that runs children concurrently |

## Async

| Symbol | Kind | Description |
| --- | --- | --- |
| [`AsyncCombinator`](async.md#asynccombinator) | class | Base class for async combinators |
| [`AsyncPredicate`](async.md#asyncpredicate) | class | Async predicate |
| [`AsyncPredicateFactory`](async.md#asyncpredicatefactory) | class | Factory for parameterized async predicates |
| [`AsyncTransform`](async.md#asynctransform) | class | Async transform with error tracking |
| [`AsyncTransformFactory`](async.md#asynctransformfactory) | class | Factory for parameterized async transforms |
| [`AsyncRetry`](retry-caching.md#asyncretry) | class | Async retry with backoff |

## Validation

| Symbol | Kind | Description |
| --- | --- | --- |
| [`ValidationResult`](validation.md#validationresult) | class | Result with ok, errors, and ctx |
| [`ValidatingCombinator`](validation.md#validatingcombinator) | class | Base for validating combinators |
| [`ValidatingPredicate`](validation.md#validatingpredicate) | class | Predicate with error message |
| [`AsyncValidatingCombinator`](validation.md#asyncvalidatingcombinator) | class | Async base for validating combinators |
| [`AsyncValidatingPredicate`](validation.md#asyncvalidatingpredicate) | class | Async predicate with error message |

## Tracing

| Symbol | Kind | Description |
| --- | --- | --- |
| [`TraceHook`](tracing.md#tracehook) | protocol | Protocol for custom trace hooks |
| [`TraceConfig`](tracing.md#traceconfig) | class | Configuration for tracing behavior |
| [`PrintHook`](tracing.md#printhook) | class | Simple stdout tracing |
| [`LoggingHook`](tracing.md#logginghook) | class | Python logging integration |
| [`OpenTelemetryHook`](tracing.md#opentelemetryhook) | class | OpenTelemetry integration |

## Retry & Caching

| Symbol | Kind | Description |
| --- | --- | --- |
| [`Retry`](retry-caching.md#retry) | class | Retry with configurable backoff |
| [`AsyncRetry`](retry-caching.md#asyncretry) | class | Async retry with backoff |
| [`RetryResult`](retry-caching.md#retryresult) | class | Result from `run_with_info()` |
| [`CachedPredicate`](retry-caching.md#cachedpredicate) | class | Predicate with result caching |
| [`AsyncCachedPredicate`](retry-caching.md#asynccachedpredicate) | class | Async predicate with caching |

## Concurrency

| Symbol | Kind | Description |
| --- | --- | --- |
| [`AsyncTimeout`](concurrency.md#asynctimeout) | class | Timeout wrapper |
| [`with_timeout`](concurrency.md#with_timeout) | function | Create timeout wrapper |
| [`AsyncLimited`](concurrency.md#asynclimited) | class | Concurrency limiter |
| [`limited`](concurrency.md#limited) | function | Create concurrency limiter |
| [`AsyncCircuitBreaker`](concurrency.md#asynccircuitbreaker) | class | Circuit breaker pattern |
| [`circuit_breaker`](concurrency.md#circuit_breaker) | function | Create circuit breaker |
| [`CircuitState`](concurrency.md#circuitstate) | enum | Circuit breaker states |
| [`CircuitBreakerStats`](concurrency.md#circuitbreakerstats) | class | Circuit breaker statistics |

## Temporal

| Symbol | Kind | Description |
| --- | --- | --- |
| [`during_hours`](temporal.md#during_hours) | function | Check if current hour is in range |
| [`on_weekdays`](temporal.md#on_weekdays) | function | Check if today is Monday-Friday |
| [`on_days`](temporal.md#on_days) | function | Check if today is one of the specified days |
| [`after_date`](temporal.md#after_date) | function | Check if today is after date |
| [`before_date`](temporal.md#before_date) | function | Check if today is before date |
| [`between_dates`](temporal.md#between_dates) | function | Check if today is in date range |

## Utilities

| Symbol | Kind | Description |
| --- | --- | --- |
| [`Always`](utilities.md#always) | class | Always succeeds |
| [`Never`](utilities.md#never) | class | Always fails |
| [`Debug`](utilities.md#debug) | class | Prints context and succeeds |
