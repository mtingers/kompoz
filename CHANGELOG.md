# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-01

### Fixed

- Replace `assert` statements in `OpenTelemetryHook` with explicit `RuntimeError` raises (safe under `python -O`)
- Add input validation to `Retry` and `AsyncRetry` constructors (`max_attempts >= 1`, `backoff >= 0`, `jitter >= 0`)
- Clean up per-key locks in `AsyncCachedPredicate` to prevent unbounded memory growth

### Added

- GitHub Actions CI workflow (lint, typecheck, test matrix across Python 3.10-3.13)
- mypy configuration and dev dependency
- Coverage configuration (`fail_under = 85`)
- Pre-commit hooks configuration
- `Makefile` with common development targets
- `SECURITY.md` with vulnerability reporting instructions
- `CHANGELOG.md`
- Project URLs in `pyproject.toml`

### Changed

- Expanded Ruff lint rules to include `B` (bugbear), `SIM` (simplify), `RUF` (ruff-specific)
- Widened timing thresholds in tests for CI reliability

## [0.1.0] - 2026-01-31

### Added

- Core combinator framework (`Combinator`, `Predicate`, `Transform`)
- Operator overloading (`&`, `|`, `~`, `>>`)
- Async combinators (`AsyncCombinator`, `async_rule`, `async_transform`)
- Validation combinators with error collection (`vrule`, `async_vrule`)
- Caching support (`cached_rule`, `async_cached_rule`, `use_cache`, `use_cache_shared`)
- Retry combinators with configurable backoff (`Retry`, `AsyncRetry`)
- Concurrency utilities (`parallel_or`, `parallel_and`, `with_timeout`, `limited`, `circuit_breaker`)
- Temporal predicates (`during_hours`, `on_weekdays`, `on_days`, `after_date`, `before_date`, `between_dates`)
- Expression parser and registry for config-driven rules
- Tracing and explain functionality (`use_tracing`, `PrintHook`, `LoggingHook`, `OpenTelemetryHook`, `explain`)
- Utility combinators (`Always`, `Never`, `Debug`, `Try`)
