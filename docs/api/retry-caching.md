# Retry & Caching

## Retry

```python
class Retry(Combinator[T])
```

Retry combinator with configurable backoff.

**Constructor:**

```python
Retry(
    inner: Combinator[T],
    max_attempts: int = 3,
    backoff: float = 0.0,
    exponential: bool = False,
    jitter: float = 0.0,
    on_retry: Callable[[int, Exception | None, float], None] | None = None,
)
```

**Parameters:**

- `inner` -- The combinator to retry
- `max_attempts` -- Maximum number of attempts (>= 1)
- `backoff` -- Initial delay between retries in seconds (>= 0)
- `exponential` -- Whether to double the delay each attempt
- `jitter` -- Random jitter added to delay (>= 0)
- `on_retry` -- Optional callback `(attempt, error, delay)` called before each retry

**Attributes:**

- `last_error: Exception | None` -- Error from last attempt (not thread-safe)
- `attempts_made: int` -- Number of attempts in last run (not thread-safe)

**Methods:**

- `run(ctx: T) -> tuple[bool, T]` -- Execute with retries
- `run_with_info(ctx: T) -> RetryResult[T]` -- Thread-safe alternative returning full metadata

---

## AsyncRetry

```python
class AsyncRetry(AsyncCombinator[T])
```

Async retry with the same interface as `Retry`. The `on_retry` callback can be sync or async.

**Methods:**

- `async run(ctx: T) -> tuple[bool, T]` -- Execute with retries
- `async run_with_info(ctx: T) -> RetryResult[T]` -- Concurrency-safe alternative

---

## RetryResult

```python
@dataclass
class RetryResult(Generic[T]):
    ok: bool
    ctx: T
    attempts_made: int
    last_error: Exception | None
```

Result from `Retry.run_with_info()` or `AsyncRetry.run_with_info()`.

**Example:**

```python
info = fetch.run_with_info(request)
print(f"ok={info.ok}, attempts={info.attempts_made}, error={info.last_error}")
```

---

## CachedPredicate

```python
class CachedPredicate(Predicate[T])
```

Predicate with result caching. Results are cached within a `use_cache()` scope. Created by `@cached_rule`.

---

## AsyncCachedPredicate

```python
class AsyncCachedPredicate(AsyncPredicate[T])
```

Async predicate with result caching. Created by `@async_cached_rule`.
