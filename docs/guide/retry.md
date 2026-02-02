# Retry Logic

Retry failed operations with configurable backoff.

## Basic Retry

```python
from kompoz import Retry

# Simple retry
fetch = Retry(fetch_from_api, max_attempts=3)

# Exponential backoff
fetch = Retry(
    fetch_from_api,
    max_attempts=5,
    backoff=1.0,       # Initial delay in seconds
    exponential=True,  # Double delay each attempt
    jitter=0.1         # Random jitter to avoid thundering herd
)

ok, result = fetch.run(request)
```

## Observability Hooks

Retry combinators support observability via callbacks and state tracking:

```python
from kompoz import Retry, AsyncRetry

# Callback for monitoring retries
def on_retry(attempt: int, error: Exception | None, delay: float):
    print(f"Retry {attempt}: error={error}, waiting {delay}s")
    metrics.increment("api.retries", tags={"attempt": attempt})

fetch = Retry(
    fetch_from_api,
    max_attempts=3,
    backoff=1.0,
    on_retry=on_retry  # Called before each retry
)

ok, result = fetch.run(request)

# After execution, check state
print(f"Total attempts: {fetch.attempts_made}")
print(f"Last error: {fetch.last_error}")
```

### Thread-safe alternative: `run_with_info()`

Like `last_error` on transforms, the `attempts_made` and `last_error` attributes on
`Retry` / `AsyncRetry` are mutated on each call. Use `run_with_info()` for a pure
alternative that returns all metadata in a `RetryResult`:

```python
info = fetch.run_with_info(request)
print(f"ok={info.ok}, attempts={info.attempts_made}, error={info.last_error}")
```

!!! note
    `run_with_info()` is available on both `Retry` and `AsyncRetry`.

## Async Retry

For async retries, the callback can be sync or async:

```python
async def on_retry_async(attempt, error, delay):
    await log_to_service(f"Retry {attempt}")

fetch = AsyncRetry(
    fetch_from_api,
    max_attempts=3,
    on_retry=on_retry_async  # Async callback supported
)
```

## DSL Integration

Use the `:retry` modifier in DSL expressions:

```python
loaded = reg.load("fetch_user:retry(3)")              # Retry up to 3 times
loaded = reg.load("fetch_user:retry(3, 1.0)")         # With 1s backoff
loaded = reg.load("fetch_user:retry(3, 1.0, true)")   # Exponential backoff
```
