# Concurrency

## AsyncTimeout

```python
@dataclass
class AsyncTimeout(AsyncCombinator[T])
```

Wrap an async combinator with a timeout. If execution exceeds the timeout, returns `(False, ctx)` or calls the `on_timeout` handler.

**Fields:**

- `inner: AsyncCombinator[T]` -- The combinator to wrap
- `timeout: float` -- Timeout in seconds
- `on_timeout: Callable[[T], T] | None` -- Optional callback to modify context on timeout
- `timed_out: bool` -- Whether the last execution timed out (not concurrency-safe)

---

## with_timeout

```python
with_timeout(
    combinator: AsyncCombinator[T],
    timeout: float,
    on_timeout: Callable[[T], T] | None = None,
) -> AsyncTimeout[T]
```

Factory function for `AsyncTimeout`.

```python
from kompoz import with_timeout

result = await with_timeout(slow_api, timeout=5.0).run(ctx)
```

---

## AsyncLimited

```python
class AsyncLimited(AsyncCombinator[T])
```

Limit concurrent executions using a semaphore.

**Constructor:**

```python
AsyncLimited(
    inner: AsyncCombinator[T],
    max_concurrent: int,
    name: str | None = None,
)
```

**Parameters:**

- `inner` -- The combinator to wrap
- `max_concurrent` -- Maximum number of concurrent executions
- `name` -- Optional name to share semaphore across limiters

---

## limited

```python
limited(
    combinator: AsyncCombinator[T],
    max_concurrent: int,
    name: str | None = None,
) -> AsyncLimited[T]
```

Factory function for `AsyncLimited`.

```python
from kompoz import limited

# Instance-specific limit
limited_api = limited(api_check, max_concurrent=5)

# Shared limit across multiple combinators
check_a = limited(api_a, max_concurrent=10, name="api_pool")
check_b = limited(api_b, max_concurrent=10, name="api_pool")
```

---

## AsyncCircuitBreaker

```python
class AsyncCircuitBreaker(AsyncCombinator[T])
```

Circuit breaker pattern for fault tolerance.

**Constructor:**

```python
AsyncCircuitBreaker(
    inner: AsyncCombinator[T],
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 1,
    on_state_change: Callable | None = None,
)
```

**Properties:**

- `state: CircuitState` -- Current circuit state

**Methods:**

- `async run(ctx: T) -> tuple[bool, T]` -- Execute (or reject if circuit is open)
- `get_stats() -> CircuitBreakerStats` -- Get current statistics
- `async reset() -> None` -- Manually reset to CLOSED state

---

## circuit_breaker

```python
circuit_breaker(
    combinator: AsyncCombinator[T],
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 1,
    on_state_change: Callable | None = None,
) -> AsyncCircuitBreaker[T]
```

Factory function for `AsyncCircuitBreaker`.

```python
from kompoz import circuit_breaker

protected = circuit_breaker(flaky_api, failure_threshold=5, recovery_timeout=30.0)
```

---

## CircuitState

```python
class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Requests rejected
    HALF_OPEN = "half_open" # Testing recovery
```

---

## CircuitBreakerStats

```python
@dataclass
class CircuitBreakerStats:
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_success_time: float | None
    half_open_successes: int
```
