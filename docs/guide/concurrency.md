# Concurrency

Utilities for controlling async execution: timeouts, rate limiting, and circuit breakers.

## Timeout

Wrap an async combinator with a timeout. If execution doesn't complete in time, it returns failure:

```python
from kompoz import with_timeout, async_rule

@async_rule
async def slow_api_check(ctx):
    return await api.check(ctx.id)

# Timeout after 5 seconds
result = await with_timeout(slow_api_check, timeout=5.0).run(ctx)

# With custom timeout handler
def handle_timeout(ctx):
    ctx.timed_out = True
    return ctx

result = await with_timeout(
    slow_api_check,
    timeout=5.0,
    on_timeout=handle_timeout
).run(ctx)
```

## Concurrency Limiting

Limit concurrent executions using a semaphore. Useful for rate-limiting API calls or database connections:

```python
from kompoz import limited

# Max 5 concurrent API calls
limited_api = limited(api_check, max_concurrent=5)

# Run many tasks, but only 5 at a time
results = await asyncio.gather(*[
    limited_api.run(ctx) for ctx in many_contexts
])
```

### Shared Semaphores

Use named semaphores to share a concurrency limit across multiple combinators:

```python
# Shared limit across multiple combinators
check_a = limited(api_check_a, max_concurrent=10, name="api_pool")
check_b = limited(api_check_b, max_concurrent=10, name="api_pool")
# check_a and check_b share the same 10-slot semaphore
```

## Circuit Breaker

Implements the circuit breaker pattern for fault tolerance. The circuit monitors failures and "trips" (opens) when failures exceed a threshold, preventing cascading failures.

### States

- **CLOSED** -- Normal operation, requests pass through
- **OPEN** -- Too many failures, requests rejected immediately
- **HALF_OPEN** -- Testing if service recovered

### Basic Usage

```python
from kompoz import circuit_breaker, CircuitState

protected = circuit_breaker(
    flaky_api,
    failure_threshold=5,    # Open after 5 failures
    recovery_timeout=30.0,  # Try again after 30 seconds
)

ok, result = await protected.run(ctx)

# Check circuit state
if protected.state == CircuitState.OPEN:
    print("Circuit is open, service likely down")

# Get detailed stats
stats = protected.get_stats()
print(f"Failures: {stats.failure_count}")
```

### State Change Callbacks

```python
def on_state_change(old_state, new_state, stats):
    print(f"Circuit {old_state} -> {new_state}")

protected = circuit_breaker(
    flaky_api,
    failure_threshold=5,
    recovery_timeout=30.0,
    on_state_change=on_state_change,
)
```

### Manual Reset

```python
await protected.reset()  # Force back to CLOSED state
```
