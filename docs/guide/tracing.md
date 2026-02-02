# Tracing & Debugging

## Explain Rules

Generate plain English explanations of what a rule does:

```python
from kompoz import explain

rule = is_admin | (is_active & ~is_banned & account_older_than(30))
print(explain(rule))

# Output:
# Check passes if ANY of:
#   * Check: is_admin
#   * ALL of:
#     * Check: is_active
#     * NOT: is_banned
#     * Check: account_older_than(30)
```

## Tracing Execution

Trace rule execution with built-in hooks or custom implementations:

```python
from kompoz import use_tracing, run_traced, PrintHook, TraceConfig

# Option 1: Context manager (traces all run() calls in scope)
with use_tracing(PrintHook()):
    rule.run(user)

# Option 2: Explicit tracing
run_traced(rule, user, PrintHook())
```

Output:

```
-> OR
  -> Predicate(is_admin)
  <- Predicate(is_admin) ✗ (0.02ms)
  -> AND
    -> Predicate(is_active)
    <- Predicate(is_active) ✓ (0.01ms)
  <- AND ✓ (0.15ms)
<- OR ✓ (0.20ms)
```

## Async Tracing

Async combinators fully support tracing via the same `use_tracing()` context manager:

```python
from kompoz import use_tracing, run_async_traced, PrintHook, async_rule

@async_rule
async def check_permission(user):
    return await db.has_permission(user.id)

@async_rule
async def check_quota(user):
    return await db.check_quota(user.id)

can_proceed = check_permission & check_quota

# Option 1: Context manager works with async
with use_tracing(PrintHook()):
    ok, result = await can_proceed.run(user)

# Option 2: Explicit async tracing
ok, result = await run_async_traced(can_proceed, user, PrintHook())
```

Output:

```
-> AsyncAND
  -> AsyncPredicate(check_permission)
  <- AsyncPredicate(check_permission) ✓ (15.23ms)
  -> AsyncPredicate(check_quota)
  <- AsyncPredicate(check_quota) ✓ (8.41ms)
<- AsyncAND ✓ (23.89ms)
```

## Trace Configuration

```python
from kompoz import TraceConfig

# Trace only leaf predicates (skip AND/OR/NOT)
with use_tracing(PrintHook(), TraceConfig(include_leaf_only=True)):
    rule.run(user)

# Limit trace depth
with use_tracing(PrintHook(), TraceConfig(max_depth=2)):
    rule.run(user)

# Disable nested tracing (top-level only)
with use_tracing(PrintHook(), TraceConfig(nested=False)):
    rule.run(user)
```

## Built-in Hooks

```python
from kompoz import PrintHook, LoggingHook

# PrintHook - prints to stdout
hook = PrintHook(indent="  ", show_ctx=False)

# LoggingHook - uses Python logging
import logging
logger = logging.getLogger("kompoz")
hook = LoggingHook(logger, level=logging.DEBUG)
```

## Custom Hooks

Implement the `TraceHook` protocol:

```python
class MyHook:
    def on_enter(self, name: str, ctx, depth: int):
        """Called before combinator runs. Return a span token."""
        print(f"Starting {name}")
        return time.time()

    def on_exit(self, span, name: str, ok: bool, duration_ms: float, depth: int):
        """Called after combinator completes."""
        print(f"Finished {name}: {'OK' if ok else 'FAIL'} in {duration_ms:.2f}ms")

    def on_error(self, span, name: str, error: Exception, duration_ms: float, depth: int):
        """Optional: called if combinator raises."""
        print(f"Error in {name}: {error}")
```

## OpenTelemetry Integration

```python
from opentelemetry import trace
from kompoz import use_tracing, OpenTelemetryHook

tracer = trace.get_tracer("my-service")

with use_tracing(OpenTelemetryHook(tracer)):
    rule.run(user)  # Creates spans for each combinator
```
