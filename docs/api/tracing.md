# Tracing Classes

## TraceHook

```python
class TraceHook(Protocol)
```

Protocol for custom trace hooks. Implement this to create your own tracing backend.

**Methods:**

- `on_enter(name: str, ctx: Any, depth: int) -> Any` -- Called before a combinator runs. Return a span token.
- `on_exit(span: Any, name: str, ok: bool, duration_ms: float, depth: int) -> None` -- Called after a combinator completes.
- `on_error(span: Any, name: str, error: Exception, duration_ms: float, depth: int) -> None` -- Called if a combinator raises (optional).

**Example:**

```python
class MyHook:
    def on_enter(self, name, ctx, depth):
        return time.time()

    def on_exit(self, span, name, ok, duration_ms, depth):
        print(f"{name}: {'OK' if ok else 'FAIL'} ({duration_ms:.2f}ms)")

    def on_error(self, span, name, error, duration_ms, depth):
        print(f"{name}: ERROR {error}")
```

---

## TraceConfig

```python
@dataclass
class TraceConfig:
    include_leaf_only: bool = False
    max_depth: int | None = None
    nested: bool = True
```

Configuration for tracing behavior.

**Fields:**

- `include_leaf_only` -- Only trace leaf predicates, skip AND/OR/NOT containers
- `max_depth` -- Maximum nesting depth to trace
- `nested` -- Whether to trace nested combinators (set `False` for top-level only)

---

## PrintHook

```python
class PrintHook
```

Built-in hook that prints trace output to stdout.

**Constructor:**

```python
PrintHook(indent: str = "  ", show_ctx: bool = False)
```

---

## LoggingHook

```python
class LoggingHook
```

Built-in hook that uses Python's `logging` module.

**Constructor:**

```python
LoggingHook(logger: logging.Logger, level: int = logging.DEBUG)
```

---

## OpenTelemetryHook

```python
class OpenTelemetryHook
```

Hook that creates OpenTelemetry spans for each combinator.

Requires the `opentelemetry` extra: `pip install kompoz[opentelemetry]`

**Constructor:**

```python
OpenTelemetryHook(tracer: opentelemetry.trace.Tracer)
```

**Example:**

```python
from opentelemetry import trace
from kompoz import OpenTelemetryHook, use_tracing

tracer = trace.get_tracer("my-service")
with use_tracing(OpenTelemetryHook(tracer)):
    rule.run(user)
```
