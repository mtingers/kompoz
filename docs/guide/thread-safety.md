# Thread Safety

Kompoz combinators are lightweight and mostly stateless, but a few attributes are
mutated during execution. If you share a combinator instance across threads or async
tasks, use the pure alternatives listed below.

## Mutable Attributes

| Mutable attribute | On class | Pure alternative |
| --- | --- | --- |
| `last_error` | `Transform` / `AsyncTransform` | `run_with_error()` -> `(ok, ctx, error)` |
| `last_error` | `Retry` / `AsyncRetry` | `run_with_info()` -> `RetryResult` |
| `attempts_made` | `Retry` / `AsyncRetry` | `run_with_info()` -> `RetryResult` |

## Context Mutation in OR Chains

When transforms are combined with `|`, the left-hand side may modify the context
*before* the right-hand side sees it. If your context is a mutable object (e.g. a
dataclass), the fallback branch receives the already-mutated value. To avoid surprises,
return new objects from each transform rather than mutating in place:

```python
from dataclasses import replace

@pipe
def enrich(user: User) -> User:
    # Safe: returns a new object instead of mutating
    return replace(user, enriched=True)
```
