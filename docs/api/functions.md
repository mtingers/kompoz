# Functions

## explain

```python
explain(combinator: Combinator) -> str
```

Generate a plain English explanation of what a rule does.

```python
from kompoz import explain

rule = is_admin | (is_active & ~is_banned)
print(explain(rule))
# Check passes if ANY of:
#   * Check: is_admin
#   * ALL of:
#     * Check: is_active
#     * NOT: is_banned
```

---

## if_then_else

```python
if_then_else(
    condition: Combinator[T],
    then_branch: Combinator[T],
    else_branch: Combinator[T],
) -> Combinator[T]
```

Create a conditional combinator. If `condition` succeeds, run `then_branch`; otherwise run `else_branch`.

```python
from kompoz import if_then_else

pricing = if_then_else(is_premium, apply_discount, charge_full_price)
ok, user = pricing.run(user)
```

---

## async_if_then_else

```python
async_if_then_else(
    condition: AsyncCombinator[T],
    then_branch: AsyncCombinator[T],
    else_branch: AsyncCombinator[T],
) -> AsyncCombinator[T]
```

Async version of `if_then_else`.

---

## parse_expression

```python
parse_expression(text: str) -> dict
```

Parse an expression string into a config dictionary. Used internally by `Registry.load()`.

```python
from kompoz import parse_expression

config = parse_expression("is_admin & is_active")
```

---

## use_tracing

```python
use_tracing(hook: TraceHook, config: TraceConfig | None = None) -> ContextManager
```

Context manager that enables tracing for all `run()` calls within the scope.

```python
from kompoz import use_tracing, PrintHook

with use_tracing(PrintHook()):
    rule.run(user)
```

---

## run_traced

```python
run_traced(
    combinator: Combinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig | None = None,
) -> tuple[bool, T]
```

Run a combinator with explicit tracing.

```python
from kompoz import run_traced, PrintHook

ok, result = run_traced(rule, user, PrintHook())
```

---

## run_async_traced

```python
run_async_traced(
    combinator: AsyncCombinator[T],
    ctx: T,
    hook: TraceHook,
    config: TraceConfig | None = None,
) -> tuple[bool, T]
```

Run an async combinator with explicit tracing.

```python
ok, result = await run_async_traced(async_rule, user, PrintHook())
```

---

## use_cache

```python
use_cache() -> ContextManager
```

Context manager that enables caching for `@cached_rule` and `@async_cached_rule` predicates within the scope.

```python
from kompoz import use_cache

with use_cache():
    rule.run(user)  # Executes
    rule.run(user)  # Uses cache
```

---

## parallel_and

```python
parallel_and(*combinators: AsyncCombinator[T]) -> AsyncCombinator[T]
```

Create an async AND combinator that runs all children concurrently via `asyncio.gather()`. All children receive the same original context (not chained). Returns `(all_ok, original_ctx)`.

```python
from kompoz import parallel_and

parallel = parallel_and(check_a, check_b, check_c)
ok, result = await parallel.run(ctx)
```
