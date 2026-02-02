# Functional Programming Patterns

Kompoz combinators map naturally to common functional programming idioms. This guide
demonstrates eight patterns using an order-processing pipeline.

## Domain Model

All examples use a frozen (immutable) dataclass:

```python
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Order:
    items: list[str]
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    discount: float = 0.0
    status: str = "pending"
```

## 1. Pipeline Composition

Chain transforms with `&` to build left-to-right data pipelines:

```python
from kompoz import pipe, pipe_args

@pipe
def calculate_subtotal(order: Order) -> Order:
    return replace(order, subtotal=len(order.items) * 9.99)

@pipe_args
def apply_tax(order: Order, rate: float) -> Order:
    return replace(order, tax=round(order.subtotal * rate, 2))

@pipe
def compute_total(order: Order) -> Order:
    return replace(order, total=round(order.subtotal + order.tax - order.discount, 2))

# Each step feeds the next
checkout = calculate_subtotal & apply_tax(0.08) & compute_total
```

## 2. Railway-Oriented Programming

Failures short-circuit the pipeline automatically:

```python
from kompoz import rule

@rule
def has_items(order: Order) -> bool:
    return len(order.items) > 0

@pipe
def validate_inventory(order: Order) -> Order:
    if "out_of_stock" in order.items:
        raise ValueError("Item unavailable")
    return order

# If any step fails the rest is skipped
safe_checkout = has_items & validate_inventory & checkout
```

```
['a', 'b']                    -> total=$21.58
[]                             -> FAILED (short-circuited)
['a', 'out_of_stock']         -> FAILED (short-circuited)
```

## 3. Fallback Chains

Use `|` to try alternatives -- the first success wins:

```python
@pipe
def from_cache(order: Order) -> Order:
    raise KeyError("cache miss")

@pipe
def from_primary(order: Order) -> Order:
    raise ConnectionError("primary down")

@pipe
def from_fallback(order: Order) -> Order:
    return replace(order, tax=5.0)

# First success wins, rest are skipped
resolve_tax = from_cache | from_primary | from_fallback
```

## 4. Conditional Branching

Use `if_then_else` for explicit branching -- exactly one branch runs:

```python
from kompoz import if_then_else

@rule
def is_premium(order: Order) -> bool:
    return len(order.items) >= 5

@pipe
def apply_premium_discount(order: Order) -> Order:
    return replace(order, discount=round(order.subtotal * 0.20, 2))

@pipe
def no_discount(order: Order) -> Order:
    return replace(order, discount=0.0)

apply_pricing = if_then_else(is_premium, apply_premium_discount, no_discount)
```

## 5. Higher-Order Combinators

Use parameterized factories (`@rule_args`, `@pipe_args`) for partially applied combinators:

```python
from kompoz import rule_args, pipe_args

@rule_args
def min_order_value(order: Order, threshold: float) -> bool:
    return order.subtotal >= threshold

@pipe_args
def add_flat_fee(order: Order, fee: float) -> Order:
    return replace(order, total=order.total + fee)

# Partially apply to create specialized combinators
qualifies_for_free_shipping = min_order_value(50.0)
add_shipping = add_flat_fee(5.99)
```

## 6. Pure Error Handling

Use `run_with_error()` instead of checking `last_error` for thread-safe error access:

```python
@pipe
def parse_quantity(raw: str) -> str:
    n = int(raw)
    if n <= 0:
        raise ValueError("quantity must be positive")
    return raw

for raw in ["5", "abc", "-1"]:
    ok, result, error = parse_quantity.run_with_error(raw)
    if ok:
        print(f"  {raw!r} -> ok")
    else:
        print(f"  {raw!r} -> error: {error}")
```

```
  '5'   -> ok
  'abc' -> error: invalid literal for int() with base 10: 'abc'
  '-1'  -> error: quantity must be positive
```

## 7. Retry as a Combinator

Wrap flaky operations with `Retry`:

```python
from kompoz import Retry

@pipe
def flaky_service(order: Order) -> Order:
    # Simulates a service that fails intermittently
    ...

resilient_confirm = Retry(flaky_service, max_attempts=5, backoff=0.0)
info = resilient_confirm.run_with_info(order)
print(f"ok={info.ok}, attempts={info.attempts_made}")
```

## 8. Full Pipeline

Combine all patterns into a complete pipeline:

```python
full_pipeline = (
    has_items              # gate: must have items
    & validate_inventory   # gate: all items in stock
    & calculate_subtotal   # transform: compute subtotal
    & apply_pricing        # branch: premium vs standard discount
    & apply_tax(0.08)      # transform: add tax
    & compute_total        # transform: final total
)
```

!!! tip
    See the full runnable example in [`examples/functional_example.py`](https://github.com/mtingers/kompoz/blob/main/examples/functional_example.py).
