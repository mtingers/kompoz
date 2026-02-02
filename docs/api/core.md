# Core Classes

## Combinator

```python
class Combinator(Generic[T], ABC)
```

Abstract base class for all combinators.

**Methods:**

- `run(ctx: T) -> tuple[bool, T]` -- Execute the combinator
- `if_else(then: Combinator[T], else_: Combinator[T]) -> Combinator[T]` -- Conditional branching
- `__and__(other)` -- `a & b` (AND)
- `__or__(other)` -- `a | b` (OR)
- `__invert__()` -- `~a` (NOT)
- `__rshift__(other)` -- `a >> b` (THEN)

---

## Predicate

```python
class Predicate(Combinator[T])
```

Checks a condition without modifying the context. Returns `(True, ctx)` on success, `(False, ctx)` on failure.

Supports `__eq__` and `__hash__` -- usable in sets and as dictionary keys.

**Constructor:**

```python
Predicate(fn: Callable[[T], bool], name: str = "")
```

**Example:**

```python
from kompoz import Predicate

is_positive: Predicate[int] = Predicate(lambda x: x > 0, "is_positive")
ok, result = is_positive.run(5)  # (True, 5)
```

---

## PredicateFactory

```python
class PredicateFactory(Generic[T])
```

Factory for parameterized predicates. Created by `@rule_args`.

**Example:**

```python
@rule_args
def credit_above(user, threshold):
    return user.credit_score > threshold

# Returns a Predicate[User]
check = credit_above(700)
```

---

## Transform

```python
class Transform(Combinator[T])
```

Transforms the context. Returns `(True, new_ctx)` on success, `(False, original_ctx)` on exception.

Supports `__eq__` and `__hash__`.

**Attributes:**

- `last_error: Exception | None` -- The last exception (not thread-safe)

**Methods:**

- `run(ctx: T) -> tuple[bool, T]` -- Execute the transform
- `run_with_error(ctx: T) -> tuple[bool, T, Exception | None]` -- Thread-safe error access

**Example:**

```python
@pipe
def double(x: int) -> int:
    return x * 2

ok, result = double.run(5)  # (True, 10)

# Thread-safe error handling
ok, result, error = double.run_with_error("bad")
```

---

## TransformFactory

```python
class TransformFactory(Generic[T])
```

Factory for parameterized transforms. Created by `@pipe_args`.

**Example:**

```python
@pipe_args
def add(data, n):
    return data + n

add_ten = add(10)  # Returns a Transform
```

---

## Try

```python
class Try(Combinator[T])
```

Wraps a function, converting exceptions to `(False, ctx)`.

**Constructor:**

```python
Try(fn: Callable[[T], T], name: str = "")
```

---

## Registry

```python
class Registry(Generic[T])
```

Register predicates and load rules from expression strings or `.kpz` files.

**Methods:**

- `predicate` -- Decorator to register a predicate
- `load(expression: str) -> Combinator[T]` -- Parse and build a combinator from an expression
- `load_file(path: str) -> Combinator[T]` -- Load expression from a `.kpz` file

**Example:**

```python
reg = Registry[User]()

@reg.predicate
def is_admin(u):
    return u.is_admin

loaded = reg.load("is_admin & is_active")
loaded = reg.load_file("access_control.kpz")
```

---

## ExpressionParser

```python
class ExpressionParser
```

Parser for human-readable rule expressions. Used internally by `Registry.load()`.

**Methods:**

- `parse(text: str) -> dict` -- Parse expression into a config dict

See also: [`parse_expression()`](functions.md#parse_expression)
