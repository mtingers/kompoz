# Async Classes

## AsyncCombinator

```python
class AsyncCombinator(Generic[T], ABC)
```

Base class for async combinators. Integrates with `use_tracing()`.

**Methods:**

- `async run(ctx: T) -> tuple[bool, T]` -- Execute the combinator
- `if_else(then, else_) -> AsyncCombinator[T]` -- Conditional branching
- `__and__`, `__or__`, `__invert__`, `__rshift__` -- Operator overloading

---

## AsyncPredicate

```python
class AsyncPredicate(AsyncCombinator[T])
```

Async predicate that checks a condition without modifying context.

**Constructor:**

```python
AsyncPredicate(fn: Callable[[T], Awaitable[bool]], name: str = "")
```

**Example:**

```python
@async_rule
async def has_permission(user):
    return await db.check_permission(user.id)
```

---

## AsyncPredicateFactory

```python
class AsyncPredicateFactory(Generic[T])
```

Factory for parameterized async predicates. Created by `@async_rule_args`.

**Example:**

```python
@async_rule_args
async def has_role(user, role):
    return await db.check_role(user.id, role)

check = has_role("admin")  # AsyncPredicate[User]
```

---

## AsyncTransform

```python
class AsyncTransform(AsyncCombinator[T])
```

Async transform that modifies context.

**Attributes:**

- `last_error: Exception | None` -- The last exception (not concurrency-safe)

**Methods:**

- `async run(ctx: T) -> tuple[bool, T]` -- Execute the transform
- `async run_with_error(ctx: T) -> tuple[bool, T, Exception | None]` -- Concurrency-safe error access

**Example:**

```python
@async_pipe
async def load_profile(user):
    user.profile = await api.get_profile(user.id)
    return user

ok, result, error = await load_profile.run_with_error(user)
```

---

## AsyncTransformFactory

```python
class AsyncTransformFactory(Generic[T])
```

Factory for parameterized async transforms. Created by `@async_pipe_args`.

**Example:**

```python
@async_pipe_args
async def add_metadata(ctx, key, value):
    ctx.metadata[key] = value
    return ctx

transform = add_metadata("source", "api")  # AsyncTransform
```
