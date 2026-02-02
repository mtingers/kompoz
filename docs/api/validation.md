# Validation Classes

## ValidationResult

```python
@dataclass
class ValidationResult(Generic[T]):
    ok: bool
    errors: list[str]
    ctx: T
```

Result from `.validate()` on a validating combinator.

**Methods:**

- `raise_if_invalid(exc_type: type[Exception]) -> None` -- Raise an exception if validation failed

**Example:**

```python
result = can_trade.validate(user)
if not result.ok:
    print(result.errors)
    # ["User Bob must be an admin", "Account must be older than 30 days"]

result.raise_if_invalid(ValueError)
```

---

## ValidatingCombinator

```python
class ValidatingCombinator(Generic[T])
```

Base class for validating combinators. Supports `&`, `|`, and `~` operators.

**Methods:**

- `validate(ctx: T) -> ValidationResult[T]` -- Run validation and collect all errors

---

## ValidatingPredicate

```python
class ValidatingPredicate(ValidatingCombinator[T])
```

Predicate with an error message. Created by `@vrule` and `@vrule_args`.

The error can be a format string (using `{ctx}` and parameter names) or a callable `(ctx) -> str`.

**Example:**

```python
@vrule(error="User {ctx.name} must be an admin")
def is_admin(user):
    return user.is_admin
```

---

## AsyncValidatingCombinator

```python
class AsyncValidatingCombinator(Generic[T])
```

Async base class for validating combinators. Supports `&`, `|`, and `~` operators.

**Methods:**

- `async validate(ctx: T) -> ValidationResult[T]` -- Run async validation

---

## AsyncValidatingPredicate

```python
class AsyncValidatingPredicate(AsyncValidatingCombinator[T])
```

Async predicate with an error message. Created by `@async_vrule` and `@async_vrule_args`.

**Example:**

```python
@async_vrule(error="User must have permission")
async def has_permission(user):
    return await db.check_permission(user.id)
```
