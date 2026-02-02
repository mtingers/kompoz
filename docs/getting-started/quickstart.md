# Quick Start

## 1. Define Rules

```python
from kompoz import rule, rule_args

# Simple rules (single argument)
@rule
def is_admin(user):
    return user.is_admin

@rule
def is_banned(user):
    return user.is_banned

# Parameterized rules (extra arguments)
@rule_args
def credit_above(user, threshold):
    return user.credit_score > threshold
```

## 2. Compose Rules

```python
# Simple AND
must_be_active_admin = is_admin & is_active

# OR with fallback
can_access = is_admin | (is_active & ~is_banned)

# Complex nested logic
api_access = is_admin | (
    is_active
    & ~is_banned
    & account_older_than(30)
    & (credit_above(650) | has_override)
)
```

## 3. Run Rules

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    is_admin: bool = False
    is_active: bool = True
    is_banned: bool = False
    account_age_days: int = 0
    credit_score: int = 500

user = User("Alice", account_age_days=60, credit_score=700)
ok, _ = api_access.run(user)
print(f"Access: {'granted' if ok else 'denied'}")
```

## Type Hints

Kompoz is fully typed. For best results with type checkers like Pyright/mypy, use the correct decorators:

```python
from kompoz import rule, rule_args, Predicate, Registry

# Simple rule (single argument) - use @rule
@rule
def is_admin(user: User) -> bool:
    return user.is_admin

# Parameterized rule (extra arguments) - use @rule_args
@rule_args
def older_than(user: User, days: int) -> bool:
    return user.account_age_days > days

# For inline Predicates, add explicit type annotation
is_positive: Predicate[int] = Predicate(lambda x: x > 0, "is_positive")

# Registry should be typed
reg: Registry[User] = Registry()
```

The `@rule` decorator returns `Predicate[T]`, while `@rule_args` returns a factory that produces `Predicate[T]`. This separation ensures Pyright can properly infer types.

## Pydantic Compatibility

Kompoz works seamlessly with Pydantic models:

```python
from pydantic import BaseModel, EmailStr
from kompoz import rule, rule_args, vrule_args, Registry

class User(BaseModel):
    name: str
    email: EmailStr
    is_admin: bool = False
    is_active: bool = True
    account_age_days: int = 0
    credit_score: int = 500

# Rules work with Pydantic models just like dataclasses
@rule
def is_admin(user: User) -> bool:
    return user.is_admin

@rule
def is_active(user: User) -> bool:
    return user.is_active

@rule_args
def credit_above(user: User, threshold: int) -> bool:
    return user.credit_score > threshold

# Compose rules
can_trade = is_active & credit_above(600)

# Use with Pydantic model
user = User(name="Alice", email="alice@example.com", credit_score=750)
ok, _ = can_trade.run(user)  # True

# Validation rules with Pydantic
@vrule_args(error="User {ctx.name} must have credit score above {score}")
def credit_at_least(user: User, score: int) -> bool:
    return user.credit_score >= score

# Registry with Pydantic models
reg = Registry[User]()

@reg.predicate
def is_verified(user: User) -> bool:
    return user.is_active and user.account_age_days > 30

# Load from DSL
rule = reg.load("is_admin | (is_active & is_verified)")
```

Since Pydantic models behave like regular Python objects with attribute access, all Kompoz features work out of the box -- including validation, async rules, caching, and the expression DSL.
