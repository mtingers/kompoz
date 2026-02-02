# Validation with Error Messages

Get descriptive error messages when rules fail.

## Basic Validation

```python
from kompoz import vrule, vrule_args, ValidationResult

@vrule(error="User {ctx.name} must be an admin")
def is_admin(user):
    return user.is_admin

@vrule(error=lambda u: f"{u.name} is BANNED!")
def not_banned(user):
    return not user.is_banned

@vrule_args(error="Account must be older than {days} days")
def account_older_than(user, days):
    return user.account_age_days > days

# Compose validating rules - collects ALL error messages
can_trade = is_admin & not_banned & account_older_than(30)

# Validate and get errors
result = can_trade.validate(user)
if not result.ok:
    print(result.errors)
    # ["User Bob must be an admin", "Account must be older than 30 days"]

# Raise exception if invalid
result.raise_if_invalid(ValueError)
```

## NOT Operator

Validating rules support the NOT operator:

```python
@vrule(error="User must not be an admin")
def is_admin(user):
    return user.is_admin

# ~is_admin returns a ValidatingCombinator that inverts the check
regular_users_only = ~is_admin & is_active

result = regular_users_only.validate(admin_user)
# result.ok = False, result.errors = ["NOT condition failed (inner passed)"]
```

## Async Validation

Async validation works identically to sync validation:

```python
from kompoz import async_vrule, async_vrule_args

@async_vrule(error="User must have permission")
async def has_permission(user):
    return await db.check_permission(user.id)

@async_vrule(error=lambda u: f"{u.name} is banned!")
async def not_banned(user):
    return not await db.is_banned(user.id)

@async_vrule_args(error="Credit score must be above {min_score}")
async def credit_above(user, min_score):
    score = await db.get_score(user.id)
    return score >= min_score

# Compose - collects ALL error messages
can_trade = has_permission & not_banned & credit_above(700)

# Validate
result = await can_trade.validate(user)
if not result.ok:
    print(result.errors)
```
