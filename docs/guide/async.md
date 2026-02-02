# Async Support

For rules that need to hit databases or APIs.

## Async Rules and Transforms

```python
from kompoz import async_rule, async_rule_args, async_pipe, AsyncRetry

@async_rule
async def has_permission(user):
    return await db.check_permission(user.id)

@async_rule_args
async def has_role(user, role):
    return await db.check_role(user.id, role)

@async_pipe
async def load_profile(user):
    user.profile = await api.get_profile(user.id)
    return user

# Compose async rules
can_admin = has_permission & has_role("admin")

# Run async
ok, result = await can_admin.run(user)

# Async retry with exponential backoff
resilient = AsyncRetry(fetch_data, max_attempts=3, backoff=1.0, exponential=True)
ok, result = await resilient.run(request)
```

## Error Tracking

Async transforms track errors just like sync transforms:

```python
@async_pipe
async def fetch_user_data(user_id):
    return await api.get_user(user_id)

ok, result = await fetch_user_data.run(invalid_id)
if not ok:
    print(f"API error: {fetch_user_data.last_error}")
```

## Parallel Execution

Use `parallel_and()` to run multiple async checks concurrently instead of sequentially:

```python
from kompoz import parallel_and, async_rule

@async_rule
async def check_permissions(user):
    return await db.check_permissions(user.id)

@async_rule
async def check_quota(user):
    return await api.check_quota(user.id)

@async_rule
async def check_billing(user):
    return await billing.is_active(user.id)

# Sequential: runs one after another (~300ms if each takes 100ms)
sequential = check_permissions & check_quota & check_billing

# Parallel: runs all concurrently (~100ms total)
parallel = parallel_and(check_permissions, check_quota, check_billing)

ok, result = await parallel.run(user)
```

Key differences from `&`:

- All children receive the **same original context** (not chained)
- All checks run **concurrently** via `asyncio.gather()`
- Returns `(all_ok, original_ctx)` -- context is never modified
- With `AsyncValidatingCombinator`, collects **all errors** concurrently

### Parallel Validation

```python
from kompoz import parallel_and, async_vrule

@async_vrule(error="No permission")
async def check_permissions(user):
    return await db.check_permissions(user.id)

@async_vrule(error="Quota exceeded")
async def check_quota(user):
    return await api.check_quota(user.id)

# Validates all concurrently, collects all errors
checks = parallel_and(check_permissions, check_quota)
result = await checks.validate(user)
# result.errors might be ["No permission", "Quota exceeded"]
```
