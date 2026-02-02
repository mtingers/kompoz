# Caching / Memoization

Avoid re-running expensive predicates.

## Basic Caching

```python
from kompoz import cached_rule, use_cache

@cached_rule
def expensive_check(user):
    return slow_database_query(user.id)

@cached_rule(key=lambda u: u.id)
def check_by_id(user):
    return api_call(user.id)

# Results cached within this scope
with use_cache():
    rule.run(user)  # Executes
    rule.run(user)  # Uses cache
    rule.run(user)  # Uses cache
```

## Async Caching

Async caching works the same way:

```python
from kompoz import async_cached_rule, use_cache

@async_cached_rule
async def fetch_permissions(user):
    return await db.get_permissions(user.id)

@async_cached_rule(key=lambda u: u.id)
async def fetch_by_id(user):
    return await api.fetch(user.id)

# Cache works with async rules too
with use_cache():
    await rule.run(user)  # Executes
    await rule.run(user)  # Uses cache
```

## DSL Integration

Use the `:cached` modifier in DSL expressions:

```python
loaded = reg.load("expensive_check:cached")
loaded = reg.load("slow_query:cached:retry(3)")
```
