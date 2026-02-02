# Expression DSL

Load rules from human-readable expressions instead of code.

## Registry & Basic Syntax

```python
from kompoz import Registry
from dataclasses import dataclass

@dataclass
class User:
    is_admin: bool = False
    is_active: bool = True
    is_banned: bool = False
    account_age_days: int = 0

# Create registry and register predicates
reg = Registry[User]()

@reg.predicate
def is_admin(u):
    return u.is_admin

@reg.predicate
def is_active(u):
    return u.is_active

@reg.predicate
def is_banned(u):
    return u.is_banned

@reg.predicate
def account_older_than(u, days):
    return u.account_age_days > days

# Load rules from expressions
loaded = reg.load("is_admin & is_active")
loaded = reg.load("is_admin AND is_active")  # same thing
```

## Expression Operators

Both symbol and word syntax are supported:

| Symbol          | Word             | Meaning                             |
| --------------- | ---------------- | ----------------------------------- |
| `&`             | `AND`            | All conditions must pass            |
| `\|`            | `OR`             | Any condition must pass             |
| `~`, `!`        | `NOT`            | Invert the condition                |
| `>>`            | `THEN`           | Always run both, keep second result |
| `a ? b : c`     | `IF a THEN b ELSE c` | Conditional branching          |
| `()`            |                  | Grouping                            |

## Modifiers

Postfix modifiers add retry and caching behavior:

| Modifier                | Meaning                                 |
| ----------------------- | --------------------------------------- |
| `:retry(n)`             | Retry up to n times on failure          |
| `:retry(n, backoff)`    | Retry with backoff delay (seconds)      |
| `:retry(n, b, true)`    | Exponential backoff                     |
| `:retry(n, b, true, j)` | With jitter                             |
| `:cached`               | Cache result within `use_cache()` scope |

Modifiers can be chained: `rule:cached:retry(3)`

## Examples

```python
# Simple rules
loaded = reg.load("is_admin")
loaded = reg.load("is_active")

# AND - all must pass
loaded = reg.load("is_admin & is_active")
loaded = reg.load("is_admin AND is_active")

# OR - any must pass
loaded = reg.load("is_admin | is_premium")
loaded = reg.load("is_admin OR is_premium")

# NOT - invert result
loaded = reg.load("~is_banned")
loaded = reg.load("NOT is_banned")
loaded = reg.load("!is_banned")

# Parameterized rules
loaded = reg.load("account_older_than(30)")
loaded = reg.load("credit_above(700)")

# Grouping with parentheses
loaded = reg.load("is_admin | (is_active & ~is_banned)")

# Conditional branching - IF/THEN/ELSE
loaded = reg.load("IF is_premium THEN apply_discount ELSE charge_full")

# Ternary syntax (equivalent to IF/THEN/ELSE)
loaded = reg.load("is_premium ? apply_discount : charge_full")

# Sequence - always run both, keep second result
loaded = reg.load("validate >> transform >> format_output")

# Modifiers - retry on failure
loaded = reg.load("fetch_user:retry(3)")              # Retry up to 3 times
loaded = reg.load("fetch_user:retry(3, 1.0)")         # With 1s backoff
loaded = reg.load("fetch_user:retry(3, 1.0, true)")   # Exponential backoff

# Modifiers - caching
loaded = reg.load("expensive_check:cached")           # Cache results

# Modifiers on grouped expressions
loaded = reg.load("(fetch_primary | fetch_fallback):retry(5)")

# Chain modifiers
loaded = reg.load("slow_query:cached:retry(3)")

# Complex expressions
loaded = reg.load("""
    is_admin
    | (is_active & ~is_banned & account_older_than(30))
""")

# Complex with modifiers
loaded = reg.load("""
    is_admin
    | (is_active & ~is_banned & fetch_permissions:retry(3, 1.0))
""")
```

## Multi-line Expressions

Newlines are ignored, so you can format for readability:

```python
loaded = reg.load("""
    is_admin
    & is_active
    & ~is_banned
    & account_older_than(30)
""")

# Comments are supported
loaded = reg.load("""
    is_admin           # must be admin
    & ~is_banned       # and not banned
    & account_older_than(30)  # with mature account
""")
```

## Load from File

Save expressions in `.kpz` files (Kompoz expression format):

```title="access_control.kpz"
# access_control.kpz
# Comments are supported

is_admin | (is_active & ~is_banned & account_older_than(30))
```

```python
loaded = reg.load_file("access_control.kpz")
```

With modifiers:

```title="resilient_access.kpz"
# resilient_access.kpz
# Retry flaky permission checks

is_admin
| (is_active
   & ~is_banned
   & fetch_permissions:retry(3, 1.0)
   & account_older_than(30))
```
