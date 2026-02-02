# Kompoz

**Composable Predicate & Transform Combinators for Python**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Kompoz lets you build complex validation rules and data pipelines using intuitive Python operators. Instead of nested `if/else` statements, write declarative, composable logic:

```python
from dataclasses import dataclass
from kompoz import rule, rule_args


@dataclass
class User:
    name: str
    is_admin: bool = False
    is_active: bool = True
    is_banned: bool = False
    account_age_days: int = 0
    credit_score: int = 500


@rule
def is_admin(user):
    return user.is_admin

@rule
def is_active(user):
    return user.is_active

@rule_args
def account_older_than(user, days):
    return user.account_age_days > days

# Combine with operators - reads like English!
can_access = is_admin | (is_active & account_older_than(30))

# Use it
ok, _ = can_access.run(user)
```

## Features

<div class="grid cards" markdown>

- **Operator Overloading** --- Use `&` (and), `|` (or), `~` (not), `>>` (then) for intuitive composition
- **Conditional Branching** --- `if_then_else()` and ternary `?:` for explicit control flow
- **Decorator Syntax** --- Clean `@rule` and `@rule_args` decorators
- **Parameterized Rules** --- `account_older_than(30)` creates reusable predicates
- **Validation with Errors** --- `@vrule` / `@async_vrule` decorators collect all error messages
- **Expression DSL** --- Human-readable rule expressions with AND/OR/NOT/IF/THEN/ELSE
- **Async Support** --- Full async/await support with tracing, validation, and parallel execution
- **Caching** --- `@cached_rule` and `use_cache()` to memoize expensive predicates
- **Time-Based Rules** --- `during_hours()`, `on_weekdays()`, `after_date()`, and more
- **Error Tracking** --- Transforms track exceptions via `last_error` attribute
- **Retry with Observability** --- Built-in retry logic with hooks for monitoring
- **Zero Dependencies** --- Core library has no external dependencies

</div>

## Quick Install

=== "pip"

    ```bash
    pip install kompoz
    ```

=== "uv"

    ```bash
    uv add kompoz
    ```

## Next Steps

- [Installation](getting-started/installation.md) -- Python version requirements and extras
- [Quick Start](getting-started/quickstart.md) -- Define, compose, and run rules in 5 minutes
- [API Reference](api/index.md) -- Full reference of all public symbols
