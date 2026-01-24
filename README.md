# Kompoz

<!--toc:start-->

- [Kompoz](#kompoz)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
    - [1. Define Rules](#1-define-rules)
    - [2. Compose Rules](#2-compose-rules)
    - [3. Run Rules](#3-run-rules)
  - [Operators](#operators)
  - [Transforms (Data Pipelines)](#transforms-data-pipelines)
  - [Expression DSL](#expression-dsl)
    - [Basic Syntax](#basic-syntax)
    - [Expression Operators](#expression-operators)
    - [Examples](#examples)
    - [Multi-line Expressions](#multi-line-expressions)
    - [Operator Precedence](#operator-precedence)
    - [Load from File](#load-from-file)
  - [Type Hints](#type-hints)
  - [Testing](#testing)
  - [Use Cases](#use-cases)
    - [Access Control](#access-control)
    - [Form Validation](#form-validation)
    - [Data Pipeline with Fallbacks](#data-pipeline-with-fallbacks)
    - [Feature Flags](#feature-flags)
  - [API Reference](#api-reference)
    - [Core Classes](#core-classes)
    - [Decorators](#decorators)
    - [Functions](#functions)
    - [Utility Combinators](#utility-combinators)
  - [Examples](#examples-1)
  - [Contributing](#contributing)
  - [License](#license)
      <!--toc:end-->

**Composable Predicate & Transform Combinators for Python**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

- **Operator Overloading**: Use `&` (and), `|` (or), `~` (not) for intuitive composition
- **Decorator Syntax**: Clean `@rule` and `@rule_args` decorators
- **Parameterized Rules**: `account_older_than(30)` creates reusable predicates
- **Expression DSL**: Human-readable rule expressions with AND/OR/NOT
- **Type Hints**: Full typing support with generics
- **Zero Dependencies**: Core library has no external dependencies

## Installation

```bash
pip install kompoz
```

## Quick Start

### 1. Define Rules

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

### 2. Compose Rules

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

### 3. Run Rules

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

## Operators

| Operator | Meaning    | Behavior                           |
| -------- | ---------- | ---------------------------------- |
| `a & b`  | AND / then | Run `b` only if `a` succeeds       |
| `a \| b` | OR / else  | Run `b` only if `a` fails          |
| `~a`     | NOT        | Invert success/failure             |
| `a >> b` | THEN       | Always run both, keep `b`'s result |

## Transforms (Data Pipelines)

```python
from kompoz import pipe, pipe_args, rule

@pipe
def parse_int(data):
    return int(data)

@pipe
def double(data):
    return data * 2

@pipe_args
def add(data, n):
    return data + n

@rule
def is_positive(data):
    return data > 0

# Build a pipeline
pipeline = parse_int & is_positive & double & add(10)

ok, result = pipeline.run("21")
# ok=True, result=52  (21 * 2 + 10)

ok, result = pipeline.run("-5")
# ok=False, result=-5  (stopped at is_positive)
```

## Expression DSL

Load rules from human-readable expressions instead of code:

### Basic Syntax

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

### Expression Operators

Both symbol and word syntax are supported:

| Symbol | Word  | Meaning                  |
| ------ | ----- | ------------------------ |
| `&`    | `AND` | All conditions must pass |
| `\|`   | `OR`  | Any condition must pass  |
| `~`    | `NOT` | Invert the condition     |
| `!`    | `NOT` | Invert (alias)           |
| `()`   |       | Grouping                 |

### Examples

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

# Complex expressions
loaded = reg.load("""
    is_admin
    | (is_active & ~is_banned & account_older_than(30))
""")
```

### Multi-line Expressions

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

### Operator Precedence

From lowest to highest:

1. `OR` / `|` (evaluated last)
2. `AND` / `&`
3. `NOT` / `~` / `!` (evaluated first)

```python
# This expression:
is_admin | is_active & ~is_banned

# Is parsed as:
is_admin | (is_active & (~is_banned))

# Use parentheses to override:
(is_admin | is_active) & ~is_banned
```

### Load from File

Save expressions in `.kpz` files (Kompoz expression format):

```
# access_control.kpz
# Comments are supported

is_admin | (is_active & ~is_banned & account_older_than(30))
```

```python
loaded = reg.load_file("access_control.kpz")
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

## Testing

Kompoz combinators are easy to test:

```python
import pytest
from kompoz import rule

@rule
def is_positive(x: int) -> bool:
    return x > 0

@rule
def is_even(x: int) -> bool:
    return x % 2 == 0

class TestRules:
    def test_simple_rule(self):
        ok, _ = is_positive.run(5)
        assert ok is True

    def test_combined_rule(self):
        combined = is_positive & is_even
        assert combined.run(4)[0] is True
        assert combined.run(3)[0] is False  # odd
        assert combined.run(-2)[0] is False  # negative

    @pytest.mark.parametrize("value,expected", [
        (4, True),
        (3, False),
        (-2, False),
        (0, False),
    ])
    def test_parametrized(self, value, expected):
        combined = is_positive & is_even
        assert combined.run(value)[0] is expected
```

## Use Cases

### Access Control

```python
can_edit = is_owner | (is_admin & ~is_suspended)
can_delete = is_owner | is_superadmin
can_view = is_public | can_edit
```

### Form Validation

```python
valid_email = matches_regex(r".+@.+\..+")
valid_password = min_length(8) & has_digit & has_uppercase
valid_form = valid_email & valid_password & accepted_terms
```

### Data Pipeline with Fallbacks

```python
fetch_data = (
    (try_primary_db | try_replica_db | try_cache)
    & validate_schema
    & transform_response
)
```

### Feature Flags

```python
show_feature = (
    is_beta_user
    | (is_premium & feature_enabled("new_dashboard"))
    | percentage_rollout(10)
)
```

## API Reference

### Core Classes

- **`Combinator[T]`**: Abstract base class for all combinators
- **`Predicate[T]`**: Checks a condition, doesn't modify context
- **`Transform[T]`**: Transforms context, fails on exception
- **`Try[T]`**: Wraps a function, converts exceptions to failure
- **`Registry[T]`**: Register and load rules from expressions

### Decorators

- **`@rule`**: Create a simple rule/predicate (single argument)
- **`@rule_args`**: Create a parameterized rule factory (multiple arguments)
- **`@pipe`**: Create a simple transform (single argument)
- **`@pipe_args`**: Create a parameterized transform factory (multiple arguments)

### Functions

- **`parse_expression(text)`**: Parse expression string into config dict

### Utility Combinators

- **`Always()`**: Always succeeds
- **`Never()`**: Always fails
- **`Debug(label)`**: Prints context and succeeds

## Examples

The `examples/` directory contains:

| File                    | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `rules_example.py`      | Using `@rule` and `@rule_args` decorators         |
| `transforms_example.py` | Using `@pipe` and `@pipe_args` for data pipelines |
| `registry_example.py`   | Loading rules from `.kpz` files                   |
| `access_control.kpz`    | Example access control expression                 |
| `trading.kpz`           | Example trading permission expression             |

Run examples:

```bash
cd kompoz
python examples/rules_example.py
python examples/transforms_example.py
python examples/registry_example.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.
