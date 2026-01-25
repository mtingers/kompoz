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
  - [Tracing & Debugging](#tracing-debugging)
    - [Explain Rules](#explain-rules)
    - [Tracing Execution](#tracing-execution)
    - [Trace Configuration](#trace-configuration)
    - [Built-in Hooks](#built-in-hooks)
    - [Custom Hooks](#custom-hooks)
    - [OpenTelemetry Integration](#opentelemetry-integration)
  - [Validation with Error Messages](#validation-with-error-messages)
  - [Async Support](#async-support)
  - [Caching / Memoization](#caching-memoization)
  - [Retry Logic](#retry-logic)
  - [Time-Based Rules](#time-based-rules)
  - [API Reference](#api-reference)
    - [Core Classes](#core-classes)
    - [Decorators](#decorators)
    - [Functions](#functions)
    - [Tracing Classes](#tracing-classes)
    - [Validation Classes](#validation-classes)
    - [Async Classes](#async-classes)
    - [Retry & Caching](#retry-caching)
    - [Temporal Combinators](#temporal-combinators)
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

## Tracing & Debugging

### Explain Rules

Generate plain English explanations of what a rule does:

```python
from kompoz import explain

rule = is_admin | (is_active & ~is_banned & account_older_than(30))
print(explain(rule))

# Output:
# Check passes if ANY of:
#   • Check: is_admin
#   • ALL of:
#     • Check: is_active
#     • NOT: is_banned
#     • Check: account_older_than(30)
```

### Tracing Execution

Trace rule execution with built-in hooks or custom implementations:

```python
from kompoz import use_tracing, run_traced, PrintHook, TraceConfig

# Option 1: Context manager (traces all run() calls in scope)
with use_tracing(PrintHook()):
    rule.run(user)

# Option 2: Explicit tracing
run_traced(rule, user, PrintHook())
```

Output:

```
-> OR
  -> Predicate(is_admin)
  <- Predicate(is_admin) ✗ (0.02ms)
  -> AND
    -> Predicate(is_active)
    <- Predicate(is_active) ✓ (0.01ms)
  <- AND ✓ (0.15ms)
<- OR ✓ (0.20ms)
```

### Trace Configuration

```python
from kompoz import TraceConfig

# Trace only leaf predicates (skip AND/OR/NOT)
with use_tracing(PrintHook(), TraceConfig(include_leaf_only=True)):
    rule.run(user)

# Limit trace depth
with use_tracing(PrintHook(), TraceConfig(max_depth=2)):
    rule.run(user)

# Disable nested tracing (top-level only)
with use_tracing(PrintHook(), TraceConfig(nested=False)):
    rule.run(user)
```

### Built-in Hooks

```python
from kompoz import PrintHook, LoggingHook

# PrintHook - prints to stdout
hook = PrintHook(indent="  ", show_ctx=False)

# LoggingHook - uses Python logging
import logging
logger = logging.getLogger("kompoz")
hook = LoggingHook(logger, level=logging.DEBUG)
```

### Custom Hooks

Implement the `TraceHook` protocol:

```python
class MyHook:
    def on_enter(self, name: str, ctx, depth: int):
        """Called before combinator runs. Return a span token."""
        print(f"Starting {name}")
        return time.time()

    def on_exit(self, span, name: str, ok: bool, duration_ms: float, depth: int):
        """Called after combinator completes."""
        print(f"Finished {name}: {'OK' if ok else 'FAIL'} in {duration_ms:.2f}ms")

    def on_error(self, span, name: str, error: Exception, duration_ms: float, depth: int):
        """Optional: called if combinator raises."""
        print(f"Error in {name}: {error}")
```

### OpenTelemetry Integration

```python
from opentelemetry import trace
from kompoz import use_tracing, OpenTelemetryHook

tracer = trace.get_tracer("my-service")

with use_tracing(OpenTelemetryHook(tracer)):
    rule.run(user)  # Creates spans for each combinator
```

## Validation with Error Messages

Get descriptive error messages when rules fail:

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

## Async Support

For rules that need to hit databases or APIs:

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

## Caching / Memoization

Avoid re-running expensive predicates:

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

## Retry Logic

Retry failed operations with configurable backoff:

```python
from kompoz import Retry

# Simple retry
fetch = Retry(fetch_from_api, max_attempts=3)

# Exponential backoff
fetch = Retry(
    fetch_from_api,
    max_attempts=5,
    backoff=1.0,       # Initial delay in seconds
    exponential=True,  # Double delay each attempt
    jitter=0.1         # Random jitter to avoid thundering herd
)

ok, result = fetch.run(request)
```

## Time-Based Rules

Create rules that depend on time, date, or day of week:

```python
from kompoz import during_hours, on_weekdays, on_days, after_date, before_date, between_dates
from datetime import date

# Time of day
business_hours = during_hours(9, 17)      # 9 AM to 5 PM
night_mode = during_hours(22, 6)          # 10 PM to 6 AM (overnight)

# Day of week
weekdays = on_weekdays()                  # Monday-Friday
mwf = on_days(0, 2, 4)                    # Mon, Wed, Fri
weekends = on_days(5, 6)                  # Sat, Sun

# Date ranges
launched = after_date(2024, 6, 1)
promo_active = before_date(2024, 12, 31)
q1_only = between_dates(date(2024, 1, 1), date(2024, 3, 31))

# Compose with other rules
can_trade = is_active & during_hours(9, 16) & on_weekdays()

# Premium users get extended hours
can_trade_premium = is_premium & during_hours(7, 20) & on_weekdays()
```

## API Reference

### Core Classes

- **`Combinator[T]`**: Abstract base class for all combinators
- **`Predicate[T]`**: Checks a condition, doesn't modify context
- **`Transform[T]`**: Transforms context, fails on exception
- **`Try[T]`**: Wraps a function, converts exceptions to failure
- **`Registry[T]`**: Register and load rules from expressions

### Decorators

- **`@rule`**: Create a simple rule/predicate
- **`@rule_args`**: Create a parameterized rule factory
- **`@pipe`**: Create a simple transform
- **`@pipe_args`**: Create a parameterized transform factory
- **`@vrule`**: Create a validating rule with error message
- **`@vrule_args`**: Create a parameterized validating rule
- **`@async_rule`**: Create an async predicate
- **`@async_rule_args`**: Create a parameterized async predicate
- **`@async_pipe`**: Create an async transform
- **`@async_pipe_args`**: Create a parameterized async transform
- **`@cached_rule`**: Create a rule with result caching

### Functions

- **`parse_expression(text)`**: Parse expression string into config dict
- **`explain(combinator)`**: Generate plain English explanation of a rule
- **`use_tracing(hook, config)`**: Context manager to enable tracing
- **`run_traced(combinator, ctx, hook, config)`**: Run with explicit tracing
- **`use_cache()`**: Context manager to enable caching

### Tracing Classes

- **`TraceHook`**: Protocol for custom trace hooks
- **`TraceConfig`**: Configuration for tracing behavior
- **`PrintHook`**: Simple stdout tracing
- **`LoggingHook`**: Python logging integration
- **`OpenTelemetryHook`**: OpenTelemetry integration

### Validation Classes

- **`ValidationResult`**: Result with ok, errors, and ctx
- **`ValidatingPredicate`**: Predicate with error message support

### Async Classes

- **`AsyncCombinator`**: Base class for async combinators
- **`AsyncPredicate`**: Async predicate
- **`AsyncTransform`**: Async transform
- **`AsyncRetry`**: Async retry with backoff

### Retry & Caching

- **`Retry`**: Retry combinator with configurable backoff
- **`CachedPredicate`**: Predicate with result caching

### Temporal Combinators

- **`during_hours(start, end)`**: Check if current hour is in range
- **`on_weekdays()`**: Check if today is Monday-Friday
- **`on_days(*days)`**: Check if today is one of the specified days
- **`after_date(year, month, day)`**: Check if today is after date
- **`before_date(year, month, day)`**: Check if today is before date
- **`between_dates(start, end)`**: Check if today is in date range

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
| `tracing_example.py`    | Tracing, debugging, and explaining rules          |
| `validation_example.py` | Validation with error messages                    |
| `async_example.py`      | Async rules, transforms, and retry                |
| `temporal_example.py`   | Time-based and date-based rules                   |
| `access_control.kpz`    | Example access control expression                 |
| `trading.kpz`           | Example trading permission expression             |

Run examples:

```bash
cd kompoz
python examples/rules_example.py
python examples/validation_example.py
python examples/async_example.py
python examples/temporal_example.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.
