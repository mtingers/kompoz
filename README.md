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
    - [Error Tracking](#error-tracking)
  - [Expression DSL](#expression-dsl)
    - [Basic Syntax](#basic-syntax)
    - [Expression Operators](#expression-operators)
    - [Modifiers](#modifiers)
    - [Examples](#examples)
    - [Multi-line Expressions](#multi-line-expressions)
    - [Operator Precedence](#operator-precedence)
    - [Load from File](#load-from-file)
  - [Type Hints](#type-hints)
  - [Pydantic Compatibility](#pydantic-compatibility)
  - [Testing](#testing)
  - [Use Cases](#use-cases)
    - [Access Control](#access-control)
    - [Form Validation](#form-validation)
    - [Data Pipeline with Fallbacks](#data-pipeline-with-fallbacks)
    - [Feature Flags](#feature-flags)
  - [Tracing & Debugging](#tracing-debugging)
    - [Explain Rules](#explain-rules)
    - [Tracing Execution](#tracing-execution)
    - [Async Tracing](#async-tracing)
    - [Trace Configuration](#trace-configuration)
    - [Built-in Hooks](#built-in-hooks)
    - [Custom Hooks](#custom-hooks)
    - [OpenTelemetry Integration](#opentelemetry-integration)
  - [Validation with Error Messages](#validation-with-error-messages)
  - [Async Support](#async-support)
  - [Caching / Memoization](#caching-memoization)
  - [Retry Logic](#retry-logic)
    - [Observability Hooks](#observability-hooks)
  - [Time-Based Rules](#time-based-rules)
  - [Equality and Hashing](#equality-and-hashing)
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

- **Operator Overloading**: Use `&` (and), `|` (or), `~` (not), `>>` (then) for intuitive composition
- **Conditional Branching**: `if_then_else()` and ternary `?:` for explicit control flow
- **Decorator Syntax**: Clean `@rule` and `@rule_args` decorators
- **Parameterized Rules**: `account_older_than(30)` creates reusable predicates
- **Validation with Errors**: `@vrule` / `@async_vrule` decorators collect all error messages
- **Expression DSL**: Human-readable rule expressions with AND/OR/NOT/IF/THEN/ELSE
- **Async Support**: Full async/await support with tracing and validation
- **Caching**: `@cached_rule` and `use_cache()` to memoize expensive predicates
- **Time-Based Rules**: `during_hours()`, `on_weekdays()`, `after_date()`, and more
- **Error Tracking**: Transforms track exceptions via `last_error` attribute
- **Retry with Observability**: Built-in retry logic with hooks for monitoring
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

| Operator                | Meaning    | Behavior                                   |
| ----------------------- | ---------- | ------------------------------------------ |
| `a & b`                 | AND / then | Run `b` only if `a` succeeds               |
| `a \| b`                | OR / else  | Run `b` only if `a` fails                  |
| `~a`                    | NOT        | Invert success/failure                     |
| `a >> b`                | THEN       | Always run both, keep `b`'s result         |
| `a.if_else(b, c)`       | IF/ELSE    | If `a` succeeds run `b`, otherwise run `c` |

The `>>` operator is useful for pipelines where you want to run steps unconditionally:

```python
# Logging pipeline - log runs regardless of validation result
pipeline = validate_input >> log_attempt >> process_data

# Cleanup pattern - cleanup always runs
operation = do_work >> cleanup
```

### Conditional Branching

Use `.if_else()` or the standalone `if_then_else()` for explicit branching. Unlike `|` (which is a fallback), conditional branching always executes exactly one branch:

```python
from kompoz import if_then_else

# Method syntax
pricing = is_premium.if_else(apply_discount, charge_full_price)

# Function syntax
pricing = if_then_else(is_premium, apply_discount, charge_full_price)

ok, user = pricing.run(user)
```

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

### Error Tracking

Transforms track exceptions via the `last_error` attribute:

```python
@pipe
def risky_transform(data):
    return int(data)  # May raise ValueError

ok, result = risky_transform.run("not a number")
if not ok:
    print(f"Failed: {risky_transform.last_error}")
    # Failed: invalid literal for int() with base 10: 'not a number'
```

This also works for async transforms:

```python
@async_pipe
async def fetch_data(url):
    async with aiohttp.get(url) as resp:
        return await resp.json()

ok, result = await fetch_data.run("https://api.example.com")
if not ok:
    print(f"Request failed: {fetch_data.last_error}")
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

| Symbol          | Word             | Meaning                             |
| --------------- | ---------------- | ----------------------------------- |
| `&`             | `AND`            | All conditions must pass            |
| `\|`            | `OR`             | Any condition must pass             |
| `~`, `!`        | `NOT`            | Invert the condition                |
| `>>`            | `THEN`           | Always run both, keep second result |
| `a ? b : c`     | `IF a THEN b ELSE c` | Conditional branching          |
| `()`            |                  | Grouping                            |


### Modifiers

Postfix modifiers add retry and caching behavior:

| Modifier                | Meaning                                 |
| ----------------------- | --------------------------------------- |
| `:retry(n)`             | Retry up to n times on failure          |
| `:retry(n, backoff)`    | Retry with backoff delay (seconds)      |
| `:retry(n, b, true)`    | Exponential backoff                     |
| `:retry(n, b, true, j)` | With jitter                             |
| `:cached`               | Cache result within `use_cache()` scope |

Modifiers can be chained: `rule:cached:retry(3)`

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

1. `IF/THEN/ELSE` / `? :` (conditional branching)
2. `OR` / `|`
3. `THEN` / `>>`
4. `AND` / `&`
5. `NOT` / `~` / `!`
6. `:modifier` (evaluated first, binds tightest)

```python
# This expression:
is_admin | is_active & ~is_banned

# Is parsed as:
is_admin | (is_active & (~is_banned))

# Use parentheses to override:
(is_admin | is_active) & ~is_banned

# Conditionals have lowest precedence:
a | b ? c : d  # Parsed as: (a | b) ? c : d

# THEN is between OR and AND:
a | b >> c & d  # Parsed as: a | ((b >> c) & d)

# Modifiers bind to their immediate left:
a & b:retry(3)  # Only b gets retry, not (a & b)

# Use grouping to apply modifier to compound expression:
(a & b):retry(3)  # Both a and b are retried together
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

With modifiers:

```
# resilient_access.kpz
# Retry flaky permission checks

is_admin
| (is_active
   & ~is_banned
   & fetch_permissions:retry(3, 1.0)
   & account_older_than(30))
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

Since Pydantic models behave like regular Python objects with attribute access, all Kompoz features work out of the box — including validation, async rules, caching, and the expression DSL.

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
# Using Python API
fetch_data = (
    (try_primary_db | try_replica_db | try_cache)
    & validate_schema
    & transform_response
)

# With explicit retry
from kompoz import Retry

resilient_fetch = Retry(
    try_primary_db | try_replica_db,
    max_attempts=3,
    backoff=1.0,
    exponential=True
)

# Using DSL with :retry modifier
reg.load("(try_primary | try_replica):retry(3, 1.0, true) & validate")
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

### Async Tracing

Async combinators fully support tracing via the same `use_tracing()` context manager:

```python
from kompoz import use_tracing, run_async_traced, PrintHook, async_rule

@async_rule
async def check_permission(user):
    return await db.has_permission(user.id)

@async_rule
async def check_quota(user):
    return await db.check_quota(user.id)

can_proceed = check_permission & check_quota

# Option 1: Context manager works with async
with use_tracing(PrintHook()):
    ok, result = await can_proceed.run(user)

# Option 2: Explicit async tracing
ok, result = await run_async_traced(can_proceed, user, PrintHook())
```

Output:

```
-> AsyncAND
  -> AsyncPredicate(check_permission)
  <- AsyncPredicate(check_permission) ✓ (15.23ms)
  -> AsyncPredicate(check_quota)
  <- AsyncPredicate(check_quota) ✓ (8.41ms)
<- AsyncAND ✓ (23.89ms)
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

### Async Validation

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

Async transforms track errors just like sync transforms:

```python
@async_pipe
async def fetch_user_data(user_id):
    return await api.get_user(user_id)

ok, result = await fetch_user_data.run(invalid_id)
if not ok:
    print(f"API error: {fetch_user_data.last_error}")
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

### Observability Hooks

Retry combinators support observability via callbacks and state tracking:

```python
from kompoz import Retry, AsyncRetry

# Callback for monitoring retries
def on_retry(attempt: int, error: Exception | None, delay: float):
    print(f"Retry {attempt}: error={error}, waiting {delay}s")
    metrics.increment("api.retries", tags={"attempt": attempt})

fetch = Retry(
    fetch_from_api,
    max_attempts=3,
    backoff=1.0,
    on_retry=on_retry  # Called before each retry
)

ok, result = fetch.run(request)

# After execution, check state
print(f"Total attempts: {fetch.attempts_made}")
print(f"Last error: {fetch.last_error}")
```

For async retries, the callback can be sync or async:

```python
async def on_retry_async(attempt, error, delay):
    await log_to_service(f"Retry {attempt}")

fetch = AsyncRetry(
    fetch_from_api,
    max_attempts=3,
    on_retry=on_retry_async  # Async callback supported
)
```

## Time-Based Rules

Create rules that depend on time, date, or day of week:

```python
from kompoz import during_hours, on_weekdays, on_days, after_date, before_date, between_dates
from datetime import date

# Time of day (end hour is exclusive by default)
business_hours = during_hours(9, 17)      # 9:00 AM to 4:59 PM
night_mode = during_hours(22, 6)          # 10:00 PM to 5:59 AM (overnight)

# Include the end hour with inclusive_end=True
full_hours = during_hours(9, 17, inclusive_end=True)  # 9:00 AM to 5:59 PM

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

## Equality and Hashing

`Predicate` and `Transform` objects support equality comparison and hashing, making them usable in sets and as dictionary keys:

```python
from kompoz import rule, Predicate

@rule
def is_positive(x):
    return x > 0

# Same function and name = equal
p1 = Predicate(lambda x: x > 0, "check")
p2 = Predicate(lambda x: x > 0, "check")

# Can use in sets (deduplication)
rules = {is_positive, is_positive}  # len(rules) == 1

# Can use as dict keys
rule_docs = {
    is_positive: "Checks if value is greater than zero",
    is_even: "Checks if value is divisible by 2",
}
```

## API Reference

### Core Classes

- **`Combinator[T]`**: Abstract base class for all combinators. Has `.if_else(then, else)` method.
- **`Predicate[T]`**: Checks a condition, doesn't modify context. Supports `__eq__` and `__hash__`.
- **`PredicateFactory[T]`**: Factory for parameterized predicates (created by `@rule_args`)
- **`Transform[T]`**: Transforms context, fails on exception. Has `last_error` attribute. Supports `__eq__` and `__hash__`.
- **`TransformFactory[T]`**: Factory for parameterized transforms (created by `@pipe_args`)
- **`Try[T]`**: Wraps a function, converts exceptions to failure
- **`Registry[T]`**: Register and load rules from expressions
- **`ExpressionParser`**: Parser for human-readable rule expressions

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
- **`@async_vrule`**: Create an async validating rule with error message
- **`@async_vrule_args`**: Create a parameterized async validating rule
- **`@cached_rule`**: Create a rule with result caching

### Functions

- **`parse_expression(text)`**: Parse expression string into config dict
- **`explain(combinator)`**: Generate plain English explanation of a rule
- **`if_then_else(cond, then_branch, else_branch)`**: Create conditional combinator
- **`async_if_then_else(cond, then_branch, else_branch)`**: Create async conditional combinator
- **`use_tracing(hook, config)`**: Context manager to enable tracing
- **`run_traced(combinator, ctx, hook, config)`**: Run with explicit tracing
- **`run_async_traced(combinator, ctx, hook, config)`**: Run async combinator with explicit tracing
- **`use_cache()`**: Context manager to enable caching

### Tracing Classes

- **`TraceHook`**: Protocol for custom trace hooks
- **`TraceConfig`**: Configuration for tracing behavior
- **`PrintHook`**: Simple stdout tracing
- **`LoggingHook`**: Python logging integration
- **`OpenTelemetryHook`**: OpenTelemetry integration

### Validation Classes

- **`ValidationResult`**: Result with ok, errors, and ctx
- **`ValidatingCombinator`**: Base class for validating combinators. Supports `&`, `|`, and `~` operators.
- **`ValidatingPredicate`**: Predicate with error message support
- **`AsyncValidatingCombinator`**: Async base class for validating combinators
- **`AsyncValidatingPredicate`**: Async predicate with error message support

### Async Classes

- **`AsyncCombinator`**: Base class for async combinators. Has `.if_else(then, else)` method. Integrates with `use_tracing()`.
- **`AsyncPredicate`**: Async predicate
- **`AsyncPredicateFactory`**: Factory for parameterized async predicates
- **`AsyncTransform`**: Async transform. Has `last_error` attribute.
- **`AsyncTransformFactory`**: Factory for parameterized async transforms
- **`AsyncRetry`**: Async retry with backoff and observability hooks

### Retry & Caching

- **`Retry`**: Retry combinator with configurable backoff. Has `on_retry` callback, `last_error`, and `attempts_made` attributes.
- **`AsyncRetry`**: Async retry with same features as `Retry`
- **`CachedPredicate`**: Predicate with result caching

### Temporal Combinators

- **`during_hours(start, end, tz=None, inclusive_end=False)`**: Check if current hour is in range. Use `inclusive_end=True` to include the end hour.
- **`on_weekdays()`**: Check if today is Monday-Friday
- **`on_days(*days)`**: Check if today is one of the specified days (0=Monday, 6=Sunday)
- **`after_date(year, month, day)`**: Check if today is after date
- **`before_date(year, month, day)`**: Check if today is before date
- **`between_dates(start, end)`**: Check if today is in date range

### Utility Combinators

- **`Always()`**: Always succeeds
- **`Never()`**: Always fails
- **`Debug(label)`**: Prints context and succeeds

## Examples

The `examples/` directory contains:

| File                        | Description                                       |
| --------------------------- | ------------------------------------------------- |
| `rules_example.py`          | Using `@rule` and `@rule_args` decorators         |
| `transforms_example.py`     | Using `@pipe` and `@pipe_args` for data pipelines |
| `registry_example.py`       | Loading rules from `.kpz` files                   |
| `tracing_example.py`        | Tracing, debugging, and explaining rules          |
| `validation_example.py`     | Validation with error messages                    |
| `async_example.py`          | Async rules, transforms, and retry                |
| `temporal_example.py`       | Time-based and date-based rules                   |
| `then_operator_example.py`  | Using `>>` (THEN) for sequencing                  |
| `access_control.kpz`        | Access control with AND/OR/NOT                    |
| `trading.kpz`               | Tiered trading permissions                        |
| `pipeline.kpz`              | Data pipeline with `>>` (THEN) operator           |
| `pricing.kpz`               | IF/THEN/ELSE conditional branching                |
| `tiered_pricing.kpz`        | Nested IF/THEN/ELSE for multi-tier logic          |
| `content_moderation.kpz`    | Word-syntax keywords (AND, OR, NOT)               |
| `data_enrichment.kpz`       | `:retry` and `:cached` modifiers                  |
| `fraud_detection.kpz`       | Complex nested logic with modifiers               |
| `feature_flags.kpz`         | Ternary `? :` syntax with `:cached`               |

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
