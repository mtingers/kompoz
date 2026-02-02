# Operators

Kompoz uses Python's operator overloading to provide an intuitive syntax for composing rules.

## Operator Table

| Operator                | Meaning    | Behavior                                   |
| ----------------------- | ---------- | ------------------------------------------ |
| `a & b`                 | AND / then | Run `b` only if `a` succeeds               |
| `a \| b`                | OR / else  | Run `b` only if `a` fails                  |
| `~a`                    | NOT        | Invert success/failure                     |
| `a >> b`                | THEN       | Always run both, keep `b`'s result         |
| `a.if_else(b, c)`       | IF/ELSE    | If `a` succeeds run `b`, otherwise run `c` |

## The `>>` (THEN) Operator

The `>>` operator is useful for pipelines where you want to run steps unconditionally:

```python
# Logging pipeline - log runs regardless of validation result
pipeline = validate_input >> log_attempt >> process_data

# Cleanup pattern - cleanup always runs
operation = do_work >> cleanup
```

## Conditional Branching

Use `.if_else()` or the standalone `if_then_else()` for explicit branching. Unlike `|` (which is a fallback), conditional branching always executes exactly one branch:

```python
from kompoz import if_then_else

# Method syntax
pricing = is_premium.if_else(apply_discount, charge_full_price)

# Function syntax
pricing = if_then_else(is_premium, apply_discount, charge_full_price)

ok, user = pricing.run(user)
```

## Operator Precedence

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
