# Examples

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

## Example Files

The `examples/` directory contains runnable examples:

| File                        | Description                                       |
| --------------------------- | ------------------------------------------------- |
| `rules_example.py`          | Using `@rule` and `@rule_args` decorators         |
| `transforms_example.py`     | Using `@pipe` and `@pipe_args` for data pipelines |
| `registry_example.py`       | Loading rules from `.kpz` files                   |
| `tracing_example.py`        | Tracing, debugging, and explaining rules          |
| `validation_example.py`     | Validation with error messages                    |
| `async_example.py`          | Async rules, transforms, and retry                |
| `temporal_example.py`       | Time-based and date-based rules                   |
| `functional_example.py`     | Functional programming patterns and composition   |
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

Run any example:

```bash
cd kompoz
python examples/rules_example.py
python examples/validation_example.py
python examples/async_example.py
python examples/temporal_example.py
```
