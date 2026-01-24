"""
Example: Using @pipe decorators for data pipelines

This example shows how to use the @pipe and @pipe_args decorators
to create composable data transformations.
"""

from dataclasses import dataclass

from kompoz import pipe, pipe_args, rule


@dataclass
class Data:
    value: float
    error: str | None = None


# =============================================================================
# Simple transforms (single argument)
# =============================================================================


@pipe
def double(data: Data) -> Data:
    """Multiply value by 2."""
    return Data(data.value * 2)


@pipe
def square(data: Data) -> Data:
    """Square the value."""
    return Data(data.value**2)


@pipe
def negate(data: Data) -> Data:
    """Negate the value."""
    return Data(-data.value)


@pipe
def to_int(data: Data) -> Data:
    """Convert to integer."""
    return Data(int(data.value))


@pipe
def validate_positive(data: Data) -> Data:
    """Raise error if not positive (will cause transform to fail)."""
    if data.value <= 0:
        raise ValueError("Value must be positive")
    return data


# =============================================================================
# Parameterized transforms (extra arguments)
# =============================================================================


@pipe_args
def add(data: Data, n: float) -> Data:
    """Add N to the value."""
    return Data(data.value + n)


@pipe_args
def multiply(data: Data, factor: float) -> Data:
    """Multiply by factor."""
    return Data(data.value * factor)


@pipe_args
def clamp(data: Data, min_val: float, max_val: float) -> Data:
    """Clamp value to range."""
    return Data(max(min_val, min(max_val, data.value)))


@pipe_args
def round_to(data: Data, decimals: int) -> Data:
    """Round to N decimal places."""
    return Data(round(data.value, decimals))


# =============================================================================
# Predicates for conditional logic
# =============================================================================


@rule
def is_positive(data: Data) -> bool:
    """Check if value is positive."""
    return data.value > 0


@rule
def is_even(data: Data) -> bool:
    """Check if value is even integer."""
    return data.value == int(data.value) and int(data.value) % 2 == 0


# =============================================================================
# Compose pipelines
# =============================================================================

# Simple chain: double then add 10
double_and_add = double & add(10)

# Multi-step: double -> square -> round
process = double & square & round_to(2)

# With validation: only process positive numbers
safe_process = is_positive & double & square

# Fallback with OR: try primary, fallback to simpler transform
with_fallback = validate_positive | negate

# Complex pipeline
normalize = (
    clamp(0, 100)  # ensure 0-100 range
    & multiply(0.01)  # convert to 0-1
    & round_to(4)  # round to 4 decimals
)


# =============================================================================
# Run the pipelines
# =============================================================================

if __name__ == "__main__":
    test_values = [5.0, -3.0, 42.5, 0.0, 150.0]

    print("=== Double and Add 10 ===")
    print("Pipeline: double & add(10)\n")

    for val in test_values:
        data = Data(val)
        ok, result = double_and_add.run(data)
        print(f"  {val} -> {result.value} (ok={ok})")

    print("\n=== Safe Process (positive only) ===")
    print("Pipeline: is_positive & double & square\n")

    for val in test_values:
        data = Data(val)
        ok, result = safe_process.run(data)
        if ok:
            print(f"  {val} -> {result.value}")
        else:
            print(f"  {val} -> SKIPPED (not positive)")

    print("\n=== Normalize to 0-1 ===")
    print("Pipeline: clamp(0, 100) & multiply(0.01) & round_to(4)\n")

    for val in test_values:
        data = Data(val)
        ok, result = normalize.run(data)
        print(f"  {val} -> {result.value}")

    print("\n=== With Fallback ===")
    print("Pipeline: validate_positive | negate\n")
    print("(If positive validation fails, negate instead)\n")

    for val in test_values:
        data = Data(val)
        ok, result = with_fallback.run(data)
        print(f"  {val} -> {result.value}")

    print("\n=== Step-by-step trace for 5.0 ===")
    data = Data(5.0)
    print(f"  Input: {data.value}")

    _, data = double.run(data)
    print(f"  After double: {data.value}")

    _, data = square.run(data)
    print(f"  After square: {data.value}")

    _, data = round_to(2).run(data)
    print(f"  After round_to(2): {data.value}")
