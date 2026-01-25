"""
Example: Using the >> (THEN) operator for pipelines

The >> operator runs BOTH sides regardless of success/failure,
and returns the result of the RIGHT side.

This is useful for:
- Data transformation pipelines
- Logging/auditing (run logger >> actual_operation)
- Cleanup operations that should always run
- Chaining transforms where you want all steps to execute

Contrast with &:
- a & b: Run b ONLY if a succeeds (short-circuit)
- a >> b: Run b ALWAYS, return b's result
"""

from dataclasses import dataclass, field

from kompoz import Registry, explain, pipe


@dataclass
class Request:
    raw_data: str
    parsed: dict | None = None
    validated: bool = False
    transformed: dict | None = None
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# =============================================================================
# Python API Example
# =============================================================================


@pipe
def parse_json(req: Request) -> Request:
    """Parse raw JSON data."""
    import json

    try:
        req.parsed = json.loads(req.raw_data)
        print(f"  [parse] Parsed: {req.parsed}")
    except json.JSONDecodeError as e:
        req.errors.append(f"Parse error: {e}")
        print(f"  [parse] Failed: {e}")
    return req


@pipe
def validate_schema(req: Request) -> Request:
    """Validate the parsed data has required fields."""
    if req.parsed is None:
        req.errors.append("No parsed data to validate")
        print("  [validate] Skipped: no parsed data")
        return req

    required = ["name", "value"]
    missing = [f for f in required if f not in req.parsed]
    if missing:
        req.errors.append(f"Missing fields: {missing}")
        print(f"  [validate] Failed: missing {missing}")
    else:
        req.validated = True
        print("  [validate] OK")
    return req


@pipe
def transform_data(req: Request) -> Request:
    """Transform the validated data."""
    if not req.validated:
        print("  [transform] Skipped: not validated")
        return req

    if req.parsed:
        req.transformed = {
            "name": req.parsed["name"].upper(),
            "value": req.parsed["value"] * 2,
        }
        print(f"  [transform] Result: {req.transformed}")
    else:
        req.errors.append("req.parsed is unset??")
    return req


@pipe
def log_result(req: Request) -> Request:
    """Log the final result (always runs with >>)."""
    if req.errors:
        print(f"  [log] Errors: {req.errors}")
    else:
        print(f"  [log] Success: {req.transformed}")
    return req


# Using >> to build a pipeline where all steps run
# Even if validation fails, we still want to log
pipeline_with_logging = parse_json >> validate_schema >> transform_data >> log_result

# Compare with & which would stop at first failure
pipeline_short_circuit = parse_json & validate_schema & transform_data & log_result


# =============================================================================
# DSL Example
# =============================================================================


@dataclass
class Data:
    value: int


reg = Registry[Data]()


@reg.transform
def double(d: Data) -> Data:
    """Multiply value by 2."""
    return Data(d.value * 2)


@reg.transform
def add_ten(d: Data) -> Data:
    """Add 10 to value."""
    return Data(d.value + 10)


@reg.transform
def square(d: Data) -> Data:
    """Square the value."""
    return Data(d.value**2)


@reg.predicate
def is_positive(d: Data) -> bool:
    """Check if value is positive."""
    return d.value > 0


@reg.predicate
def is_even(d: Data) -> bool:
    """Check if value is even."""
    return d.value % 2 == 0


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PYTHON API: >> vs & comparison")
    print("=" * 60)

    # Test with valid JSON
    print("\n--- Valid input: {'name': 'test', 'value': 21} ---")
    req = Request('{"name": "test", "value": 21}')
    ok, result = pipeline_with_logging.run(req)
    print(f"Final: ok={ok}, transformed={result.transformed}")

    # Test with invalid JSON - >> continues to log
    print("\n--- Invalid JSON (>> continues to log) ---")
    req = Request("not json")
    ok, result = pipeline_with_logging.run(req)
    print(f"Final: ok={ok}, errors={result.errors}")

    # Test with missing fields
    print("\n--- Missing fields (>> continues to log) ---")
    req = Request('{"name": "test"}')
    ok, result = pipeline_with_logging.run(req)
    print(f"Final: ok={ok}, errors={result.errors}")

    print()
    print("=" * 60)
    print("DSL EXAMPLES")
    print("=" * 60)

    # Load pipelines from DSL expressions
    print("\n--- Symbol syntax: double >> add_ten >> square ---")
    pipeline1 = reg.load("double >> add_ten >> square")
    ok, result = pipeline1.run(Data(5))
    print("  5 -> double(5)=10 -> add_ten(10)=20 -> square(20)=400")
    print(f"  Result: {result.value} (ok={ok})")

    print("\n--- Word syntax: double THEN add_ten THEN square ---")
    pipeline2 = reg.load("double THEN add_ten THEN square")
    ok, result = pipeline2.run(Data(3))
    print("  3 -> 6 -> 16 -> 256")
    print(f"  Result: {result.value} (ok={ok})")

    print("\n--- Mixed with predicates: is_positive & double >> add_ten ---")
    pipeline3 = reg.load("is_positive & double >> add_ten")
    print("  Precedence: (is_positive & double) >> add_ten")

    ok, result = pipeline3.run(Data(5))
    print(f"  Data(5): {result.value} (ok={ok})")

    ok, result = pipeline3.run(Data(-5))
    print(f"  Data(-5): {result.value} (ok={ok})")
    print("  Note: is_positive fails, but >> still runs add_ten")

    print("\n--- Fallback pipeline: (try_primary | try_backup) >> log ---")
    # Conceptual example of fallback with logging

    print()
    print("=" * 60)
    print("EXPLAIN: What does >> do?")
    print("=" * 60)
    print()

    rule_to_explain = reg.load("double >> add_ten >> square")
    print(explain(rule_to_explain))

    print()
    print("=" * 60)
    print("OPERATOR PRECEDENCE")
    print("=" * 60)
    print("""
    From lowest to highest:

    1. OR  (|)   - Fallback, runs right if left fails
    2. THEN (>>) - Sequence, always runs both
    3. AND (&)   - Chain, runs right only if left succeeds
    4. NOT (~)   - Invert result

    Examples:
      a | b >> c    parses as   a | (b >> c)
      a >> b & c    parses as   a >> (b & c)
      a & b >> c    parses as   (a & b) >> c
      a | b & c     parses as   a | (b & c)
    """)
