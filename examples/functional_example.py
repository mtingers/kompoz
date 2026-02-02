"""
Example: Functional programming patterns with Kompoz

This example shows how Kompoz combinators map to common functional
programming idioms: pipelines, railway-oriented programming,
fallback chains, conditional branching, and pure error handling.
"""

from dataclasses import dataclass, replace

from kompoz import Retry, if_then_else, pipe, pipe_args, rule, rule_args

# =============================================================================
# Domain model (frozen for immutability)
# =============================================================================


@dataclass(frozen=True)
class Order:
    items: list[str]
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    discount: float = 0.0
    status: str = "pending"


# =============================================================================
# 1. Pipeline composition — chain transforms with &
# =============================================================================


@pipe
def calculate_subtotal(order: Order) -> Order:
    """Sum item prices (simplified: $9.99 per item)."""
    return replace(order, subtotal=len(order.items) * 9.99)


@pipe_args
def apply_tax(order: Order, rate: float) -> Order:
    """Apply tax rate to the subtotal."""
    return replace(order, tax=round(order.subtotal * rate, 2))


@pipe
def compute_total(order: Order) -> Order:
    """Compute final total from subtotal, tax, and discount."""
    return replace(order, total=round(order.subtotal + order.tax - order.discount, 2))


# Compose left-to-right: each step feeds the next
checkout = calculate_subtotal & apply_tax(0.08) & compute_total


# =============================================================================
# 2. Railway-oriented programming — failures short-circuit
# =============================================================================


@rule
def has_items(order: Order) -> bool:
    """Order must contain at least one item."""
    return len(order.items) > 0


@pipe
def validate_inventory(order: Order) -> Order:
    """Raise if any item is out of stock."""
    if "out_of_stock" in order.items:
        raise ValueError("Item unavailable")
    return order


# If any step fails the rest is skipped automatically
safe_checkout = has_items & validate_inventory & checkout


# =============================================================================
# 3. Fallback chains — try alternatives with |
# =============================================================================


@pipe
def from_cache(order: Order) -> Order:
    """Simulate cache miss."""
    raise KeyError("cache miss")


@pipe
def from_primary(order: Order) -> Order:
    """Simulate primary service down."""
    raise ConnectionError("primary down")


@pipe
def from_fallback(order: Order) -> Order:
    """Fallback: apply a flat shipping estimate."""
    return replace(order, tax=5.0)


# First success wins, rest are skipped
resolve_tax = from_cache | from_primary | from_fallback


# =============================================================================
# 4. Conditional branching — if/then/else
# =============================================================================


@rule
def is_premium(order: Order) -> bool:
    """Premium orders have 5+ items."""
    return len(order.items) >= 5


@pipe
def apply_premium_discount(order: Order) -> Order:
    """20% off for premium orders."""
    return replace(order, discount=round(order.subtotal * 0.20, 2))


@pipe
def no_discount(order: Order) -> Order:
    """Standard orders get no discount."""
    return replace(order, discount=0.0)


# Exactly one branch runs (not a fallback)
apply_pricing = if_then_else(is_premium, apply_premium_discount, no_discount)


# =============================================================================
# 5. Higher-order combinators — parameterized factories
# =============================================================================


@rule_args
def min_order_value(order: Order, threshold: float) -> bool:
    """Order subtotal must meet a minimum."""
    return order.subtotal >= threshold


@pipe_args
def add_flat_fee(order: Order, fee: float) -> Order:
    """Add a flat fee to the total."""
    return replace(order, total=order.total + fee)


# Partially apply to create specialized combinators
qualifies_for_free_shipping = min_order_value(50.0)
add_shipping = add_flat_fee(5.99)


# =============================================================================
# 6. Pure error handling — run_with_error()
# =============================================================================


@pipe
def parse_quantity(raw: str) -> str:
    """Parse and validate a quantity string."""
    n = int(raw)
    if n <= 0:
        raise ValueError("quantity must be positive")
    return raw


# =============================================================================
# 7. Retry as a combinator
# =============================================================================

attempt_count = 0


@pipe
def flaky_service(order: Order) -> Order:
    """Simulate a service that fails twice then succeeds."""
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ConnectionError(f"attempt {attempt_count} failed")
    return replace(order, status="confirmed")


resilient_confirm = Retry(flaky_service, max_attempts=5, backoff=0.0)


# =============================================================================
# 8. Full pipeline — combining patterns
# =============================================================================

full_pipeline = (
    has_items  # gate: must have items
    & validate_inventory  # gate: all items in stock
    & calculate_subtotal  # transform: compute subtotal
    & apply_pricing  # branch: premium vs standard discount
    & apply_tax(0.08)  # transform: add tax
    & compute_total  # transform: final total
)


# =============================================================================
# Run examples
# =============================================================================

if __name__ == "__main__":
    # --- 1. Pipeline composition ---
    print("=== 1. Pipeline Composition ===")
    print("Pipeline: calculate_subtotal & apply_tax(0.08) & compute_total\n")

    order = Order(items=["widget", "gadget", "gizmo"])
    ok, result = checkout.run(order)
    print(
        f"  {len(order.items)} items -> subtotal=${result.subtotal}, tax=${result.tax}, total=${result.total}"
    )

    # --- 2. Railway-oriented programming ---
    print("\n=== 2. Railway-Oriented Programming ===")
    print("Pipeline: has_items & validate_inventory & checkout\n")

    for items in [["a", "b"], [], ["a", "out_of_stock"]]:
        ok, result = safe_checkout.run(Order(items=items))
        if ok:
            print(f"  {items!r:30s} -> total=${result.total}")
        else:
            print(f"  {items!r:30s} -> FAILED (short-circuited)")

    # --- 3. Fallback chains ---
    print("\n=== 3. Fallback Chains ===")
    print("Pipeline: from_cache | from_primary | from_fallback\n")

    ok, result = resolve_tax.run(Order(items=["x"]))
    print(f"  tax=${result.tax} (resolved via fallback)")

    # --- 4. Conditional branching ---
    print("\n=== 4. Conditional Branching ===")
    print("Pipeline: if_then_else(is_premium, apply_premium_discount, no_discount)\n")

    for items in [["a", "b"], ["a", "b", "c", "d", "e", "f"]]:
        order = Order(items=items, subtotal=len(items) * 9.99)
        ok, result = apply_pricing.run(order)
        label = "premium" if len(items) >= 5 else "standard"
        print(f"  {len(items)} items ({label}) -> discount=${result.discount}")

    # --- 5. Higher-order combinators ---
    print("\n=== 5. Higher-Order Combinators ===")
    print("Pipeline: qualifies_for_free_shipping(50) | add_shipping(5.99)\n")

    shipping_pipeline = (
        calculate_subtotal & compute_total & (qualifies_for_free_shipping | add_shipping)
    )
    for count in [3, 6]:
        order = Order(items=["x"] * count)
        ok, result = shipping_pipeline.run(order)
        print(f"  {count} items (subtotal=${result.subtotal}) -> total=${result.total}")

    # --- 6. Pure error handling ---
    print("\n=== 6. Pure Error Handling (run_with_error) ===\n")

    for raw in ["5", "abc", "-1"]:
        ok, result, error = parse_quantity.run_with_error(raw)
        if ok:
            print(f"  {raw!r:5s} -> ok")
        else:
            print(f"  {raw!r:5s} -> error: {error}")

    # --- 7. Retry ---
    print("\n=== 7. Retry as a Combinator ===\n")

    attempt_count = 0
    info = resilient_confirm.run_with_info(Order(items=["x"]))
    print(f"  ok={info.ok}, attempts={info.attempts_made}, status={info.ctx.status}")

    # --- 8. Full pipeline ---
    print("\n=== 8. Full Pipeline ===")
    print("Pipeline: has_items & validate_inventory & calculate_subtotal")
    print("          & apply_pricing & apply_tax(0.08) & compute_total\n")

    orders = [
        Order(items=["a", "b", "c"]),
        Order(items=["a", "b", "c", "d", "e", "f"]),
        Order(items=[]),
        Order(items=["a", "out_of_stock"]),
    ]

    for order in orders:
        ok, result = full_pipeline.run(order)
        if ok:
            print(
                f"  {len(order.items)} items -> "
                f"sub=${result.subtotal} disc=${result.discount} "
                f"tax=${result.tax} total=${result.total}"
            )
        else:
            print(f"  {len(order.items)} items -> FAILED")
