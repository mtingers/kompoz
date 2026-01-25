"""
Example: Tracing and Explaining Rules

This example shows how to use tracing hooks and the explain function
to understand what rules do and how they execute.
"""

from dataclasses import dataclass

from kompoz import (
    LoggingHook,
    PrintHook,
    TraceConfig,
    explain,
    rule,
    rule_args,
    run_traced,
    use_tracing,
)


@dataclass
class User:
    name: str
    is_admin: bool = False
    is_active: bool = True
    is_banned: bool = False
    account_age_days: int = 0
    credit_score: int = 500


# =============================================================================
# Define Rules
# =============================================================================


@rule
def is_admin(user: User) -> bool:
    return user.is_admin


@rule
def is_active(user: User) -> bool:
    return user.is_active


@rule
def is_banned(user: User) -> bool:
    return user.is_banned


@rule_args
def account_older_than(user: User, days: int) -> bool:
    return user.account_age_days > days


@rule_args
def credit_above(user: User, score: int) -> bool:
    return user.credit_score > score


# Build a complex rule
can_access = is_admin | (
    is_active & ~is_banned & account_older_than(30) & credit_above(600)
)


# =============================================================================
# Examples
# =============================================================================

if __name__ == "__main__":
    # Test user
    user = User("Bob", is_active=True, account_age_days=60, credit_score=700)

    # -------------------------------------------------------------------------
    # 1. Explain what the rule does
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("EXPLAIN: What does this rule do?")
    print("=" * 60)
    print()
    print(explain(can_access))
    print()

    # -------------------------------------------------------------------------
    # 2. Run without tracing
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("RUN: Basic execution (no tracing)")
    print("=" * 60)
    print()
    ok, _ = can_access.run(user)
    print(f"User: {user.name}")
    print(f"Result: {'✓ Access granted' if ok else '✗ Access denied'}")
    print()

    # -------------------------------------------------------------------------
    # 3. Run with tracing (context manager)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("TRACED RUN: Using use_tracing() context manager")
    print("=" * 60)
    print()
    with use_tracing(PrintHook()):
        ok, _ = can_access.run(user)
    print()
    print(f"Result: {'✓ Access granted' if ok else '✗ Access denied'}")
    print()

    # -------------------------------------------------------------------------
    # 4. Run with explicit tracing
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("TRACED RUN: Using run_traced() explicitly")
    print("=" * 60)
    print()
    ok, _ = run_traced(can_access, user, PrintHook(show_ctx=True))
    print()
    print(f"Result: {'✓ Access granted' if ok else '✗ Access denied'}")
    print()

    # -------------------------------------------------------------------------
    # 5. Trace only leaf predicates (skip AND/OR/NOT)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("TRACED RUN: Leaf predicates only")
    print("=" * 60)
    print()
    with use_tracing(PrintHook(), TraceConfig(include_leaf_only=True)):
        ok, _ = can_access.run(user)
    print()
    print(f"Result: {'✓ Access granted' if ok else '✗ Access denied'}")
    print()

    # -------------------------------------------------------------------------
    # 6. Limit trace depth
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("TRACED RUN: Max depth = 2")
    print("=" * 60)
    print()
    with use_tracing(PrintHook(), TraceConfig(max_depth=2)):
        ok, _ = can_access.run(user)
    print()
    print(f"Result: {'✓ Access granted' if ok else '✗ Access denied'}")
    print()

    # -------------------------------------------------------------------------
    # 7. Using LoggingHook with Python logging
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("TRACED RUN: Using LoggingHook")
    print("=" * 60)
    print()

    import logging

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    logger = logging.getLogger("kompoz.example")

    with use_tracing(LoggingHook(logger), TraceConfig(nested=False)):
        ok, _ = can_access.run(user)
    print()
    print(f"Result: {'✓ Access granted' if ok else '✗ Access denied'}")
    print()

    # -------------------------------------------------------------------------
    # 8. Custom Hook Example
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("CUSTOM HOOK: Collecting execution trace")
    print("=" * 60)
    print()

    class CollectorHook:
        """Collects trace events into a list."""

        def __init__(self):
            self.events = []

        def on_enter(self, name, ctx, depth):
            self.events.append(("enter", name, depth))
            return len(self.events)

        def on_exit(self, span, name, ok, duration_ms, depth):
            self.events.append(("exit", name, ok, duration_ms, depth))

        def on_error(self, span, name, error, duration_ms, depth):
            print(f"{'  ' * depth}<- {name} ERROR: {error}")

    collector = CollectorHook()
    with use_tracing(collector, TraceConfig(include_leaf_only=True)):
        ok, _ = can_access.run(user)

    print("Collected events:")
    for event in collector.events:
        print(f"  {event}")
    print()
    print(f"Result: {'✓ Access granted' if ok else '✗ Access denied'}")
