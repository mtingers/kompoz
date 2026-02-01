"""
Example: Using @rule decorators for access control

This example shows how to create a bunch of rules in a
loop and how max recursion won't be hit (stack based
recursion).
"""

from dataclasses import dataclass

from opentelemetry import trace

from kompoz import (
    PrintHook,
    rule,
    rule_args,
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
# Simple rules (single argument)
# =============================================================================


@rule
def is_admin(user: User) -> bool:
    """User has admin privileges."""
    return user.is_admin


@rule
def is_active(user: User) -> bool:
    """User account is active."""
    return user.is_active


@rule
def is_banned(user: User) -> bool:
    """User is banned."""
    return user.is_banned


# =============================================================================
# Parameterized rules (extra arguments)
# =============================================================================


@rule_args
def account_older_than(user: User, days: int) -> bool:
    """User account is older than N days."""
    return user.account_age_days > days


@rule_args
def credit_above(user: User, score: int) -> bool:
    """User credit score is above threshold."""
    return user.credit_score > score


@rule_args
def credit_between(user: User, min_score: int, max_score: int) -> bool:
    """User credit score is within range."""
    return min_score <= user.credit_score <= max_score


# =============================================================================
# Compose rules using operators
# =============================================================================


# =============================================================================
# Run the rules
# =============================================================================

# resource = Resource(attributes={"service.name": "kompoz-tracing-example"})
# trace_provider = TracerProvider(resource=resource)
# otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
# trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
# trace.set_tracer_provider(trace_provider)

tracer = trace.get_tracer(__name__)

if __name__ == "__main__":
    users = [
        User("Alice", is_admin=True),
        User("Bob", is_active=True, account_age_days=60, credit_score=700),
        User("Charlie", is_active=True, is_banned=True),
        User("Dave", is_active=True, account_age_days=5, credit_score=400),
    ]

    can_access = is_admin | (is_active & ~is_banned & account_older_than(30))

    for _ in range(1000):
        can_access = can_access & (is_active | is_admin)

    with use_tracing(PrintHook()):
        ok, _ = can_access.run(users[0])

    # print(explain(can_access))
