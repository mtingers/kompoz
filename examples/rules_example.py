"""
Example: Using @rule decorators for access control

This example shows how to use the @rule and @rule_args decorators
to create composable predicates with operator overloading.
"""

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

# AND: all must pass
active_admin = is_admin & is_active

# OR: any must pass
privileged = is_admin | is_active

# NOT: invert result
not_banned = ~is_banned

# Parameterized: call to get a Predicate
mature_account = account_older_than(30)
good_credit = credit_above(650)

# Complex composition
can_access = is_admin | (is_active & ~is_banned & account_older_than(30))

can_trade = is_admin | (
    is_active & ~is_banned & account_older_than(7) & credit_above(600)
)

# Chaining with >> (then) - runs both, keeps second result
logged_access = is_admin >> is_active  # logs admin check, returns active check


# =============================================================================
# Run the rules
# =============================================================================

if __name__ == "__main__":
    users = [
        User("Alice", is_admin=True),
        User("Bob", is_active=True, account_age_days=60, credit_score=700),
        User("Charlie", is_active=True, is_banned=True),
        User("Dave", is_active=True, account_age_days=5, credit_score=400),
    ]

    print("=== Access Check ===")
    print(f"Rule: {can_access}\n")

    for user in users:
        ok, _ = can_access.run(user)
        status = "✓ granted" if ok else "✗ denied"
        print(f"  {user.name}: {status}")

    print("\n=== Trade Check ===")
    print(f"Rule: {can_trade}\n")

    for user in users:
        ok, _ = can_trade.run(user)
        status = "✓ can trade" if ok else "✗ cannot trade"
        print(f"  {user.name}: {status}")

    print("\n=== Individual Rule Results for Bob ===")
    bob = users[1]
    print(f"  is_admin: {is_admin.run(bob)[0]}")
    print(f"  is_active: {is_active.run(bob)[0]}")
    print(f"  is_banned: {is_banned.run(bob)[0]}")
    print(f"  ~is_banned: {not_banned.run(bob)[0]}")
    print(f"  account_older_than(30): {mature_account.run(bob)[0]}")
    print(f"  credit_above(650): {good_credit.run(bob)[0]}")
