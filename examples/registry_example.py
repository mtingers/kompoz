"""
Example: Using Registry to load rules from .kpz files

This example shows how to use the Registry class to define
predicates and load rule expressions from external files.
"""

from dataclasses import dataclass
from pathlib import Path

from kompoz import Registry


@dataclass
class User:
    name: str
    is_admin: bool = False
    is_active: bool = True
    is_banned: bool = False
    is_premium: bool = False
    account_age_days: int = 0
    credit_score: int = 500


# =============================================================================
# Create registry and define predicates
# =============================================================================

reg = Registry[User]()


@reg.predicate
def is_admin(user: User) -> bool:
    return user.is_admin


@reg.predicate
def is_active(user: User) -> bool:
    return user.is_active


@reg.predicate
def is_banned(user: User) -> bool:
    return user.is_banned


@reg.predicate
def is_premium(user: User) -> bool:
    return user.is_premium


@reg.predicate
def account_older_than(user: User, days: int) -> bool:
    return user.account_age_days > days


@reg.predicate
def credit_above(user: User, score: int) -> bool:
    return user.credit_score > score


# =============================================================================
# Load rules from expressions
# =============================================================================

# Inline expression
inline_rule = reg.load("is_admin | (is_active & ~is_banned)")

# Multi-line expression
multiline_rule = reg.load("""
    is_admin
    | (is_active & ~is_banned & account_older_than(30))
""")

# From .kpz file
examples_dir = Path(__file__).parent
access_rule = reg.load_file((examples_dir / "access_control.kpz").as_posix())
trading_rule = reg.load_file((examples_dir / "trading.kpz").as_posix())


# =============================================================================
# Run the rules
# =============================================================================

if __name__ == "__main__":
    users = [
        User("Alice", is_admin=True),
        User("Bob", is_active=True, account_age_days=60, credit_score=700),
        User("Charlie", is_premium=True, is_active=True, account_age_days=30),
        User("Dave", is_active=True, is_banned=True),
        User("Eve", is_active=True, account_age_days=100, credit_score=800),
    ]

    print("=== Access Control (from access_control.kpz) ===\n")
    for user in users:
        ok, _ = access_rule.run(user)
        status = "✓" if ok else "✗"
        print(f"  {status} {user.name}")

    print("\n=== Trading Permissions (from trading.kpz) ===\n")
    for user in users:
        ok, _ = trading_rule.run(user)
        status = "✓" if ok else "✗"
        print(f"  {status} {user.name}")

    print("\n=== User Details ===\n")
    for user in users:
        flags = []
        if user.is_admin:
            flags.append("admin")
        if user.is_premium:
            flags.append("premium")
        if user.is_active:
            flags.append("active")
        if user.is_banned:
            flags.append("BANNED")

        print(f"  {user.name}: {', '.join(flags) or 'regular'}")
        print(f"    Account age: {user.account_age_days} days")
        print(f"    Credit score: {user.credit_score}")
