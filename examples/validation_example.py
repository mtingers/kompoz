"""
Example: Validation with Error Messages

This example shows how to use validating rules that collect
descriptive error messages when checks fail.
"""

from dataclasses import dataclass

from kompoz import vrule, vrule_args


@dataclass
class User:
    name: str
    email: str = ""
    is_admin: bool = False
    is_active: bool = True
    is_banned: bool = False
    account_age_days: int = 0
    credit_score: int = 500


# =============================================================================
# Define Validating Rules
# =============================================================================


@vrule(error="User {ctx.name} must be an admin")
def is_admin(user: User) -> bool:
    return user.is_admin


@vrule(error="User {ctx.name} account is not active")
def is_active(user: User) -> bool:
    return user.is_active


@vrule(error="{ctx.name} is BANNED!")
def not_banned(user: User) -> bool:
    return not user.is_banned


@vrule(error="Email is required")
def has_email(user: User) -> bool:
    return bool(user.email)


@vrule_args(error="Account must be older than {arg0} days but was {ctx.account_age_days}")
def account_older_than(user: User, days: int) -> bool:
    return user.account_age_days > days


@vrule_args(error="Credit score for must be at least {score} but was {ctx.credit_score}")
def credit_at_least(user: User, score: int) -> bool:
    return user.credit_score >= score


# =============================================================================
# Compose Rules
# =============================================================================

# All checks must pass, collecting all error messages
can_trade = is_active & not_banned & account_older_than(30) & credit_at_least(600)

# Admin or regular user requirements
can_access = is_admin | (is_active & not_banned & has_email)


# =============================================================================
# Run Validation
# =============================================================================

if __name__ == "__main__":
    users = [
        User("Alice", "alice@example.com", is_admin=True),
        User(
            "Bob",
            "bob@example.com",
            is_active=True,
            account_age_days=60,
            credit_score=700,
        ),
        User(
            "Charlie",
            "",
            is_active=True,
            is_banned=True,
            account_age_days=10,
            credit_score=400,
        ),
        User(
            "Dave",
            "dave@example.com",
            is_active=False,
            account_age_days=100,
            credit_score=800,
        ),
    ]

    print("=" * 60)
    print("TRADING VALIDATION")
    print(
        "Rule: is_active & not_banned & account_older_than(30) & credit_at_least(600)"
    )
    print("=" * 60)
    print()

    for user in users:
        result = can_trade.validate(user)
        status = "✓ APPROVED" if result.ok else "✗ DENIED"
        print(f"{user.name}: {status}")
        if result.errors:
            for error in result.errors:
                print(f"    - {error}")
        print()

    print("=" * 60)
    print("ACCESS VALIDATION")
    print("Rule: is_admin | (is_active & not_banned & has_email)")
    print("=" * 60)
    print()

    for user in users:
        result = can_access.validate(user)
        status = "✓ ACCESS GRANTED" if result.ok else "✗ ACCESS DENIED"
        print(f"{user.name}: {status}")
        if result.errors:
            for error in result.errors:
                print(f"    - {error}")
        print()

    # Using raise_if_invalid
    print("=" * 60)
    print("RAISE IF INVALID")
    print("=" * 60)
    print()

    user = User("BadUser", is_banned=True, account_age_days=5)
    result = can_trade.validate(user)
    try:
        result.raise_if_invalid()
        print("Validation passed!")
    except ValueError as e:
        print(f"Validation failed: {e}")
