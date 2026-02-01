"""
Example: Time-Based Rules

This example shows how to use temporal predicates to create
rules that depend on time, date, and day of week.
"""

from dataclasses import dataclass
from datetime import date, datetime

from kompoz import (
    after_date,
    before_date,
    between_dates,
    during_hours,
    on_days,
    on_weekdays,
    rule,
)


@dataclass
class TradeRequest:
    user_id: int
    symbol: str
    amount: float
    is_premium: bool = False


@dataclass
class Promotion:
    code: str
    discount: float


# =============================================================================
# Basic Rules
# =============================================================================


@rule
def is_premium_user(req: TradeRequest) -> bool:
    return req.is_premium


@rule
def is_valid_amount(req: TradeRequest) -> bool:
    return 0 < req.amount <= 100000


# =============================================================================
# Time-Based Rules
# =============================================================================

# Trading hours: 9:30 AM to 4:00 PM
market_hours = during_hours(9, 16)

# Extended hours for premium: 7 AM to 8 PM
extended_hours = during_hours(7, 20)

# Weekdays only (no weekend trading)
trading_days = on_weekdays()

# Specific maintenance window: Sundays
maintenance_day = on_days(6)  # Sunday = 6

# Feature launch date
feature_launch = after_date(2025, 1, 1)

# Promotion period
holiday_promo = between_dates(date(2025, 12, 20), date(2025, 12, 31))

# Beta period ends
beta_ends = before_date(2026, 6, 1)


# =============================================================================
# Composed Rules
# =============================================================================

# Standard trading: valid amount + market hours + weekdays
can_trade_standard = is_valid_amount & market_hours & trading_days

# Premium trading: extended hours
can_trade_premium = is_premium_user & is_valid_amount & extended_hours & trading_days

# Combined: premium OR standard rules
can_trade = can_trade_premium | can_trade_standard

# System available (not during maintenance)
system_available = ~maintenance_day

# New feature access (after launch, during beta)
can_access_beta_feature = feature_launch & beta_ends


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    now = datetime.now()
    today = date.today()

    print("=" * 60)
    print("CURRENT TIME INFO")
    print("=" * 60)
    print()
    print(f"  Date: {today}")
    print(f"  Time: {now.strftime('%H:%M')}")
    print(f"  Day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][now.weekday()]}")
    print()

    # Test requests
    requests = [
        TradeRequest(1, "AAPL", 1000, is_premium=True),
        TradeRequest(2, "GOOG", 5000, is_premium=False),
        TradeRequest(3, "TSLA", 500000, is_premium=True),  # Too large
    ]

    print("=" * 60)
    print("TEMPORAL CHECKS")
    print("=" * 60)
    print()

    req = requests[0]

    checks = [
        ("Market hours (9-16)", market_hours),
        ("Extended hours (7-20)", extended_hours),
        ("Weekdays only", trading_days),
        ("Not maintenance day", system_available),
        ("After 2025-01-01", feature_launch),
        ("Before 2026-06-01", beta_ends),
    ]

    for name, check in checks:
        ok, _ = check.run(req)
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    print()
    print("=" * 60)
    print("TRADING PERMISSION")
    print("=" * 60)
    print()

    for req in requests:
        # Check standard trading
        ok_std, _ = can_trade_standard.run(req)
        ok_prm, _ = can_trade_premium.run(req)
        ok_any, _ = can_trade.run(req)

        print(
            f"User {req.user_id} - {req.symbol} ${req.amount:,.0f} (premium={req.is_premium})"
        )
        print(f"  Standard trading: {'✓' if ok_std else '✗'}")
        print(f"  Premium trading:  {'✓' if ok_prm else '✗'}")
        print(f"  Can trade (any):  {'✓' if ok_any else '✗'}")
        print()

    print("=" * 60)
    print("PROMOTION CHECK")
    print("=" * 60)
    print()

    promo = Promotion("HOLIDAY2025", 0.20)
    ok, _ = holiday_promo.run(promo)
    print(f"Holiday promo active (Dec 20-31): {'✓ YES' if ok else '✗ NO'}")

    print()
    print("=" * 60)
    print("COMPOSING TIME RULES")
    print("=" * 60)
    print()

    # Example: Flash sale - only on Fridays from 12-2 PM
    friday_only = on_days(4)  # Friday
    lunch_hours = during_hours(12, 14)
    flash_sale = friday_only & lunch_hours

    ok, _ = flash_sale.run(None)  # Context doesn't matter for time checks
    print(f"Flash sale active (Fri 12-2pm): {'✓ YES' if ok else '✗ NO'}")

    # Night owl mode: available 10 PM to 6 AM
    night_hours = during_hours(22, 6)  # Overnight range
    ok, _ = night_hours.run(None)
    print(f"Night owl mode (10pm-6am): {'✓ YES' if ok else '✗ NO'}")
