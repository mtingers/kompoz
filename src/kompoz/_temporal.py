"""Time-based / temporal predicates."""

from __future__ import annotations

from datetime import date, datetime

from kompoz._core import Combinator
from kompoz._types import T


# =============================================================================
# Time-Based / Temporal Predicates
# =============================================================================


class during_hours(Combinator[T]):
    """
    Predicate that passes only during specified hours.

    By default, the end hour is exclusive (e.g., during_hours(9, 17) means
    9:00-16:59). Set inclusive_end=True to include the end hour.

    Example:
        # 9:00 AM to 4:59 PM (end exclusive, default)
        business_hours = during_hours(9, 17)

        # 9:00 AM to 5:59 PM (end inclusive)
        business_hours = during_hours(9, 17, inclusive_end=True)

        # Overnight: 10:00 PM to 5:59 AM
        night_shift = during_hours(22, 6)

        # With timezone
        trading_hours = during_hours(9, 16, tz="America/New_York")

    Args:
        start_hour: Start hour (0-23), inclusive
        end_hour: End hour (0-23), exclusive by default
        tz: Optional timezone name (e.g., "America/New_York")
        inclusive_end: If True, include the end hour (default: False)
    """

    def __init__(
        self,
        start_hour: int,
        end_hour: int,
        tz: str | None = None,
        inclusive_end: bool = False,
    ):
        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
            raise ValueError("Hours must be 0-23")

        self.start_hour = start_hour
        self.end_hour = end_hour
        self.tz = tz
        self.inclusive_end = inclusive_end

    def _get_current_hour(self) -> int:
        """Get current hour, optionally in specified timezone."""
        if self.tz:
            try:
                from zoneinfo import ZoneInfo

                now = datetime.now(ZoneInfo(self.tz))
            except ImportError:
                # Fallback for Python < 3.9
                now = datetime.now()
        else:
            now = datetime.now()
        return now.hour

    def _execute(self, ctx: T) -> tuple[bool, T]:
        hour = self._get_current_hour()

        if self.start_hour <= self.end_hour:
            # Normal range (e.g., 9 to 17)
            if self.inclusive_end:
                ok = self.start_hour <= hour <= self.end_hour
            else:
                ok = self.start_hour <= hour < self.end_hour
        else:
            # Overnight range (e.g., 22 to 6)
            if self.inclusive_end:
                ok = hour >= self.start_hour or hour <= self.end_hour
            else:
                ok = hour >= self.start_hour or hour < self.end_hour

        return ok, ctx

    def __repr__(self) -> str:
        if self.inclusive_end:
            return (
                f"during_hours({self.start_hour}, {self.end_hour}, inclusive_end=True)"
            )
        return f"during_hours({self.start_hour}, {self.end_hour})"


class on_weekdays(Combinator[T]):
    """
    Predicate that passes only on weekdays (Monday-Friday).

    Example:
        weekday_only = on_weekdays()
        can_trade = is_active & on_weekdays() & during_hours(9, 16)
    """

    def _execute(self, ctx: T) -> tuple[bool, T]:
        # Monday = 0, Sunday = 6
        weekday = datetime.now().weekday()
        return weekday < 5, ctx

    def __repr__(self) -> str:
        return "on_weekdays()"


class on_days(Combinator[T]):
    """
    Predicate that passes only on specified days of the week.

    Example:
        # Monday, Wednesday, Friday
        mwf = on_days(0, 2, 4)

        # Weekends only
        weekends = on_days(5, 6)
    """

    def __init__(self, *days: int):
        """
        Args:
            days: Day numbers where Monday=0, Sunday=6
        """
        for d in days:
            if not 0 <= d <= 6:
                raise ValueError("Days must be 0-6 (Monday=0, Sunday=6)")
        self.days = set(days)

    def _execute(self, ctx: T) -> tuple[bool, T]:
        weekday = datetime.now().weekday()
        return weekday in self.days, ctx

    def __repr__(self) -> str:
        return f"on_days({', '.join(map(str, sorted(self.days)))})"


class after_date(Combinator[T]):
    """
    Predicate that passes only after a specified date.

    Example:
        # Feature available after launch
        post_launch = after_date(2024, 6, 1)

        # Using date object
        from datetime import date
        post_launch = after_date(date(2024, 6, 1))
    """

    def __init__(
        self, year_or_date: int | date, month: int | None = None, day: int | None = None
    ):
        if isinstance(year_or_date, date):
            self.date = year_or_date
        else:
            if month is None or day is None:
                raise ValueError("Must provide month and day with year")
            self.date = date(year_or_date, month, day)

    def _execute(self, ctx: T) -> tuple[bool, T]:
        today = date.today()
        return today > self.date, ctx

    def __repr__(self) -> str:
        return f"after_date({self.date})"


class before_date(Combinator[T]):
    """
    Predicate that passes only before a specified date.

    Example:
        # Promo ends on specific date
        promo_active = before_date(2024, 12, 31)
    """

    def __init__(
        self, year_or_date: int | date, month: int | None = None, day: int | None = None
    ):
        if isinstance(year_or_date, date):
            self.date = year_or_date
        else:
            if month is None or day is None:
                raise ValueError("Must provide month and day with year")
            self.date = date(year_or_date, month, day)

    def _execute(self, ctx: T) -> tuple[bool, T]:
        today = date.today()
        return today < self.date, ctx

    def __repr__(self) -> str:
        return f"before_date({self.date})"


class between_dates(Combinator[T]):
    """
    Predicate that passes only between two dates (inclusive).

    Example:
        # Holiday promotion
        holiday_promo = between_dates(date(2024, 12, 20), date(2024, 12, 31))

        # Q1 only
        q1 = between_dates(2024, 1, 1, 2024, 3, 31)
    """

    start_date: date
    end_date: date

    def __init__(
        self,
        start: date | int,
        end_or_start_month: date | int,
        start_day: int | None = None,
        end_year: int | None = None,
        end_month: int | None = None,
        end_day: int | None = None,
    ):
        if isinstance(start, date) and isinstance(end_or_start_month, date):
            self.start_date = start
            self.end_date = end_or_start_month
        elif isinstance(start, int) and isinstance(end_or_start_month, int):
            # Constructor: between_dates(y1, m1, d1, y2, m2, d2)
            if (
                start_day is None
                or end_year is None
                or end_month is None
                or end_day is None
            ):
                raise ValueError("Must provide all 6 arguments for date range")
            self.start_date = date(start, end_or_start_month, start_day)
            self.end_date = date(end_year, end_month, end_day)
        else:
            raise TypeError("Arguments must be either two date objects or six integers")

    def _execute(self, ctx: T) -> tuple[bool, T]:
        today = date.today()
        return self.start_date <= today <= self.end_date, ctx

    def __repr__(self) -> str:
        return f"between_dates({self.start_date}, {self.end_date})"


