# Time-Based Rules

Create rules that depend on time, date, or day of week.

## Time of Day

```python
from kompoz import during_hours

# End hour is exclusive by default
business_hours = during_hours(9, 17)      # 9:00 AM to 4:59 PM
night_mode = during_hours(22, 6)          # 10:00 PM to 5:59 AM (overnight)

# Include the end hour with inclusive_end=True
full_hours = during_hours(9, 17, inclusive_end=True)  # 9:00 AM to 5:59 PM
```

## Day of Week

```python
from kompoz import on_weekdays, on_days

weekdays = on_weekdays()                  # Monday-Friday
mwf = on_days(0, 2, 4)                    # Mon, Wed, Fri
weekends = on_days(5, 6)                  # Sat, Sun
```

## Date Ranges

```python
from kompoz import after_date, before_date, between_dates
from datetime import date

launched = after_date(2024, 6, 1)
promo_active = before_date(2024, 12, 31)
q1_only = between_dates(date(2024, 1, 1), date(2024, 3, 31))
```

## Timezone Support

All temporal functions accept an optional `tz` parameter (a timezone name string). Without it, the local system time is used.

```python
from kompoz import during_hours, on_weekdays, after_date

# Business hours in New York, regardless of server timezone
nyse_hours = during_hours(9, 16, tz="America/New_York") & on_weekdays(tz="America/New_York")

# Feature launch date in Tokyo time
launched_jp = after_date(2024, 6, 1, tz="Asia/Tokyo")
```

This uses `zoneinfo.ZoneInfo` from the standard library (Python 3.9+).

## Composition

Combine temporal rules with other rules:

```python
# Trading only during business hours on weekdays
can_trade = is_active & during_hours(9, 16) & on_weekdays()

# Premium users get extended hours
can_trade_premium = is_premium & during_hours(7, 20) & on_weekdays()
```
