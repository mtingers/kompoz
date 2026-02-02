# Temporal Combinators

## during_hours

```python
during_hours(
    start: int,
    end: int,
    tz: tzinfo | None = None,
    inclusive_end: bool = False,
) -> Predicate
```

Check if the current hour is within the given range.

- `start` / `end` -- Hours (0-23)
- `tz` -- Optional timezone (defaults to local time)
- `inclusive_end` -- If `True`, include the end hour

Supports overnight ranges (e.g. `during_hours(22, 6)` for 10 PM to 5:59 AM).

```python
from kompoz import during_hours

business_hours = during_hours(9, 17)
night_mode = during_hours(22, 6)
full_hours = during_hours(9, 17, inclusive_end=True)
```

---

## on_weekdays

```python
on_weekdays() -> Predicate
```

Check if today is Monday through Friday.

```python
from kompoz import on_weekdays

weekdays = on_weekdays()
```

---

## on_days

```python
on_days(*days: int) -> Predicate
```

Check if today is one of the specified days. Days use Monday=0, Sunday=6.

```python
from kompoz import on_days

mwf = on_days(0, 2, 4)      # Mon, Wed, Fri
weekends = on_days(5, 6)     # Sat, Sun
```

---

## after_date

```python
after_date(year: int, month: int, day: int) -> Predicate
```

Check if today is on or after the given date.

```python
from kompoz import after_date

launched = after_date(2024, 6, 1)
```

---

## before_date

```python
before_date(year: int, month: int, day: int) -> Predicate
```

Check if today is on or before the given date.

```python
from kompoz import before_date

promo_active = before_date(2024, 12, 31)
```

---

## between_dates

```python
between_dates(start: date, end: date) -> Predicate
```

Check if today is within the date range (inclusive).

```python
from kompoz import between_dates
from datetime import date

q1_only = between_dates(date(2024, 1, 1), date(2024, 3, 31))
```
