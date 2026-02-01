"""Tests for temporal combinators: during_hours, on_weekdays, on_days, after_date, before_date, between_dates."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch

import pytest

from kompoz import (
    after_date,
    before_date,
    between_dates,
    during_hours,
    on_days,
    on_weekdays,
)


class TestDuringHours:
    def test_within_range(self):
        h = during_hours(9, 17)
        with patch.object(h, "_get_current_hour", return_value=12):
            ok, _ = h.run(None)
            assert ok

    def test_before_range(self):
        h = during_hours(9, 17)
        with patch.object(h, "_get_current_hour", return_value=8):
            ok, _ = h.run(None)
            assert not ok

    def test_at_start(self):
        h = during_hours(9, 17)
        with patch.object(h, "_get_current_hour", return_value=9):
            ok, _ = h.run(None)
            assert ok

    def test_at_end_exclusive(self):
        h = during_hours(9, 17)
        with patch.object(h, "_get_current_hour", return_value=17):
            ok, _ = h.run(None)
            assert not ok

    def test_at_end_inclusive(self):
        h = during_hours(9, 17, inclusive_end=True)
        with patch.object(h, "_get_current_hour", return_value=17):
            ok, _ = h.run(None)
            assert ok

    def test_overnight_range(self):
        h = during_hours(22, 6)
        with patch.object(h, "_get_current_hour", return_value=23):
            ok, _ = h.run(None)
            assert ok
        with patch.object(h, "_get_current_hour", return_value=3):
            ok, _ = h.run(None)
            assert ok
        with patch.object(h, "_get_current_hour", return_value=12):
            ok, _ = h.run(None)
            assert not ok

    def test_overnight_inclusive_end(self):
        h = during_hours(22, 6, inclusive_end=True)
        with patch.object(h, "_get_current_hour", return_value=6):
            ok, _ = h.run(None)
            assert ok

    def test_overnight_exclusive_end(self):
        h = during_hours(22, 6)
        with patch.object(h, "_get_current_hour", return_value=6):
            ok, _ = h.run(None)
            assert not ok

    def test_invalid_hours(self):
        with pytest.raises(ValueError, match="Hours must be 0-23"):
            during_hours(-1, 17)
        with pytest.raises(ValueError, match="Hours must be 0-23"):
            during_hours(9, 25)

    def test_repr(self):
        assert repr(during_hours(9, 17)) == "during_hours(9, 17)"
        assert "inclusive_end=True" in repr(during_hours(9, 17, inclusive_end=True))

    def test_context_passthrough(self):
        h = during_hours(0, 23)
        with patch.object(h, "_get_current_hour", return_value=12):
            _, ctx = h.run("my_context")
            assert ctx == "my_context"


class TestOnWeekdays:
    def test_weekday(self):
        w = on_weekdays()
        # Monday = 0
        with patch("kompoz._temporal.datetime") as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 0
            ok, _ = w.run(None)
            assert ok

    def test_weekend(self):
        w = on_weekdays()
        # Saturday = 5
        with patch("kompoz._temporal.datetime") as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 5
            ok, _ = w.run(None)
            assert not ok

    def test_repr(self):
        assert repr(on_weekdays()) == "on_weekdays()"


class TestOnDays:
    def test_matching_day(self):
        d = on_days(0, 2, 4)  # Mon, Wed, Fri
        with patch("kompoz._temporal.datetime") as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 2  # Wednesday
            ok, _ = d.run(None)
            assert ok

    def test_non_matching_day(self):
        d = on_days(0, 2, 4)
        with patch("kompoz._temporal.datetime") as mock_dt:
            mock_dt.now.return_value.weekday.return_value = 1  # Tuesday
            ok, _ = d.run(None)
            assert not ok

    def test_invalid_day(self):
        with pytest.raises(ValueError, match="Days must be 0-6"):
            on_days(7)

    def test_repr(self):
        assert repr(on_days(0, 2)) == "on_days(0, 2)"


class TestAfterDate:
    def test_after(self):
        a = after_date(2020, 1, 1)
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2025, 6, 1)
            ok, _ = a.run(None)
            assert ok

    def test_before(self):
        a = after_date(2030, 1, 1)
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2025, 6, 1)
            ok, _ = a.run(None)
            assert not ok

    def test_same_day(self):
        a = after_date(2025, 6, 1)
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2025, 6, 1)
            ok, _ = a.run(None)
            assert not ok  # after, not on

    def test_with_date_object(self):
        a = after_date(date(2020, 1, 1))
        assert a.date == date(2020, 1, 1)

    def test_missing_args(self):
        with pytest.raises(ValueError):
            after_date(2020, 1)

    def test_repr(self):
        assert "2020-01-01" in repr(after_date(2020, 1, 1))


class TestBeforeDate:
    def test_before(self):
        b = before_date(2030, 1, 1)
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2025, 6, 1)
            ok, _ = b.run(None)
            assert ok

    def test_after(self):
        b = before_date(2020, 1, 1)
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2025, 6, 1)
            ok, _ = b.run(None)
            assert not ok

    def test_same_day(self):
        b = before_date(2025, 6, 1)
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2025, 6, 1)
            ok, _ = b.run(None)
            assert not ok  # before, not on

    def test_with_date_object(self):
        b = before_date(date(2030, 12, 31))
        assert b.date == date(2030, 12, 31)

    def test_repr(self):
        assert "2030-12-31" in repr(before_date(2030, 12, 31))


class TestBetweenDates:
    def test_within_range(self):
        b = between_dates(date(2025, 1, 1), date(2025, 12, 31))
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2025, 6, 15)
            ok, _ = b.run(None)
            assert ok

    def test_before_range(self):
        b = between_dates(date(2025, 1, 1), date(2025, 12, 31))
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2024, 12, 31)
            ok, _ = b.run(None)
            assert not ok

    def test_after_range(self):
        b = between_dates(date(2025, 1, 1), date(2025, 12, 31))
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2026, 1, 1)
            ok, _ = b.run(None)
            assert not ok

    def test_on_start_boundary(self):
        b = between_dates(date(2025, 6, 1), date(2025, 6, 30))
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2025, 6, 1)
            ok, _ = b.run(None)
            assert ok  # inclusive

    def test_on_end_boundary(self):
        b = between_dates(date(2025, 6, 1), date(2025, 6, 30))
        with patch("kompoz._temporal.date") as mock_date:
            mock_date.today.return_value = date(2025, 6, 30)
            ok, _ = b.run(None)
            assert ok  # inclusive

    def test_integer_constructor(self):
        b = between_dates(2025, 1, 1, 2025, 12, 31)
        assert b.start_date == date(2025, 1, 1)
        assert b.end_date == date(2025, 12, 31)

    def test_integer_constructor_missing_args(self):
        with pytest.raises(ValueError):
            between_dates(2025, 1, 1, 2025, 12)

    def test_mixed_types_error(self):
        with pytest.raises(TypeError):
            between_dates(date(2025, 1, 1), 12)

    def test_repr(self):
        b = between_dates(date(2025, 1, 1), date(2025, 12, 31))
        assert "2025-01-01" in repr(b)
        assert "2025-12-31" in repr(b)
