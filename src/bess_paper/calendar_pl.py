"""Polish public holidays (statutory days off), known years in advance.

Usage:
    from bess_paper.calendar_pl import is_holiday
    is_holiday(date(2025, 12, 24))   # True: Christmas Eve is a day off from 2025
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def easter(year: int) -> date:
    """Easter Sunday (Gregorian calendar, anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return date(year, month, day)


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset[date]:
    e = easter(year)
    days = {
        date(year, 1, 1), date(year, 1, 6), e, e + timedelta(days=1), date(year, 5, 1), date(year, 5, 3),
        e + timedelta(days=49), e + timedelta(days=60), date(year, 8, 15), date(year, 11, 1), date(year, 11, 11),
        date(year, 12, 25), date(year, 12, 26),
    }
    if year >= 2025:
        days.add(date(year, 12, 24))    # Christmas Eve, a statutory day off from 2025
    return frozenset(days)


def is_holiday(day: date) -> bool:
    return day in holidays(day.year)
