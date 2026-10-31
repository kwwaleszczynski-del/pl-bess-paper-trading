"""As-of features for the price forecast of delivery day D, made on D-1 before the SDAC gate closure.

Information set at the decision (D-1, published by about 07:00 Polish time, five hours before the 12:00 gate; the
plan is issued by the first scheduled run that starts between then and the gate):
- day-ahead prices of delivery days up to D-1 (published around 14:00 on D-2, not revised afterwards);
- hourly wind, PV and demand of PSE `his-wlk-cal` for hours that end by 06:00 on D-1;
- the calendar of D (weekday, public holidays, season).

Every feature of day D is looked up by date arithmetic from days D-1, D-2 and D-7 only, so building the features
of all days at once uses no information from after the decision of each day. The test suite checks this by
changing all data after the cut-off and comparing the features.

Usage:
    X = build_features(prices, system, days)       # one row per quarter-hour of each day in ``days``
    y = target(prices, X)
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from bess_paper.calendar_pl import is_holiday
from bess_paper.config import FUND_CUTOFF_LOCAL, QH_FROM, TZ
from bess_paper.data import day_intervals

MINUTES = np.arange(0, 24 * 60, 15)
FEATURES = [
    "minute", "dow", "hol", "hol_lag1", "doy_sin", "doy_cos", "qh_product",
    "lag1", "lag2", "lag7", "lag_week",
    "d1_mean", "d1_min", "d1_max", "d1_std", "d1_night", "d1_mid", "d1_eve", "d1_neg", "d7_mean",
    "wind_d2", "pv_d2", "demand_d2", "wind_d1_night", "demand_d1_night", "wind_d1_last",
]


def minute_of_day(start_utc) -> np.ndarray:
    local = pd.DatetimeIndex(start_utc).tz_convert(TZ)
    return np.asarray(local.hour * 60 + local.minute)


def price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Prices as delivery day x local minute of day (0, 15, ..., 1425).

    The repeated hour of the autumn DST day is averaged; the missing hour of the spring DST day is interpolated
    from its neighbours within the same day.
    """
    frame = pd.DataFrame({"day": prices["business_date"].to_numpy(), "minute": minute_of_day(prices["start_utc"]),
                          "price": prices["price"].to_numpy(dtype=float)})
    m = frame.pivot_table(index="day", columns="minute", values="price", aggfunc="mean").reindex(columns=MINUTES)
    return m.T.interpolate(limit_direction="both").T


def _window(m: pd.DataFrame, lo_h: int, hi_h: int) -> pd.Series:
    return m.loc[:, (m.columns >= lo_h * 60) & (m.columns < hi_h * 60)].mean(axis=1)


def day_stats(m: pd.DataFrame) -> pd.DataFrame:
    """Shape of each delivery day's price curve."""
    return pd.DataFrame({
        "mean": m.mean(axis=1), "min": m.min(axis=1), "max": m.max(axis=1), "std": m.std(axis=1),
        "night": _window(m, 0, 6), "mid": _window(m, 10, 15), "eve": _window(m, 17, 21),
        "neg": (m < 0).mean(axis=1),
    })


def system_stats(system: pd.DataFrame) -> pd.DataFrame:
    """Daily system statistics per local date: whole-day means and the hours before the morning cut-off."""
    local = pd.DatetimeIndex(system.index).tz_convert(TZ)
    frame = system.assign(day=local.date, hour=local.hour)
    cutoff = FUND_CUTOFF_LOCAL.hour
    early = frame[frame["hour"] < cutoff]
    last = frame[frame["hour"] == cutoff - 1]
    return pd.DataFrame({
        "wind": frame.groupby("day")["wind"].mean(),
        "pv": frame.groupby("day")["pv"].mean(),
        "demand": frame.groupby("day")["demand"].mean(),
        "wind_night": early.groupby("day")["wind"].mean(),
        "demand_night": early.groupby("day")["demand"].mean(),
        "wind_last": last.groupby("day")["wind"].mean(),
    })


def _shift(days, k: int) -> list[date]:
    return [d - timedelta(days=k) for d in days]


def build_features(prices: pd.DataFrame, system: pd.DataFrame, days: list[date]) -> pd.DataFrame:
    """Feature rows for every quarter-hour of the delivery days ``days``.

    Columns: ``day``, ``start_utc`` and :data:`FEATURES`. Missing inputs give NaN (the model handles them).
    """
    rows = pd.DataFrame({"start_utc": pd.DatetimeIndex(np.concatenate([day_intervals(d) for d in days]))
                         if days else pd.DatetimeIndex([], tz="UTC")})
    rows["day"] = [d for d in days for _ in range(len(day_intervals(d)))]
    rows["minute"] = minute_of_day(rows["start_utc"])
    day = rows["day"]

    m = price_matrix(prices)
    long = m.stack()

    def lag(k: int) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([_shift(day, k), rows["minute"]])
        return long.reindex(idx).to_numpy()

    for k in (1, 2, 7):
        rows[f"lag{k}"] = lag(k)
    rows["lag_week"] = pd.DataFrame({k: lag(k) for k in range(1, 8)}).mean(axis=1).to_numpy()

    stats = day_stats(m)
    d1 = stats.reindex(_shift(day, 1)).to_numpy()
    for j, name in enumerate(stats.columns):
        rows[f"d1_{name}"] = d1[:, j]
    rows["d7_mean"] = stats["mean"].reindex(_shift(day, 7)).to_numpy()

    sys = system_stats(system)
    d2 = sys.reindex(_shift(day, 2))
    rows["wind_d2"], rows["pv_d2"], rows["demand_d2"] = d2["wind"].to_numpy(), d2["pv"].to_numpy(), \
        d2["demand"].to_numpy()
    s1 = sys.reindex(_shift(day, 1))
    rows["wind_d1_night"] = s1["wind_night"].to_numpy()
    rows["demand_d1_night"] = s1["demand_night"].to_numpy()
    rows["wind_d1_last"] = s1["wind_last"].to_numpy()

    ts = pd.to_datetime(pd.Series(day))
    rows["dow"] = ts.dt.day_of_week.to_numpy()
    rows["hol"] = np.array([is_holiday(d) for d in day], dtype=float)
    rows["hol_lag1"] = np.array([is_holiday(d) for d in _shift(day, 1)], dtype=float)
    doy = ts.dt.day_of_year.to_numpy()
    rows["doy_sin"], rows["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    rows["qh_product"] = np.array([d >= QH_FROM for d in day], dtype=float)
    return rows[["day", "start_utc", *FEATURES]]


def lagged_prices(prices: pd.DataFrame, day: date, k: int = 1) -> np.ndarray:
    """Prices of D-k at the local times of the quarter-hours of D (for k=1 the ``yesterday`` forecast)."""
    prev = day - timedelta(days=k)
    minutes = minute_of_day(day_intervals(day))
    m = price_matrix(prices[prices["business_date"] == prev])
    if prev not in m.index:
        return np.full(len(minutes), np.nan)
    return m.loc[prev].reindex(minutes).to_numpy(dtype=float)


def target(prices: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Actual price of each feature row (NaN where not published)."""
    by_time = prices.set_index("start_utc")["price"]
    return by_time.reindex(pd.DatetimeIndex(rows["start_utc"])).to_numpy(dtype=float)
