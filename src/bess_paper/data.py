"""PSE data through the `pse-opendata` client: day-ahead prices and hourly system data.

Prices are returned per quarter-hour (one row per 15-minute delivery interval, 92/96/100 rows per day): from
2024-06-14 the SDAC price of the Polish bidding zone (`csdac-pln`), which repeats the hourly price in four
quarter-hours until 2025-09-30. Before 2024-06-14 (only used to train the first backtest models) the hourly RCE of
the legacy export is spread over its four quarter-hours the same way.

Usage:
    client = make_client(Path("data"))
    prices = load_prices(client, date(2024, 6, 14), date(2026, 10, 27))
    system = load_system(client, date(2024, 6, 14), date(2026, 10, 26))
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pse_opendata as pse

from bess_paper.config import FIRST_DAY, PRICE_REPORT, TZ

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["start_utc", "business_date", "price"]
SYSTEM_FIELDS = ("wind", "pv", "demand")


def make_client(data_dir: Path | None, offline: bool = False) -> pse.Client:
    """Client with its monthly cache in ``<data_dir>/pse`` (no cache when ``data_dir`` is None)."""
    cache = Path(data_dir) / "pse" if data_dir is not None else None
    return pse.Client(cache_dir=cache, offline=offline)


def _month_end(day: date) -> date:
    """Requests run to the end of the month, so the client caches the current month too (and re-checks it later)."""
    return (pd.Timestamp(day) + pd.offsets.MonthEnd(0)).date()


def _quarters(hourly: pd.DataFrame, column: str) -> pd.DataFrame:
    """Spread hourly rows over their four quarter-hours (same value, same business date)."""
    parts = []
    for k in range(4):
        part = hourly[["start_utc", "business_date", column]].copy()
        part["start_utc"] = part["start_utc"] + pd.Timedelta(minutes=15 * k)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values("start_utc", ignore_index=True)


def load_prices(client: pse.Client, start: date, end: date) -> pd.DataFrame:
    """Quarter-hourly day-ahead prices (PLN/MWh) for delivery days ``start``..``end``, sorted by ``start_utc``."""
    frames = []
    if start < FIRST_DAY:
        legacy = client.history("rce", start, min(end, FIRST_DAY - timedelta(days=1)))
        legacy = legacy[legacy["source"] == "legacy"].rename(columns={"rce": "price"})
        frames.append(_quarters(legacy, "price"))
    if end >= FIRST_DAY:
        api = client.get(PRICE_REPORT, max(start, FIRST_DAY), _month_end(end), dropna=True)
        api = api[api["business_date"] <= end]
        frames.append(api.rename(columns={"csdac_pln": "price"})[PRICE_COLUMNS])
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)
    out = out.dropna(subset=["price"]).drop_duplicates("start_utc").sort_values("start_utc", ignore_index=True)
    out["start_utc"] = pd.to_datetime(out["start_utc"], utc=True).dt.as_unit("ns")
    return out[PRICE_COLUMNS]


def load_system(client: pse.Client, start: date, end: date) -> pd.DataFrame:
    """Hourly wind, PV and demand (MW), indexed by the hour start in UTC (legacy export before 2024-06-14)."""
    parts = []
    if start < FIRST_DAY:
        old_end = min(end, FIRST_DAY - timedelta(days=1))
        series = [client.history(name, start, old_end).set_index("start_utc")[name] for name in SYSTEM_FIELDS]
        parts.append(pd.concat(series, axis=1))
    if end >= FIRST_DAY:
        api = client.get("his-wlk-cal", max(start, FIRST_DAY), _month_end(end), fields=["wi", "pv", "demand"])
        api = api[api["business_date"] <= end]
        hourly = pse.to_hourly(api).rename(columns={"wi": "wind"}).set_index("start_utc")
        parts.append(hourly[list(SYSTEM_FIELDS)])
    if not parts:
        return pd.DataFrame(columns=list(SYSTEM_FIELDS), index=pd.DatetimeIndex([], tz="UTC", name="start_utc"))
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True)).as_unit("ns")
    out.index.name = "start_utc"
    return out


def complete_days(prices: pd.DataFrame) -> list[date]:
    """Delivery days with a price for every quarter-hour of the local day (92, 96 or 100 intervals)."""
    counts = prices.groupby("business_date")["price"].count()
    return [d for d, n in counts.items() if n == expected_intervals(d)]


def expected_intervals(day: date) -> int:
    """Quarter-hours in a Polish local day: 92 on the spring DST day, 100 on the autumn one, else 96."""
    start = pd.Timestamp(day).tz_localize(TZ)
    end = pd.Timestamp(day + timedelta(days=1)).tz_localize(TZ)
    return int((end - start) / pd.Timedelta(minutes=15))


def day_intervals(day: date) -> pd.DatetimeIndex:
    """Start times (UTC) of the quarter-hours of a Polish local day."""
    start = pd.Timestamp(day).tz_localize(TZ).tz_convert("UTC")
    return pd.date_range(start, periods=expected_intervals(day), freq="15min").as_unit("ns")
