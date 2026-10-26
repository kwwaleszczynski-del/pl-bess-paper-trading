"""Synthetic market data for offline tests (no network, no PSE data)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess_paper.config import QH_FROM, ModelParams
from bess_paper.data import day_intervals
from bess_paper.features import minute_of_day


def days_between(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def synthetic_prices(start: date, end: date, seed: int = 0) -> pd.DataFrame:
    """Quarter-hourly prices with a daily shape, weekly pattern and noise; hourly products before 2025-10-01."""
    rng = np.random.default_rng(seed)
    frames = []
    for d in days_between(start, end):
        ts = day_intervals(d)
        hour = minute_of_day(ts) / 60.0
        level = 400 + 60 * rng.standard_normal() - (80 if d.weekday() >= 5 else 0)
        shape = -150 * np.exp(-((hour - 12.5) / 2.5) ** 2) + 180 * np.exp(-((hour - 19) / 1.8) ** 2) \
            + 60 * np.exp(-((hour - 7.5) / 1.2) ** 2)
        price = level + shape + 25 * rng.standard_normal(len(ts))
        if d < QH_FROM:
            price = np.repeat(price.reshape(-1, 4).mean(axis=1), 4)
        frames.append(pd.DataFrame({"start_utc": ts, "business_date": d, "price": price}))
    return pd.concat(frames, ignore_index=True)


def synthetic_system(start: date, end: date, seed: int = 1) -> pd.DataFrame:
    """Hourly wind, PV and demand (MW) indexed by the hour start in UTC."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start).tz_localize("Europe/Warsaw").tz_convert("UTC"),
                        pd.Timestamp(end + timedelta(days=1)).tz_localize("Europe/Warsaw").tz_convert("UTC"),
                        freq="h", inclusive="left").as_unit("ns")
    hour = idx.tz_convert("Europe/Warsaw").hour.to_numpy()
    wind = np.clip(4000 + np.cumsum(rng.standard_normal(len(idx))) * 300, 0, 9000)
    pv = np.clip(8000 * np.sin(np.pi * (hour - 6) / 14), 0, None)
    demand = 18000 + 3000 * np.sin(np.pi * (hour - 6) / 18) + 500 * rng.standard_normal(len(idx))
    return pd.DataFrame({"wind": wind, "pv": pv, "demand": demand}, index=pd.DatetimeIndex(idx, name="start_utc"))


@pytest.fixture
def fast_model() -> ModelParams:
    return ModelParams(train_days=60, retrain_every=7, max_iter=30, min_samples_leaf=20)
