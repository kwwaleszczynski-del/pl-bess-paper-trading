"""No look-ahead: the inputs of day D must not change when anything after the decision on D-1 changes."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess_paper.config import FUND_CUTOFF_LOCAL
from bess_paper.data import complete_days, expected_intervals
from bess_paper.features import FEATURES, build_features, lagged_prices
from bess_paper.forecast import train_mask
from bess_paper.live import as_of, local_time

from conftest import synthetic_prices, synthetic_system

START, END = date(2025, 9, 1), date(2025, 11, 30)


@pytest.fixture(scope="module")
def data():
    return synthetic_prices(START, END), synthetic_system(START, END)


def scramble_after(prices, system, day):
    """Replace every price of day >= ``day`` and every system value after 06:00 on day-1 by noise."""
    rng = np.random.default_rng(99)
    p = prices.copy()
    future = p["business_date"] >= day
    p.loc[future, "price"] = rng.normal(0, 1000, future.sum())
    s = system.copy()
    cutoff = pd.Timestamp(local_time(day - timedelta(days=1), FUND_CUTOFF_LOCAL))
    late = (s.index + pd.Timedelta(hours=1)) > cutoff
    s.loc[late, :] = rng.normal(0, 1e5, (late.sum(), s.shape[1]))
    return p, s


@pytest.mark.parametrize("day", [date(2025, 9, 20), date(2025, 10, 1), date(2025, 10, 26), date(2025, 11, 3)])
def test_features_ignore_everything_after_the_decision(data, day):
    prices, system = data
    before = build_features(prices, system, [day])
    after = build_features(*scramble_after(prices, system, day), [day])
    assert np.allclose(before[FEATURES].to_numpy(float), after[FEATURES].to_numpy(float), equal_nan=True)


def test_features_do_use_the_previous_day(data):
    prices, system = data
    day = date(2025, 10, 15)
    changed = prices.copy()
    changed.loc[changed["business_date"] == day - timedelta(days=1), "price"] += 100.0
    a = build_features(prices, system, [day])
    b = build_features(changed, system, [day])
    assert np.allclose(b["lag1"] - a["lag1"], 100.0)
    assert np.allclose(b["d1_mean"] - a["d1_mean"], 100.0)


def test_system_data_of_the_morning_are_used_up_to_the_cutoff_only(data):
    prices, system = data
    day = date(2025, 10, 15)
    s = system.copy()
    cutoff = pd.Timestamp(local_time(day - timedelta(days=1), FUND_CUTOFF_LOCAL))
    last_hour = s.index == cutoff - pd.Timedelta(hours=1)          # 05:00-06:00 on D-1: allowed
    s.loc[last_hour, "wind"] += 1000.0
    a, b = build_features(prices, system, [day]), build_features(prices, s, [day])
    assert (b["wind_d1_last"] - a["wind_d1_last"]).abs().min() == pytest.approx(1000.0)


def test_rows_per_day_follow_daylight_saving_time(data):
    prices, system = data
    for day, n in [(date(2025, 10, 26), 100), (date(2025, 10, 27), 96)]:
        assert len(build_features(prices, system, [day])) == n == expected_intervals(day)
    assert expected_intervals(date(2026, 3, 29)) == 92


def test_yesterday_signal_equals_lag_feature(data):
    prices, system = data
    for day in [date(2025, 10, 1), date(2025, 10, 26), date(2025, 10, 27), date(2025, 11, 10)]:
        lag1 = build_features(prices, system, [day])["lag1"].to_numpy()
        assert np.allclose(lagged_prices(prices, day), lag1)


def test_training_rows_end_before_the_first_forecast_day(data, fast_model):
    prices, system = data
    days = complete_days(prices)
    X = build_features(prices, system, days)
    y = prices.set_index("start_utc")["price"].reindex(pd.DatetimeIndex(X["start_utc"])).to_numpy()
    first = date(2025, 11, 10)
    mask = train_mask(X, y, first, fast_model)
    used = set(X.loc[mask, "day"])
    assert max(used) == first - timedelta(days=1)
    assert min(used) >= first - timedelta(days=fast_model.train_days)


def test_live_as_of_view_drops_future_data(data):
    prices, system = data
    day = date(2025, 11, 10)
    p, s = as_of(prices, system, day)
    assert p["business_date"].max() == day - timedelta(days=1)
    cutoff = pd.Timestamp(local_time(day - timedelta(days=1), FUND_CUTOFF_LOCAL))
    assert ((s.index + pd.Timedelta(hours=1)) <= cutoff).all()
    assert (s.index + pd.Timedelta(hours=1) == cutoff).any()
