"""Walk-forward backtest on synthetic data: models see only the past, results are consistent."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import bess_paper.backtest as bt
from bess_paper.backtest import comparisons, run_backtest, summary
from bess_paper.config import FUND_CUTOFF_LOCAL
from bess_paper.live import local_time

from conftest import synthetic_prices, synthetic_system


@pytest.fixture(scope="module")
def market():
    start, end = date(2025, 7, 1), date(2025, 10, 31)
    return synthetic_prices(start, end, seed=21), synthetic_system(start, end, seed=22)


def test_models_are_trained_on_earlier_days_only(market, fast_model, monkeypatch):
    prices, system = market
    seen = []
    real_fit = bt.fit

    def spy(X, y, params):
        seen.append(max(X["day"]))
        return real_fit(X, y, params)

    monkeypatch.setattr(bt, "fit", spy)
    daily, _ = run_backtest(prices, system, start=date(2025, 9, 22), end=date(2025, 10, 12), params=fast_model)
    starts = [date(2025, 9, 22) + timedelta(days=7 * i) for i in range(3)]
    assert seen == [s - timedelta(days=1) for s in starts]
    assert len(daily) == 21


def test_backtest_results_are_consistent(market, fast_model):
    prices, system = market
    daily, detail = run_backtest(prices, system, start=date(2025, 9, 25), end=date(2025, 10, 8), params=fast_model)
    for s in ("model", "yesterday", "fixed", "last_week", "week_mean"):
        assert (daily["pnl_pln_perfect"] >= daily[f"pnl_pln_{s}"] - 1e-6).all()
    # hourly products until 30.09.2025: the forecasts are hourly values
    sep = detail[detail["day"] == date(2025, 9, 30)]["p50"].to_numpy().reshape(-1, 4)
    assert np.allclose(sep, sep[:, :1])
    # the daily P&L equals the quarter-hourly schedule times the price
    one = detail[detail["day"] == date(2025, 10, 3)]
    pnl = (one["actual"] * one["net_model"] * 0.25).sum() - 30.0 * (one["net_model"].clip(lower=0) * 0.25).sum()
    assert daily.loc[date(2025, 10, 3), "pnl_pln_model"] == pytest.approx(pnl)
    s = summary(daily)
    assert s["days"] == 14 and 0 < s["capture_model"] <= 1
    assert s["pnl_perfect"] == pytest.approx(daily["pnl_pln_perfect"].sum())


def test_backtest_forecast_and_plans_ignore_everything_after_the_decision(market, fast_model):
    prices, system = market
    day = date(2025, 10, 8)
    _, a = run_backtest(prices, system, start=day, end=day, params=fast_model)
    rng = np.random.default_rng(5)
    p2 = prices.copy()
    future = p2["business_date"] >= day
    p2.loc[future, "price"] = rng.normal(0, 1000, future.sum())
    s2 = system.copy()
    late = (s2.index + pd.Timedelta(hours=1)) > pd.Timestamp(local_time(day - timedelta(days=1), FUND_CUTOFF_LOCAL))
    s2.loc[late, :] = rng.normal(0, 1e5, (late.sum(), s2.shape[1]))
    _, b = run_backtest(p2, s2, start=day, end=day, params=fast_model)
    for col in ("p10", "p50", "p90", "yesterday", "net_model", "net_yesterday", "net_fixed"):
        assert np.allclose(a[col], b[col]), col
    assert not np.allclose(a["actual"], b["actual"])          # the changed prices do reach the settlement


def test_comparisons_of_strategies():
    days = pd.date_range("2025-03-01", "2025-05-15").strftime("%Y-%m-%d")
    n = len(days)
    daily = pd.DataFrame({"pnl_pln_model": np.full(n, 110.0), "pnl_pln_yesterday": np.full(n, 100.0),
                          "pnl_pln_week_mean": np.full(n, 120.0), "pnl_pln_last_week": np.full(n, 90.0),
                          "pnl_pln_fixed": np.full(n, 50.0), "pnl_pln_perfect": np.full(n, 200.0),
                          "mae_model": np.full(n, 80.0)}, index=pd.Index(days, name="date"))
    c = comparisons(daily, n_boot=200)
    assert (c["full_months"], c["partial_months"]) == (2, 1)
    y = c["model_vs"]["yesterday"]
    assert y["pln_per_mw_year"] == pytest.approx(10 * 365)
    lo, hi = y["ci95_pln_per_mw_year"]                    # partial first and last weeks make the interval wide
    assert lo < 10 * 365 < hi
    assert (y["share_days_model_better"], y["full_months_model_ahead"], y["partial_months_model_ahead"]) == (1, 2, 1)
    assert c["model_vs"]["week_mean"]["full_months_model_ahead"] == 0
    assert c["last_three_months"]["capture_model"] == pytest.approx(0.55)
