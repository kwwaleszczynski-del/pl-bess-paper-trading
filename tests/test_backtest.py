"""Walk-forward backtest on synthetic data: models see only the past, results are consistent."""
from datetime import date, timedelta

import numpy as np
import pytest

import bess_paper.backtest as bt
from bess_paper.backtest import run_backtest, summary

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
