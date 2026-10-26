from datetime import date

import numpy as np
import pytest

from bess_paper.config import BATTERY, BatteryParams
from bess_paper.optimizer import Plan, check_plan, optimize, settle, soc_path
from bess_paper.strategies import fixed_plan, plan_day, run_day

from conftest import synthetic_prices


@pytest.mark.parametrize("n,dt", [(96, 0.25), (92, 0.25), (100, 0.25), (24, 1.0), (23, 1.0), (25, 1.0)])
def test_schedule_respects_energy_balance_and_limits(n, dt):
    rng = np.random.default_rng(n)
    prices = 300 + 200 * rng.standard_normal(n)
    plan = optimize(prices, dt)
    check_plan(plan, dt)
    # state of charge follows the efficiency-weighted energy balance step by step
    soc = BATTERY.energy_mwh * BATTERY.soc_start
    for c, d, s in zip(plan.charge_mw, plan.discharge_mw, plan.soc_mwh):
        soc += BATTERY.eta * dt * c - dt * d / BATTERY.eta
        assert s == pytest.approx(soc, abs=1e-9)
    assert plan.soc_mwh[-1] == pytest.approx(BATTERY.energy_mwh * BATTERY.soc_start, abs=1e-6)


def test_flat_prices_do_not_trade():
    plan = optimize(np.full(96, 400.0))
    assert plan.charge_mw.sum() == 0 and plan.discharge_mw.sum() == 0


def test_buys_low_and_sells_high():
    prices = np.r_[np.full(48, 100.0), np.full(48, 600.0)]
    plan = optimize(prices)
    assert plan.charge_mw[:48].sum() > 0 and plan.discharge_mw[48:].sum() > 0
    assert plan.discharge_mw[:48].sum() == pytest.approx(0, abs=1e-6)
    assert settle(plan, prices)["pnl_pln"] > 0


def test_negative_prices_never_charge_and_discharge_at_once():
    rng = np.random.default_rng(3)
    prices = np.r_[-300 + 50 * rng.standard_normal(40), 200 + 50 * rng.standard_normal(56)]
    plan = optimize(prices)
    assert not ((plan.charge_mw > 1e-9) & (plan.discharge_mw > 1e-9)).any()
    check_plan(plan)


def test_settlement_by_hand():
    # charge 1 MW for one hour at 100 PLN/MWh, sell 0.8 MW for one hour at 300 PLN/MWh (hourly intervals)
    params = BatteryParams(degradation_cost=30.0)
    c = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, 0.8, 0.0])
    plan = Plan(c, d, soc_path(c, d, 1.0, params))
    res = settle(plan, np.array([100.0, 300.0, 50.0]), 1.0, params)
    assert res["gross_pln"] == pytest.approx(-100.0 + 240.0)
    assert res["pnl_pln"] == pytest.approx(140.0 - 30.0 * 0.8)
    assert res["bought_mwh"] == pytest.approx(1.0) and res["sold_mwh"] == pytest.approx(0.8)
    assert res["cycles"] == pytest.approx(0.8 / params.usable_mwh)


def test_settlement_rejects_length_mismatch():
    plan = optimize(np.linspace(100, 500, 96))
    with pytest.raises(ValueError):
        settle(plan, np.ones(92))


def test_check_plan_detects_broken_balance():
    plan = optimize(np.linspace(100, 500, 96))
    broken = Plan(plan.charge_mw, plan.discharge_mw, plan.soc_mwh + 0.01)
    with pytest.raises(ValueError):
        check_plan(broken)


@pytest.mark.parametrize("day", [date(2025, 3, 30), date(2025, 6, 2), date(2025, 10, 26), date(2026, 3, 29),
                                 date(2026, 6, 2), date(2026, 10, 25)])
def test_fixed_plan_is_feasible_and_ends_at_start(day):
    plan = fixed_plan(day)
    check_plan(plan)
    assert plan.discharge_mw.sum() > 0 and plan.charge_mw.sum() > 0


def test_hourly_products_use_the_hourly_signal():
    day = date(2025, 6, 2)                       # before 2025-10-01: hourly SDAC products
    hourly = synthetic_prices(day, day)["price"].to_numpy()
    wiggle = hourly + np.tile([0, 40, -40, 0], 24)   # sub-hourly detail that hourly products cannot trade
    plan = plan_day(wiggle, day)
    check_plan(plan)
    assert np.allclose(plan.net_mw, plan_day(hourly, day).net_mw)
    # the settlement of a quarter-hourly plan at hourly prices equals the hourly net energy times the price
    net_hourly = plan.net_mw.reshape(-1, 4).sum(axis=1) * 0.25
    gross = float(np.sum(hourly.reshape(-1, 4)[:, 0] * net_hourly))
    assert settle(plan, hourly)["gross_pln"] == pytest.approx(gross)


def test_perfect_foresight_is_an_upper_bound():
    day = date(2026, 5, 4)
    actual = synthetic_prices(day, day, seed=5)["price"].to_numpy()
    noisy = actual + np.random.default_rng(1).normal(0, 80, len(actual))
    res = run_day(day, actual, {"model": noisy, "yesterday": np.roll(actual, 8)})
    for name in ("model", "yesterday", "fixed"):
        assert res["perfect"]["pnl_pln"] >= res[name]["pnl_pln"] - 1e-6


def test_strategy_without_signal_does_not_trade():
    day = date(2026, 5, 4)
    actual = synthetic_prices(day, day)["price"].to_numpy()
    res = run_day(day, actual, {"model": np.full(len(actual), np.nan)})
    assert res["model"]["pnl_pln"] == 0 and res["yesterday"]["pnl_pln"] == 0
