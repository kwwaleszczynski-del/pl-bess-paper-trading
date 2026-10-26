"""Trading strategies of one delivery day and their settlement on the clearing prices.

All strategies trade the same battery and are settled the same way; they differ only in the price signal the
schedule is optimised on:

- ``model``: median of the quantile gradient boosting forecast (the live strategy);
- ``yesterday``: the prices of D-1 at the same local time, known at the decision;
- ``fixed``: no price signal, the same hours every day (:data:`FIXED_WINDOWS`);
- ``perfect``: the clearing prices of D themselves (perfect foresight, an upper bound no one can trade).

Until 2025-09-30 the SDAC products were hourly: on those days the price signal is averaged to hours, the market
position is the hourly net energy, and the battery may still vary its power within the hour.

Usage:
    plan = plan_day(signal, day)
    result = settle_day(plan, actual, day)
"""
from __future__ import annotations

from datetime import date

import numpy as np

from bess_paper.config import BATTERY, BatteryParams
from bess_paper.features import minute_of_day
from bess_paper.data import day_intervals
from bess_paper.forecast import to_product
from bess_paper.optimizer import Plan, optimize, settle, soc_path

STRATEGIES = ("model", "perfect", "yesterday", "fixed")

# Same local hours every day: sell into the morning peak, buy around noon, sell into the evening peak. The hours
# are those of the optimal schedule for the average price profile of 1.06.2023-13.06.2024 (legacy RCE, all before
# the first backtest day), without its small extra night cycle.
FIXED_WINDOWS = (
    (7 * 60, 9 * 60, "discharge"),
    (12 * 60, 14 * 60, "charge"),
    (19 * 60, 21 * 60, "discharge"),
)


def plan_day(signal, day: date, params: BatteryParams = BATTERY) -> Plan:
    """Optimal quarter-hourly schedule for a price signal, averaged to hourly products where they applied."""
    return optimize(to_product(signal, day), 0.25, params)


def fixed_plan(day: date, params: BatteryParams = BATTERY, windows=FIXED_WINDOWS) -> Plan:
    """Rule-based schedule: full power in each window until the state-of-charge limit.

    The last discharge window stops at the start level, so the day ends where it began.
    """
    dt = 0.25
    minutes = minute_of_day(day_intervals(day))
    e_min, e_max = params.energy_mwh * params.soc_min, params.energy_mwh * params.soc_max
    e_end = params.energy_mwh * params.soc_start
    last_discharge = max(i for i, w in enumerate(windows) if w[2] == "discharge")
    soc = e_end
    charge, discharge = np.zeros(len(minutes)), np.zeros(len(minutes))
    for t, m in enumerate(minutes):
        for i, (lo, hi, side) in enumerate(windows):
            if not lo <= m < hi:
                continue
            if side == "charge":
                charge[t] = min(params.power_mw, max(e_max - soc, 0.0) / (params.eta * dt))
            else:
                floor = e_end if i == last_discharge else e_min
                discharge[t] = min(params.power_mw, max(soc - floor, 0.0) * params.eta / dt)
        soc += params.eta * dt * charge[t] - dt * discharge[t] / params.eta
    return Plan(charge, discharge, soc_path(charge, discharge, dt, params))


def settle_day(plan: Plan, actual, params: BatteryParams = BATTERY) -> dict:
    """P&L of a quarter-hourly plan at the clearing prices of the day."""
    return settle(plan, actual, 0.25, params)


def run_day(day: date, actual, signals: dict[str, np.ndarray], params: BatteryParams = BATTERY) -> dict[str, dict]:
    """Plan and settle every strategy whose signal is available; a strategy without a signal does not trade."""
    out = {}
    for name in STRATEGIES:
        if name == "fixed":
            plan = fixed_plan(day, params)
        else:
            signal = actual if name == "perfect" else signals.get(name)
            if signal is None or not np.isfinite(signal).all():
                n = len(actual)
                plan = Plan(np.zeros(n), np.zeros(n), soc_path(np.zeros(n), np.zeros(n), 0.25, params))
            else:
                plan = plan_day(signal, day, params)
        out[name] = {"plan": plan, **settle_day(plan, actual, params)}
    return out
