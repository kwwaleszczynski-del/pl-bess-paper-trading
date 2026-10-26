"""Daily schedule of the battery as a mixed-integer linear program (HiGHS via scipy), and its settlement.

For one delivery day with prices p_t (PLN/MWh, interval length dt hours) the battery chooses charge c_t and
discharge d_t (MW) to maximise

    sum_t p_t * dt * (d_t - c_t)  -  deg_cost * sum_t dt * d_t

subject to
    soc_t = soc_{t-1} + eta * dt * c_t - dt * d_t / eta            (energy balance, MWh)
    soc_min <= soc_t <= soc_max,  soc_end = soc_start              (no free energy across days)
    0 <= c_t <= P * u_t,  0 <= d_t <= P * (1 - u_t),  u_t binary   (no simultaneous charge and discharge)
    sum_t dt * d_t <= max_cycles * usable_energy                   (throughput cap)

The prices are a forecast when the plan is made and the published clearing prices when it is settled; with the
actual prices the plan is the perfect-foresight upper bound.

Usage:
    plan = optimize(forecast_prices, dt_hours=0.25)
    pnl = settle(plan, actual_prices, dt_hours=0.25)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from bess_paper.config import BATTERY, BatteryParams


@dataclass
class Plan:
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray          # state of charge at the end of each interval

    @property
    def net_mw(self) -> np.ndarray:
        """Positive = selling (discharge), negative = buying (charge)."""
        return self.discharge_mw - self.charge_mw


def optimize(prices, dt_hours: float = 0.25, params: BatteryParams = BATTERY) -> Plan:
    """Solve the schedule for one day of prices (any number of intervals, e.g. 23-25 hours or 92-100 quarters)."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("prices must be a non-empty 1-d array of finite numbers")
    n = len(p)
    eta, P = params.eta, params.power_mw
    e_min, e_max = params.energy_mwh * params.soc_min, params.energy_mwh * params.soc_max
    soc0 = params.energy_mwh * params.soc_start

    # variable layout: [charge(n), discharge(n), soc(n), u(n)]
    ic, idd, isoc = 0, n, 2 * n
    nv = 4 * n
    cost = np.zeros(nv)
    cost[ic:ic + n] = p * dt_hours
    cost[idd:idd + n] = -(p - params.degradation_cost) * dt_hours

    eye = sparse.identity(n, format="csr")
    shift = sparse.eye(n, k=-1, format="csr")          # soc_{t-1}
    zeros = sparse.csr_matrix((n, n))
    # energy balance: soc_t - soc_{t-1} - eta*dt*c_t + dt/eta*d_t = (soc0 if t == 0 else 0)
    balance = sparse.hstack([-eta * dt_hours * eye, (dt_hours / eta) * eye, eye - shift, zeros])
    rhs = np.zeros(n)
    rhs[0] = soc0
    end_row = sparse.csr_matrix(([1.0], ([0], [isoc + n - 1])), shape=(1, nv))
    ch_mode = sparse.hstack([eye, zeros, zeros, -P * eye])          # c_t - P*u_t <= 0
    dis_mode = sparse.hstack([zeros, eye, zeros, P * eye])          # d_t + P*u_t <= P
    thr_row = sparse.csr_matrix((np.full(n, dt_hours), (np.zeros(n, int), np.arange(idd, idd + n))), shape=(1, nv))

    constraints = [
        LinearConstraint(balance, rhs, rhs),
        LinearConstraint(end_row, soc0, soc0),
        LinearConstraint(ch_mode, -np.inf, 0.0),
        LinearConstraint(dis_mode, -np.inf, P),
        LinearConstraint(thr_row, 0.0, params.max_cycles_per_day * params.usable_mwh),
    ]
    lb = np.concatenate([np.zeros(2 * n), np.full(n, e_min), np.zeros(n)])
    ub = np.concatenate([np.full(2 * n, P), np.full(n, e_max), np.ones(n)])
    integrality = np.concatenate([np.zeros(3 * n), np.ones(n)])
    res = milp(cost, constraints=constraints, bounds=Bounds(lb, ub), integrality=integrality,
               options={"disp": False, "mip_rel_gap": 1e-7})
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    x = res.x
    charge, discharge = x[ic:ic + n].copy(), x[idd:idd + n].copy()
    charge[charge < 1e-9] = 0.0
    discharge[discharge < 1e-9] = 0.0
    return Plan(charge, discharge, soc_path(charge, discharge, dt_hours, params))


def soc_path(charge, discharge, dt_hours: float = 0.25, params: BatteryParams = BATTERY) -> np.ndarray:
    """State of charge (MWh) at the end of each interval, starting from the daily start value."""
    step = params.eta * dt_hours * np.asarray(charge, float) - dt_hours * np.asarray(discharge, float) / params.eta
    return params.energy_mwh * params.soc_start + np.cumsum(step)


def settle(plan: Plan, prices, dt_hours: float = 0.25, params: BatteryParams = BATTERY) -> dict:
    """P&L of a plan at the given (actual) prices: energy revenue minus the degradation cost of the discharge."""
    p = np.asarray(prices, dtype=float)
    if len(p) != len(plan.charge_mw):
        raise ValueError(f"plan has {len(plan.charge_mw)} intervals, prices {len(p)}")
    bought = float(np.sum(plan.charge_mw) * dt_hours)
    sold = float(np.sum(plan.discharge_mw) * dt_hours)
    gross = float(np.sum(p * dt_hours * plan.net_mw))
    return {
        "pnl_pln": gross - params.degradation_cost * sold,
        "gross_pln": gross,
        "bought_mwh": bought,
        "sold_mwh": sold,
        "cycles": sold / params.usable_mwh if params.usable_mwh else 0.0,
    }


def check_plan(plan: Plan, dt_hours: float = 0.25, params: BatteryParams = BATTERY, tol: float = 1e-6) -> None:
    """Raise if a plan breaks the power, energy, end-of-day or throughput limits."""
    c, d, soc = plan.charge_mw, plan.discharge_mw, plan.soc_mwh
    if (c < -tol).any() or (d < -tol).any() or (c > params.power_mw + tol).any() or (d > params.power_mw + tol).any():
        raise ValueError("power outside [0, P]")
    if ((c > tol) & (d > tol)).any():
        raise ValueError("simultaneous charge and discharge")
    expected = soc_path(c, d, dt_hours, params)
    if not np.allclose(soc, expected, atol=1e-6):
        raise ValueError("state of charge does not follow the energy balance")
    lo, hi = params.energy_mwh * params.soc_min, params.energy_mwh * params.soc_max
    if (soc < lo - tol).any() or (soc > hi + tol).any():
        raise ValueError("state of charge outside its limits")
    if abs(soc[-1] - params.energy_mwh * params.soc_start) > 1e-5:
        raise ValueError("state of charge at the end of the day differs from the start")
    if np.sum(d) * dt_hours > params.max_cycles_per_day * params.usable_mwh + 1e-6:
        raise ValueError("throughput above the daily cap")
