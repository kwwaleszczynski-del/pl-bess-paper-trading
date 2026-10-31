"""Fixed settings: market calendar, decision time, battery and model parameters.

Usage:
    from bess_paper.config import BATTERY, GATE_LOCAL, TZ
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import numpy as np

TZ = "Europe/Warsaw"

FIRST_DAY = date(2024, 6, 14)       # first business date of the PSE reports API (and of the backtest)
QH_FROM = date(2025, 10, 1)         # first delivery day with 15-minute SDAC products (hourly before)
WARMUP_START = date(2023, 6, 1)     # legacy hourly history, used only to train the first models of the backtest

GATE_LOCAL = time(12, 0)            # SDAC order book gate closure for delivery day D, on D-1 (Polish time)
DECISION_LOCAL = time(7, 0)         # every input of the plan for D is published by about this time on D-1; the plan
                                    # is issued by the first scheduled run after it that starts before the gate
FUND_CUTOFF_LOCAL = time(6, 0)      # system data of D-1 (wind, PV, demand) are used up to this local time only

PRICE_REPORT = "csdac-pln"          # settlement price: SDAC clearing price of the Polish bidding zone


@dataclass(frozen=True)
class BatteryParams:
    """1 MW / 2 MWh battery, same assumptions as the perfect-foresight projects of this portfolio."""

    power_mw: float = 1.0
    energy_mwh: float = 2.0
    round_trip_efficiency: float = 0.87
    soc_min: float = 0.05           # fraction of energy_mwh
    soc_max: float = 0.95
    soc_start: float = 0.5          # also required at the end of each day, so days are independent
    max_cycles_per_day: float = 1.5
    degradation_cost: float = 30.0  # PLN per MWh discharged, in the objective and in the reported P&L

    @property
    def usable_mwh(self) -> float:
        return self.energy_mwh * (self.soc_max - self.soc_min)

    @property
    def eta(self) -> float:
        """One-way efficiency, split symmetrically between charging and discharging."""
        return float(np.sqrt(self.round_trip_efficiency))


BATTERY = BatteryParams()


@dataclass(frozen=True)
class ModelParams:
    """Walk-forward gradient boosting. Chosen before the backtest was run and not tuned on it."""

    train_days: int = 365           # rolling training window (delivery days)
    retrain_every: int = 7          # backtest: refit once a week (the live run refits every day)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    random_state: int = 0


MODEL = ModelParams()
