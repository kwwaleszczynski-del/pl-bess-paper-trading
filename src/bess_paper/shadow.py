"""Shadow run of the live days: the same forecast and schedule, re-computed afterwards with the backtest's data.

The live plan of day D used the PSE system data as first published on the morning of D-1 and a model refitted that
morning. The backtest sees the system data after PSE's corrections and refits weekly. To measure both differences on
the live days, every settled day of the ledger is re-planned twice with today's data:

- ``daily``: the live procedure (``make_plan``, daily refit, as of the decision of D) on the corrected data; the
  difference to the live plan is the effect of the data revisions (the model code is unchanged since the first plan);
- ``weekly``: the backtest procedure (``run_backtest``, weekly refit); its difference to ``daily`` is the refit.

The results go to ``results/`` on the main branch and never into the ledger. On days without a live plan the shadow
P&L shows what the strategy would have earned; it is simulated, not part of the live record.

Usage:
    per_day, summary = shadow_live(Path("archive"), prices, system)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from bess_paper.backtest import run_backtest
from bess_paper.config import BATTERY, DECISION_LOCAL, MODEL, BatteryParams, ModelParams
from bess_paper.features import FEATURES
from bess_paper.forecast import to_product
from bess_paper.live import load_plan, local_time, make_plan, read_ledger
from bess_paper.optimizer import Plan, settle, soc_path

log = logging.getLogger(__name__)

SYSTEM_FEATURES = ["wind_d2", "pv_d2", "demand_d2", "wind_d1_night", "demand_d1_night", "wind_d1_last"]


def _settle(net: np.ndarray, actual: np.ndarray, battery: BatteryParams) -> float:
    charge, discharge = np.clip(-net, 0, None), np.clip(net, 0, None)
    plan = Plan(charge, discharge, soc_path(charge, discharge, 0.25, battery))
    return settle(plan, actual, 0.25, battery)["pnl_pln"]


def shadow_live(archive: Path, prices: pd.DataFrame, system: pd.DataFrame, params: ModelParams = MODEL,
                battery: BatteryParams = BATTERY) -> tuple[pd.DataFrame, dict]:
    ledger = read_ledger(archive).sort_values("date")
    days = [date.fromisoformat(d) for d in ledger["date"]]
    _, weekly = run_backtest(prices, system, start=days[0], end=days[-1], params=params, battery=battery)
    rows = []
    for day, (_, led) in zip(days, ledger.iterrows(), strict=True):
        actual = to_product(prices.loc[prices["business_date"] == day].sort_values("start_utc")["price"], day)
        decision = local_time(day - timedelta(days=1), DECISION_LOCAL)
        shadow = make_plan(prices[prices["business_date"] < day], system, day, decision, params, battery,
                           run={"trigger": "shadow"})
        sd = pd.DataFrame(shadow["intervals"])
        net_d = (sd["discharge_mw"] - sd["charge_mw"]).to_numpy()
        wk = weekly[weekly["day"] == day]
        net_w = wk["net_model"].to_numpy()
        row = {"date": day.isoformat(), "status": led["status"], "pnl_perfect": led["pnl_perfect"],
               "pnl_live": led["pnl_model"] if led["status"] == "traded" else np.nan,
               "pnl_shadow_daily": _settle(net_d, actual, battery),
               "pnl_shadow_weekly": _settle(net_w, actual, battery),
               "mae_live": led["mae_model"] if led["status"] == "traded" else np.nan,
               "mae_shadow_daily": float(np.abs(sd["p50"].to_numpy() - actual).mean()),
               "mae_shadow_weekly": float(np.abs(wk["p50"].to_numpy() - actual).mean()),
               "plan_diff_weekly_mw": float(np.abs(net_w - net_d).mean())}
        plan = load_plan(archive, day) if led["status"] == "traded" else None
        if plan is not None:
            lv = pd.DataFrame(plan["intervals"])
            net_l = (lv["discharge_mw"] - lv["charge_mw"]).to_numpy()
            row["plan_diff_daily_mw"] = float(np.abs(net_l - net_d).mean())
            row["p50_diff_daily"] = float(np.abs(lv["p50"].to_numpy() - sd["p50"].to_numpy()).mean())
            inputs = archive / "inputs" / f"{day:%Y}" / f"{day.isoformat()}_features.csv"
            if inputs.exists():
                live_x = pd.read_csv(inputs)
                now_x = shadow["_features"].reset_index(drop=True)
                for col in SYSTEM_FEATURES:
                    row[f"rev_{col}"] = float((now_x[col] - live_x[col]).abs().max())
                other = [c for c in FEATURES if c not in SYSTEM_FEATURES]
                row["max_diff_other_features"] = float(np.nanmax(np.abs(now_x[other].to_numpy(float)
                                                                         - live_x[other].to_numpy(float))))
        rows.append(row)
        log.info("shadow %s: live %s, daily %.0f, weekly %.0f PLN", day, row["pnl_live"], row["pnl_shadow_daily"],
                 row["pnl_shadow_weekly"])
    per_day = pd.DataFrame(rows)
    traded = per_day[per_day["status"] == "traded"]
    summary = {
        "days": int(len(per_day)), "traded": int(len(traded)),
        "first_day": per_day["date"].iloc[0], "last_day": per_day["date"].iloc[-1],
        "traded_days": {k: float(traded[k].sum()) for k in ("pnl_live", "pnl_shadow_daily", "pnl_shadow_weekly",
                                                             "pnl_perfect")},
        "all_days": {k: float(per_day[k].sum()) for k in ("pnl_shadow_daily", "pnl_shadow_weekly", "pnl_perfect")},
        "mae_traded": {k: float(traded[k].mean()) for k in ("mae_live", "mae_shadow_daily", "mae_shadow_weekly")},
        "plan_diff_mw": {"live_vs_daily": float(traded["plan_diff_daily_mw"].mean()) if len(traded) else None,
                         "daily_vs_weekly": float(per_day["plan_diff_weekly_mw"].mean())},
        "max_system_feature_revision_mw": {c: float(traded[f"rev_{c}"].max()) for c in SYSTEM_FEATURES
                                           if f"rev_{c}" in traded},
    }
    return per_day, summary
