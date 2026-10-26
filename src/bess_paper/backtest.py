"""Walk-forward backtest with as-of inputs, from the first day of the PSE reports API.

For every week of delivery days a new set of quantile models is fitted on the 365 delivery days before the first
day of the week (the first models of 2024 are trained on the legacy hourly RCE of 2023-24). Each day is then
forecast from its own as-of features, planned by every strategy and settled on the SDAC clearing prices.

The live run refits every morning; the weekly refit of the backtest uses slightly older models, never newer ones.

Usage:
    daily, intervals = run_backtest(prices, system, start=date(2024, 6, 14), end=date(2026, 10, 26))
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

from bess_paper.config import BATTERY, FIRST_DAY, MODEL, BatteryParams, ModelParams
from bess_paper.data import complete_days
from bess_paper.features import build_features, target
from bess_paper.forecast import BENCHMARKS, fit, predict, to_product, train_mask
from bess_paper.strategies import STRATEGIES, plan_day, run_day, settle_day

log = logging.getLogger(__name__)

FORECASTERS = ("model", *BENCHMARKS)


def day_metrics(actual: np.ndarray, forecasts: dict[str, np.ndarray], quantiles: pd.DataFrame | None) -> dict:
    """Errors of each forecaster on one day, at the product resolution (hourly values before 2025-10-01)."""
    out = {}
    for name, f in forecasts.items():
        err = f - actual
        ok = np.isfinite(err)
        out[f"mae_{name}"] = float(np.abs(err[ok]).mean()) if ok.any() else np.nan
        out[f"rmse_{name}"] = float(np.sqrt((err[ok] ** 2).mean())) if ok.any() else np.nan
        # how well the forecast ranks the intervals of the day, which is what the schedule depends on
        out[f"rank_{name}"] = float(pd.Series(f[ok]).corr(pd.Series(actual[ok]), method="spearman")) \
            if ok.sum() > 2 and np.ptp(f[ok]) > 0 and np.ptp(actual[ok]) > 0 else np.nan
    if quantiles is not None:
        lo, hi = quantiles["p10"].to_numpy(), quantiles["p90"].to_numpy()
        out["cover80"] = float(((actual >= lo) & (actual <= hi)).mean())
        pin = []
        for col in quantiles.columns:
            q = int(col[1:]) / 100
            e = actual - quantiles[col].to_numpy()
            pin.append(np.maximum(q * e, (q - 1) * e).mean())
        out["pinball"] = float(np.mean(pin))
    return out


def evaluate_day(day: date, actual: np.ndarray, rows: pd.DataFrame, quantiles: pd.DataFrame,
                 battery: BatteryParams = BATTERY) -> tuple[dict, pd.DataFrame]:
    """Strategies, P&L and forecast errors of one delivery day (``rows``: its feature rows, in time order)."""
    signals = {"model": to_product(quantiles["p50"], day)}
    signals.update({name: to_product(rows[col], day) for name, col in BENCHMARKS.items()})
    q_prod = quantiles.apply(lambda s: to_product(s, day))
    results = run_day(day, actual, signals, battery)
    for name in ("last_week", "week_mean"):          # decision value of the other benchmark forecasts
        if np.isfinite(signals[name]).all():
            plan = plan_day(signals[name], day, battery)
            results[name] = {"plan": plan, **settle_day(plan, actual, battery)}
    rec: dict = {"date": day, "intervals": len(actual), "mean_price": float(actual.mean()),
                 "spread": float(actual.max() - actual.min())}
    for name, r in results.items():
        rec.update({f"{k}_{name}": v for k, v in r.items() if k != "plan"})
    rec.update(day_metrics(actual, signals, q_prod))
    detail = pd.DataFrame({"start_utc": rows["start_utc"].to_numpy(), "day": day, "actual": actual,
                           **{c: q_prod[c].to_numpy() for c in q_prod.columns},
                           "yesterday": signals["yesterday"]})
    for name in ("model", "perfect", "yesterday", "fixed"):
        detail[f"net_{name}"] = results[name]["plan"].net_mw
    detail["soc_model"] = results["model"]["plan"].soc_mwh
    return rec, detail


_STATE: dict = {}


def _init_worker(state: dict) -> None:
    _STATE.update(state)
    _STATE["by_day"] = state["X"].groupby("day").indices


def _run_block(block_start: date, block: list[date]) -> tuple[list[dict], list[pd.DataFrame]]:
    """Fit the models for one block of days on the data before its first day, then evaluate each day."""
    X, y, params, battery = _STATE["X"], _STATE["y"], _STATE["params"], _STATE["battery"]
    mask = train_mask(X, y, block_start, params)
    models = fit(X[mask], y[mask], params)
    records, details = [], []
    for day in block:
        rows = X.iloc[_STATE["by_day"][day]]
        quantiles = predict(models, rows)
        actual = to_product(y[rows.index], day)
        rec, detail = evaluate_day(day, actual, rows, quantiles, battery)
        rec["train_rows"] = int(mask.sum())
        records.append(rec)
        details.append(detail)
    return records, details


def run_backtest(prices: pd.DataFrame, system: pd.DataFrame, start: date = FIRST_DAY, end: date | None = None,
                 params: ModelParams = MODEL, battery: BatteryParams = BATTERY,
                 workers: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily results and quarter-hourly details of the walk-forward backtest over ``start``..``end``.

    ``workers`` > 1 evaluates the weekly blocks in parallel processes (same results, in any order).
    """
    days = complete_days(prices)
    test_days = [d for d in days if d >= start and (end is None or d <= end)]
    if not test_days:
        raise ValueError("no complete delivery days to backtest")
    train_from = test_days[0] - timedelta(days=params.train_days + 8)
    feature_days = [d for d in days if train_from <= d <= test_days[-1]]
    X = build_features(prices, system, feature_days)
    y = target(prices, X)
    log.info("features: %d rows, %d delivery days; backtest %s..%s (%d days)",
             len(X), len(feature_days), test_days[0], test_days[-1], len(test_days))

    blocks, block_start, remaining = [], test_days[0], list(test_days)
    while remaining:
        block_end = block_start + timedelta(days=params.retrain_every)
        block = [d for d in remaining if d < block_end]
        remaining = [d for d in remaining if d >= block_end]
        if block:
            blocks.append((block_start, block))
        block_start = block_end

    state = {"X": X, "y": y, "params": params, "battery": battery}
    records, details = [], []
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(state,)) as pool:
            futures = [pool.submit(_run_block, b0, blk) for b0, blk in blocks]
            for i, fut in enumerate(futures):
                recs, dets = fut.result()
                records += recs
                details += dets
                log.info("block %d/%d done, %.0f s", i + 1, len(blocks), time.time() - t0)
    else:
        _init_worker(state)
        for b0, blk in blocks:
            recs, dets = _run_block(b0, blk)
            records += recs
            details += dets
            log.info("%s..%s: %d training rows, %.0f s", blk[0], blk[-1], recs[0]["train_rows"], time.time() - t0)
    _STATE.clear()
    daily = pd.DataFrame(records).set_index("date").sort_index()
    detail = pd.concat(details, ignore_index=True).sort_values("start_utc", ignore_index=True)
    return daily, detail


def summary(daily: pd.DataFrame, battery: BatteryParams = BATTERY) -> dict:
    """Headline figures of a set of settled days."""
    n = len(daily)
    out: dict = {"days": n, "first_day": str(daily.index.min()), "last_day": str(daily.index.max())}
    for s in (*STRATEGIES, "last_week", "week_mean"):
        if f"pnl_pln_{s}" not in daily:
            continue
        total = float(daily[f"pnl_pln_{s}"].sum())
        out[f"pnl_{s}"] = total
        out[f"pnl_per_mw_year_{s}"] = total * 365.0 / n / battery.power_mw if n else np.nan
        out[f"cycles_per_day_{s}"] = float(daily[f"cycles_{s}"].mean())
    pf = out["pnl_perfect"]
    for s in ("model", "yesterday", "fixed", "last_week", "week_mean"):
        if f"pnl_{s}" in out:
            out[f"capture_{s}"] = out[f"pnl_{s}"] / pf if pf else np.nan
    for f in FORECASTERS:
        if f"mae_{f}" in daily:
            out[f"mae_{f}"] = float(daily[f"mae_{f}"].mean())
            out[f"rmse_{f}"] = float(np.sqrt((daily[f"rmse_{f}"] ** 2).mean()))
            out[f"rank_{f}"] = float(daily[f"rank_{f}"].mean())
    for k in ("cover80", "pinball"):
        if k in daily:
            out[k] = float(daily[k].mean())
    return out
