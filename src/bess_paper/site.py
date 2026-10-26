"""Static website: the latest plan, the live track record and, separately, the backtest.

The page is one HTML file with the data embedded as JSON and drawn with Plotly in the browser; the CSV files of
the ledger and the backtest are written next to it for download.

Usage:
    build_site(Path("results"), Path("archive"), Path("site"))
"""
from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import date, datetime, timezone
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from bess_paper.config import BATTERY, GATE_LOCAL, TZ
from bess_paper.live import load_plan, planned_days, read_ledger, settlement_prices_path
from bess_paper.strategies import FIXED_WINDOWS

log = logging.getLogger(__name__)

STRATS = ("model", "perfect", "yesterday", "fixed")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (date, datetime, pd.Timestamp)):
        return o.isoformat()
    return o


def _local(ts) -> list[str]:
    """ISO local times (Europe/Warsaw, with offset) for the browser."""
    return [t.isoformat() for t in pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).tz_convert(TZ)]


def _best_worst(daily: pd.DataFrame, pnl: str, perfect: str, n: int = 5) -> dict:
    cols = {"date": "date", pnl: "pnl", perfect: "perfect", "mae_model": "mae", "spread": "spread"}
    frame = daily.reset_index()
    have = {k: v for k, v in cols.items() if k in frame}
    frame = frame[list(have)].rename(columns=have)
    frame = frame.dropna(subset=["pnl"])
    return {"best": frame.nlargest(n, "pnl").to_dict("records"), "worst": frame.nsmallest(n, "pnl").to_dict("records")}


def backtest_payload(results: Path) -> dict | None:
    path = results / "backtest_daily.csv"
    if not path.exists():
        return None
    daily = pd.read_csv(path, dtype={"date": str}).set_index("date")
    summary = json.loads((results / "backtest_summary.json").read_text(encoding="utf-8"))
    month = daily.index.str[:7]
    pnl_cols = {s: f"pnl_pln_{s}" for s in (*STRATS, "last_week", "week_mean")}
    monthly = daily.groupby(month)[[c for c in pnl_cols.values() if c in daily]].sum()
    periods = {"all": daily.index >= "",
               "hourly": daily.index < "2025-10-01",
               "quarter": daily.index >= "2025-10-01"}
    table = {}
    for key, mask in periods.items():
        d = daily[mask]
        if d.empty:
            continue
        rows = []
        for name in ("model", "yesterday", "last_week", "week_mean", "fixed", "perfect"):
            col = pnl_cols[name]
            if col not in d:
                continue
            rows.append({
                "name": name, "pnl": d[col].sum(), "per_mw_year": d[col].sum() * 365.0 / len(d) / BATTERY.power_mw,
                "capture": d[col].sum() / d["pnl_pln_perfect"].sum(),
                "mae": d[f"mae_{name}"].mean() if f"mae_{name}" in d else None,
                "rmse": float(np.sqrt((d[f"rmse_{name}"] ** 2).mean())) if f"rmse_{name}" in d else None,
                "rank": d[f"rank_{name}"].mean() if f"rank_{name}" in d else None,
                "cycles": d[f"cycles_{name}"].mean() if f"cycles_{name}" in d else None,
            })
        table[key] = {"first": d.index.min(), "last": d.index.max(), "days": len(d), "rows": rows,
                      "cover80": d["cover80"].mean() if "cover80" in d else None}
    bw = _best_worst(daily, "pnl_pln_model", "pnl_pln_perfect")
    days_shown = sorted({r["date"] for r in bw["best"] + bw["worst"]})
    examples = {}
    ipath = results / "backtest_intervals.parquet"
    if ipath.exists() and days_shown:
        iv = pd.read_parquet(ipath)
        iv = iv[iv["day"].isin(days_shown)]
        for day, g in iv.groupby("day"):
            examples[day] = {"t": _local(g["start_utc"]), "actual": g["actual"].tolist(), "p10": g["p10"].tolist(),
                             "p50": g["p50"].tolist(), "p90": g["p90"].tolist(),
                             "net": g["net_model"].tolist(), "net_perfect": g["net_perfect"].tolist()}
    return {
        "summary": summary,
        "daily": {"date": daily.index.tolist(), **{s: daily[c].tolist() for s, c in pnl_cols.items() if c in daily},
                  "mae_model": daily["mae_model"].tolist(), "mae_yesterday": daily["mae_yesterday"].tolist()},
        "monthly": {"month": monthly.index.tolist(),
                    **{s: monthly[c].tolist() for s, c in pnl_cols.items() if c in monthly}},
        "table": table, "best": bw["best"], "worst": bw["worst"], "examples": examples,
    }


def _plan_payload(archive: Path, day: date) -> dict | None:
    plan = load_plan(archive, day)
    if plan is None:
        return None
    iv = pd.DataFrame(plan["intervals"])
    out = {k: plan[k] for k in ("delivery_date", "issued_utc", "gate_closure_utc", "code_version",
                                "expected_pnl_pln", "inputs")}
    out.update(t=_local(iv["start_utc"]), p10=iv["p10"].tolist(), p50=iv["p50"].tolist(), p90=iv["p90"].tolist(),
               net=(iv["discharge_mw"] - iv["charge_mw"]).tolist(), soc=iv["soc_mwh"].tolist())
    out["sold_mwh"] = float(iv["discharge_mw"].sum() * 0.25)
    out["bought_mwh"] = float(iv["charge_mw"].sum() * 0.25)
    out["path"] = f"plans/{day:%Y}/{day.isoformat()}.json"
    prices = settlement_prices_path(archive, day)
    if prices.exists():
        out["actual"] = pd.read_csv(prices)["price"].tolist()
    return out


def live_payload(archive: Path) -> dict:
    plans = planned_days(archive)
    ledger = read_ledger(archive)
    settled = sorted(ledger["date"]) if len(ledger) else []
    upcoming = [d for d in plans if d.isoformat() not in set(settled)]
    last_settled = date.fromisoformat(settled[-1]) if settled else None
    out: dict = {
        "start": plans[0].isoformat() if plans else None,
        "next_plan": _plan_payload(archive, upcoming[-1]) if upcoming else None,
        "last_settled": _plan_payload(archive, last_settled) if last_settled else None,
        "ledger": ledger.sort_values("date").to_dict("list") if len(ledger) else None,
    }
    if len(ledger) >= 6:            # best and worst lists only once they cannot overlap
        bw = _best_worst(ledger.set_index("date"), "pnl_model", "pnl_perfect", min(5, len(ledger) // 2))
        out.update(best=bw["best"], worst=bw["worst"])
    return out


def fixed_windows_text() -> str:
    verb = {"charge": "buy", "discharge": "sell"}
    return ", ".join(f"{verb[side]} {lo // 60:02d}:{lo % 60:02d}-{hi // 60:02d}:{hi % 60:02d}"
                     for lo, hi, side in FIXED_WINDOWS)


def build_payload(results: Path, archive: Path) -> dict:
    return _clean({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "battery": {k: getattr(BATTERY, k) for k in BATTERY.__dataclass_fields__},
        "gate_local": GATE_LOCAL.isoformat(timespec="minutes"),
        "fixed_windows": fixed_windows_text(),
        "live": live_payload(archive),
        "backtest": backtest_payload(results),
    })


def build_site(results: Path, archive: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(results, archive)
    template = resources.files("bess_paper").joinpath("templates/index.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
    (out / "index.html").write_text(html, encoding="utf-8", newline="\n")
    for name in ("backtest_daily.csv", "backtest_summary.json"):
        if (results / name).exists():
            shutil.copyfile(results / name, out / name)
    if (archive / "ledger.csv").exists():
        shutil.copyfile(archive / "ledger.csv", out / "ledger.csv")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    log.info("site written to %s", out / "index.html")
    return out / "index.html"
