"""Live paper trading: the morning plan for tomorrow and the settlement of plans whose prices are published.

Archive layout (the daily workflow keeps it on the ``track-record`` branch; when a plan was published is shown by
the GitHub Actions run that wrote it and by the push in the repository's activity log, not by commit dates, which
the client sets):

    plans/YYYY/YYYY-MM-DD.json            plan for delivery day YYYY-MM-DD, issued before the gate closure on the
                                          day before; never overwritten
    inputs/YYYY/YYYY-MM-DD_features.csv   the model inputs exactly as they were at the decision
    inputs/YYYY/YYYY-MM-DD_pk5l.json      PSE 5-day coordination plan as published at the decision (archived for
                                          later models; the API keeps only its latest version)
    settled/YYYY/YYYY-MM-DD_prices.csv    the clearing prices the day was settled on
    ledger.csv                            one row per settled delivery day, append-only

No orders are sent anywhere: a plan is a file, and its result is computed from the published SDAC prices.

Usage:
    plan = make_plan(prices, system, date(2026, 10, 28), now_utc)
    write_plan(Path("archive"), plan)
    rows = settle_pending(Path("archive"), prices)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from bess_paper import __version__
from bess_paper.backtest import day_metrics
from bess_paper.config import (
    BATTERY,
    DECISION_LOCAL,
    FUND_CUTOFF_LOCAL,
    GATE_LOCAL,
    MODEL,
    PRICE_REPORT,
    TZ,
    BatteryParams,
    ModelParams,
)
from bess_paper.data import complete_days
from bess_paper.features import FEATURES, build_features, lagged_prices, target
from bess_paper.forecast import fit, predict, to_product, train_mask
from bess_paper.optimizer import Plan, check_plan, soc_path
from bess_paper.strategies import fixed_plan, plan_day, settle_day

log = logging.getLogger(__name__)

LEDGER_COLUMNS = [
    "date", "status", "issued_utc", "settled_utc", "code_version", "intervals", "mean_price", "spread",
    "pnl_model", "pnl_perfect", "pnl_yesterday", "pnl_fixed",
    "gross_model", "sold_mwh_model", "cycles_model", "sold_mwh_perfect", "cycles_perfect",
    "mae_model", "mae_yesterday", "rank_model", "rank_yesterday", "cover80", "pinball",
]


class LateError(RuntimeError):
    """The plan would be issued after the gate closure, when the result of the auction may be known."""


def local_time(day: date, t: time) -> datetime:
    """A Polish local time on ``day`` as an aware UTC datetime."""
    return pd.Timestamp(datetime.combine(day, t)).tz_localize(TZ).tz_convert("UTC").to_pydatetime()


def gate_closure(delivery: date) -> datetime:
    return local_time(delivery - timedelta(days=1), GATE_LOCAL)


def code_version() -> str:
    sha = os.environ.get("GITHUB_SHA", "")
    if not sha:
        try:
            out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                                 cwd=Path(__file__).parent)
            sha = out.stdout.strip() if out.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            sha = ""
    ok = len(sha) >= 12 and all(ch in "0123456789abcdef" for ch in sha.lower())
    return f"{__version__}+{sha[:12]}" if ok else __version__


def run_context(replay: bool = False) -> dict:
    """What started the run that makes a plan: recorded in every plan from 16.11.2026 on.

    ``trigger`` is ``schedule`` or ``workflow_dispatch`` in GitHub Actions (with the run's link, whose start time GitHub
    records), ``local`` outside Actions and ``replay`` for ``daily --now``. A replay plan never counts as a trade.
    """
    if replay:
        return {"trigger": "replay"}
    env = os.environ
    if env.get("GITHUB_ACTIONS") != "true":
        return {"trigger": "local"}
    run_id = env.get("GITHUB_RUN_ID", "")
    return {
        "trigger": env.get("GITHUB_EVENT_NAME", "unknown"),
        "actor": env.get("GITHUB_TRIGGERING_ACTOR") or env.get("GITHUB_ACTOR", ""),
        "run_id": run_id,
        "run_attempt": env.get("GITHUB_RUN_ATTEMPT", ""),
        "run_url": f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/{env.get('GITHUB_REPOSITORY', '')}"
                   f"/actions/runs/{run_id}",
    }


def as_of(prices: pd.DataFrame, system: pd.DataFrame, delivery: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The data that exist at the decision for ``delivery``: prices up to D-1, system data up to 06:00 on D-1."""
    known = prices[prices["business_date"] < delivery]
    cutoff = local_time(delivery - timedelta(days=1), FUND_CUTOFF_LOCAL)
    hour_end = pd.DatetimeIndex(system.index) + pd.Timedelta(hours=1)
    return known, system[hour_end <= cutoff]


def make_plan(prices: pd.DataFrame, system: pd.DataFrame, delivery: date, now_utc: datetime,
              params: ModelParams = MODEL, battery: BatteryParams = BATTERY, run: dict | None = None) -> dict:
    """Forecast and schedule for ``delivery``, made from the data known before its gate closure.

    ``issued_utc`` is ``now_utc``; the daily command replaces it by the time the plan is finished and checks the gate
    again before writing it.
    """
    gate = gate_closure(delivery)
    if now_utc >= gate:
        raise LateError(f"{now_utc:%Y-%m-%d %H:%M} UTC is after the gate closure for {delivery} ({gate:%H:%M} UTC)")
    if (prices["business_date"] == delivery).any():
        raise LateError(f"prices for {delivery} are already published")
    prices, system = as_of(prices, system, delivery)
    history = [d for d in complete_days(prices) if d >= delivery - timedelta(days=params.train_days + 8)]
    X = build_features(prices, system, [*history, delivery])
    y = target(prices, X)
    mask = train_mask(X, y, delivery, params)
    if mask.sum() < 30 * 96:
        raise RuntimeError(f"only {mask.sum()} training rows before {delivery}")
    models = fit(X[mask], y[mask], params)
    rows = X[X["day"] == delivery]
    quantiles = predict(models, rows).apply(lambda s: to_product(s, delivery))
    plan = plan_day(quantiles["p50"].to_numpy(), delivery, battery)
    check_plan(plan, 0.25, battery)
    expected = settle_day(plan, quantiles["p50"].to_numpy(), battery)
    last_price_day = max(complete_days(prices))
    return {
        "delivery_date": delivery.isoformat(),
        "issued_utc": now_utc.astimezone(UTC).isoformat(timespec="seconds"),
        "gate_closure_utc": gate.isoformat(timespec="seconds"),
        "decision_local": DECISION_LOCAL.isoformat(timespec="minutes"),
        "code_version": code_version(),
        "run": run if run is not None else run_context(),
        "price_report": PRICE_REPORT,
        "inputs": {
            "last_price_day": last_price_day.isoformat(),
            "system_data_until_utc": local_time(delivery - timedelta(days=1), FUND_CUTOFF_LOCAL).isoformat(),
            "training_days": int(len(set(X.loc[mask, "day"]))),
            "training_rows": int(mask.sum()),
        },
        "battery": {k: getattr(battery, k) for k in battery.__dataclass_fields__},
        "expected_pnl_pln": round(expected["pnl_pln"], 2),
        "intervals": [
            {"start_utc": ts.isoformat(), "p10": round(float(a), 2), "p50": round(float(b), 2),
             "p90": round(float(c), 2), "charge_mw": round(float(ch), 6), "discharge_mw": round(float(dc), 6),
             "soc_mwh": round(float(s), 6)}
            for ts, a, b, c, ch, dc, s in zip(pd.DatetimeIndex(rows["start_utc"]), quantiles["p10"], quantiles["p50"],
                                              quantiles["p90"], plan.charge_mw, plan.discharge_mw, plan.soc_mwh,
                                              strict=True)
        ],
        "_features": rows[["start_utc", *FEATURES]],
    }


ARCHIVE_README = """# Track record of pl-bess-paper-trading

Written only by the daily GitHub Actions run of the main branch. Paper trading: no orders are sent to any market.

- `plans/YYYY/YYYY-MM-DD.json`: the plan for delivery day YYYY-MM-DD (price forecast and schedule), committed
  before 12:00 Polish time on the day before. A plan is never replaced.
- `inputs/YYYY/`: the model inputs at the decision (`_features.csv`) and the PSE 5-day coordination plan as
  published at that moment (`_pk5l.json`).
- `settled/YYYY/YYYY-MM-DD_prices.csv`: the SDAC clearing prices the day was settled on.
- `ledger.csv`: one row per settled delivery day, appended once and never rewritten. P&L in PLN for a
  1 MW / 2 MWh battery, net of a degradation cost of 30 PLN/MWh discharged.

Evidence of when a plan was published is kept by GitHub, not by this branch: the Actions run that wrote it (linked
in each plan from 16.11.2026 on, with what triggered it: the GitHub schedule or a manual or scripted dispatch) and the
push in the repository's activity log. Commit dates are written by the client and prove nothing on their own. Since
30.10.2026 GitHub has started the scheduled runs after the gate, so plans come from dispatches by a scheduler on the
owner's computer; mornings without one have no plan and count as no trade.
"""


def ensure_readme(archive: Path) -> None:
    """Write the archive's README, or bring it up to date (a normal new commit; history is never rewritten)."""
    path = archive / "README.md"
    if not path.exists() or path.read_text(encoding="utf-8") != ARCHIVE_README:
        archive.mkdir(parents=True, exist_ok=True)
        path.write_text(ARCHIVE_README, encoding="utf-8", newline="\n")


def plan_path(archive: Path, delivery: date) -> Path:
    return archive / "plans" / f"{delivery:%Y}" / f"{delivery.isoformat()}.json"


def write_plan(archive: Path, plan: dict) -> Path:
    """Save a plan and its inputs. An existing plan for the same day is never replaced."""
    delivery = date.fromisoformat(plan["delivery_date"])
    path = plan_path(archive, delivery)
    if path.exists():
        raise FileExistsError(f"a plan for {delivery} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in plan.items() if not k.startswith("_")}
    path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n")
    features = plan.get("_features")
    if features is not None:
        inputs = archive / "inputs" / f"{delivery:%Y}"
        inputs.mkdir(parents=True, exist_ok=True)
        features.to_csv(inputs / f"{delivery.isoformat()}_features.csv", index=False, lineterminator="\n")
    log.info("plan for %s written to %s", delivery, path)
    return path


def write_snapshot(archive: Path, delivery: date, name: str, records: list[dict]) -> Path:
    path = archive / "inputs" / f"{delivery:%Y}" / f"{delivery.isoformat()}_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=0, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


def load_plan(archive: Path, delivery: date) -> dict | None:
    path = plan_path(archive, delivery)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def planned_days(archive: Path) -> list[date]:
    return sorted(date.fromisoformat(p.stem) for p in (archive / "plans").glob("*/*.json"))


def read_ledger(archive: Path) -> pd.DataFrame:
    path = archive / "ledger.csv"
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_csv(path, dtype={"date": str, "status": str, "code_version": str})


def append_ledger(archive: Path, rows: list[dict]) -> pd.DataFrame:
    """Append settled days. Rows already in the ledger are never rewritten; a duplicate date is an error."""
    path = archive / "ledger.csv"
    ledger = read_ledger(archive)
    if not rows:
        return ledger
    new = pd.DataFrame(rows).reindex(columns=LEDGER_COLUMNS)
    clash = set(new["date"]) & set(ledger["date"])
    if clash:
        raise ValueError(f"days already settled: {sorted(clash)}")
    text = new.to_csv(index=False, header=not path.exists(), lineterminator="\n", float_format="%.4f")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return read_ledger(archive)


def _plan_from_file(plan: dict) -> tuple[Plan, pd.DataFrame]:
    iv = pd.DataFrame(plan["intervals"])
    c, d = iv["charge_mw"].to_numpy(float), iv["discharge_mw"].to_numpy(float)
    return Plan(c, d, soc_path(c, d)), iv


def settle_day_live(delivery: date, prices: pd.DataFrame, plan: dict | None, now_utc: datetime,
                    battery: BatteryParams = BATTERY) -> dict:
    """Ledger row of one delivery day: the archived plan and the benchmarks, at the published prices."""
    day_prices = prices[prices["business_date"] == delivery].sort_values("start_utc")
    actual = to_product(day_prices["price"].to_numpy(float), delivery)
    yesterday = to_product(lagged_prices(prices, delivery, 1), delivery)

    row: dict = {"date": delivery.isoformat(), "settled_utc": now_utc.isoformat(timespec="seconds"),
                 "intervals": len(actual), "mean_price": float(actual.mean()),
                 "spread": float(actual.max() - actual.min())}
    perfect = settle_day(plan_day(actual, delivery, battery), actual, battery)
    fixed = settle_day(fixed_plan(delivery, battery), actual, battery)
    yest = settle_day(plan_day(yesterday, delivery, battery), actual, battery) if np.isfinite(yesterday).all() \
        else {"pnl_pln": 0.0}
    row.update(pnl_perfect=perfect["pnl_pln"], pnl_fixed=fixed["pnl_pln"], pnl_yesterday=yest["pnl_pln"],
               sold_mwh_perfect=perfect["sold_mwh"], cycles_perfect=perfect["cycles"])
    forecasts = {"yesterday": yesterday}
    quantiles = None
    replay = plan is not None and plan.get("run", {}).get("trigger") == "replay"
    if plan is None or replay:
        # no plan, or a replayed plan (written after the fact with daily --now): no trade for the live record
        row.update(status="replay" if replay else "no_plan", pnl_model=0.0, gross_model=0.0, sold_mwh_model=0.0,
                   cycles_model=0.0)
    else:
        model_plan, iv = _plan_from_file(plan)
        if len(iv) != len(actual):
            raise ValueError(f"plan for {delivery} has {len(iv)} intervals, prices {len(actual)}")
        res = settle_day(model_plan, actual, battery)
        row.update(status="traded", issued_utc=plan["issued_utc"], code_version=plan["code_version"],
                   pnl_model=res["pnl_pln"], gross_model=res["gross_pln"], sold_mwh_model=res["sold_mwh"],
                   cycles_model=res["cycles"])
        forecasts["model"] = iv["p50"].to_numpy(float)
        quantiles = iv[["p10", "p50", "p90"]]
    metrics = day_metrics(actual, forecasts, quantiles)
    row.update({k: v for k, v in metrics.items() if k in LEDGER_COLUMNS})
    return row


def settle_pending(archive: Path, prices: pd.DataFrame, now_utc: datetime | None = None,
                   battery: BatteryParams = BATTERY) -> list[dict]:
    """Settle every delivery day from the first plan on whose prices are complete and that is not in the ledger.

    A day without a plan (e.g. the morning run failed) is recorded as ``no_plan`` with zero P&L.
    """
    now_utc = now_utc or datetime.now(UTC)
    plans = planned_days(archive)
    if not plans:
        return []
    done = set(read_ledger(archive)["date"])
    complete = set(complete_days(prices))
    rows = []
    day, last = plans[0], max(complete, default=plans[0] - timedelta(days=1))
    while day <= last:
        if day.isoformat() in done:
            pass
        elif day not in complete:
            log.warning("prices for %s are incomplete; the day stays unsettled", day)
        else:
            rows.append(settle_day_live(day, prices, load_plan(archive, day), now_utc, battery))
            write_settlement_prices(archive, day, prices)
            log.info("settled %s: %s, P&L %.0f PLN (perfect foresight %.0f)", day, rows[-1]["status"],
                     rows[-1]["pnl_model"], rows[-1]["pnl_perfect"])
        day += timedelta(days=1)
    append_ledger(archive, rows)
    return rows


def settlement_prices_path(archive: Path, day: date) -> Path:
    return archive / "settled" / f"{day:%Y}" / f"{day.isoformat()}_prices.csv"


def write_settlement_prices(archive: Path, day: date, prices: pd.DataFrame) -> None:
    path = settlement_prices_path(archive, day)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = prices[prices["business_date"] == day].sort_values("start_utc")
    out = pd.DataFrame({"start_utc": pd.DatetimeIndex(rows["start_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "price": rows["price"].to_numpy()})
    out.to_csv(path, index=False, lineterminator="\n")


def drop_late_plans(archive: Path, new_files: list[Path], now_utc: datetime) -> list[Path]:
    """Before a push: remove new plans (and their inputs) whose gate closure has passed, so none is published late."""
    dropped = []
    for path in new_files:
        if path.suffix != ".json" or path.parent.parent.name != "plans" or not path.exists():
            continue
        plan = json.loads(path.read_text(encoding="utf-8"))
        if datetime.fromisoformat(plan["gate_closure_utc"]) > now_utc:
            continue
        day = plan["delivery_date"]
        for extra in [path, *sorted((archive / "inputs" / day[:4]).glob(f"{day}_*"))]:
            extra.unlink(missing_ok=True)
            dropped.append(extra)
        log.warning("plan for %s not pushed: the gate closed at %s", day, plan["gate_closure_utc"])
    return dropped


def _block(frame: pd.DataFrame) -> dict:
    pf = float(frame["pnl_perfect"].sum())
    out: dict = {"days": int(len(frame)), "pnl_perfect": pf}
    for k in ("model", "yesterday", "fixed"):
        total = float(frame[f"pnl_{k}"].sum())
        out[f"pnl_{k}"] = total
        out[f"share_{k}"] = total / pf if pf else None
    return out


def live_summary(ledger: pd.DataFrame, archive: Path | None = None) -> dict:
    """Headline of the live record, computed from the ledger alone (so it cannot drift from it)."""
    if ledger.empty:
        return {"days": 0, "badge": "no settled day yet"}
    g = ledger.sort_values("date")
    traded = g[g["status"] == "traded"]
    out: dict = {
        "first_day": g["date"].iloc[0], "last_day": g["date"].iloc[-1], "days": int(len(g)),
        "traded": int(len(traded)), "no_plan": int((g["status"] == "no_plan").sum()),
        "replay": int((g["status"] == "replay").sum()),
        "all_days": _block(g), "traded_days": _block(traded),
    }
    if len(traded):
        out["traded_days"].update(mae_model=float(traded["mae_model"].mean()),
                                  mae_yesterday=float(traded["mae_yesterday"].mean()),
                                  cover80=float(traded["cover80"].mean()))
    if archive is not None:
        triggers: dict[str, int] = {}
        for day in traded["date"]:
            plan = load_plan(archive, date.fromisoformat(day)) or {}
            key = plan.get("run", {}).get("trigger", "not recorded")
            triggers[key] = triggers.get(key, 0) + 1
        out["plan_triggers"] = triggers
    share = out["all_days"]["share_model"]
    out["badge"] = (f"{out['traded']} of {out['days']} days traded, "
                    f"{100 * share:.1f} % of perfect foresight" if share is not None else f"{out['days']} days")
    return out


def _dmy(day: str) -> str:
    return f"{day[8:10]}.{day[5:7]}.{day[:4]}"


def live_markdown(summary: dict) -> str:
    """The live summary as the Markdown block of the README."""
    if not summary.get("days"):
        return "No delivery day has been settled yet.\n"
    a, t = summary["all_days"], summary["traded_days"]

    def cell(block: dict, k: str) -> str:
        share = block.get(f"share_{k}")
        return f"{block[f'pnl_{k}']:,.0f} ({100 * share:.1f} %)" if share is not None else f"{block[f'pnl_{k}']:,.0f}"

    lines = [
        f"Ledger from {_dmy(summary['first_day'])} to {_dmy(summary['last_day'])}: {summary['days']} settled delivery "
        f"days, {summary['traded']} with a plan published before the gate, {summary['no_plan']} without "
        f"(0 PLN for the forecast strategy)" + (f", {summary['replay']} replayed" if summary["replay"] else "") + ".",
        "P&L in PLN, in brackets the share of perfect foresight on the same days.",
        "",
        "| Days | n | Forecast + MILP | Yesterday's prices | Fixed hours | Perfect foresight |",
        "|---|---:|---:|---:|---:|---:|",
        f"| all settled days | {a['days']} | {cell(a, 'model')} | {cell(a, 'yesterday')} | {cell(a, 'fixed')} | "
        f"{a['pnl_perfect']:,.0f} |",
        f"| days with a plan | {t['days']} | {cell(t, 'model')} | {cell(t, 'yesterday')} | {cell(t, 'fixed')} | "
        f"{t['pnl_perfect']:,.0f} |",
        "",
    ]
    if t["days"]:
        lines.append(f"On the days with a plan the forecast had a mean absolute error of {t['mae_model']:.0f} PLN/MWh "
                     f"(yesterday's prices {t['mae_yesterday']:.0f}) and its 10-90 % range contained "
                     f"{100 * t['cover80']:.0f} % of prices. {t['days']} days are far too few to compare strategies.")
        lines.append("")
    return "\n".join(lines)
