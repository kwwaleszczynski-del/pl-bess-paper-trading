"""Command line interface.

Usage:
    python -m bess_paper fetch                         # download/update the PSE cache in data/
    python -m bess_paper backtest --offline            # walk-forward backtest -> results/
    python -m bess_paper daily --archive-dir archive   # settle published days, plan tomorrow (before 12:00)
    python -m bess_paper site --archive-dir archive --out site
    python -m bess_paper figures                       # README figures -> docs/img/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bess_paper.config import FIRST_DAY, MODEL, TZ, WARMUP_START

log = logging.getLogger("bess_paper")


def _date(text: str) -> date:
    return date.fromisoformat(text)


def _now(text: str | None) -> datetime:
    if not text:
        return datetime.now(timezone.utc)
    ts = pd.Timestamp(text)
    ts = ts.tz_localize(TZ) if ts.tzinfo is None else ts
    return ts.tz_convert("UTC").to_pydatetime()


def cmd_fetch(args) -> int:
    from bess_paper.data import load_prices, load_system, make_client
    client = make_client(args.data_dir)
    end = args.end or (datetime.now(timezone.utc) + timedelta(days=1)).date()
    prices = load_prices(client, args.start, end)
    system = load_system(client, args.start, min(end, datetime.now(timezone.utc).date()))
    log.info("prices: %d quarter-hours %s..%s; system: %d hours", len(prices), prices["business_date"].min(),
             prices["business_date"].max(), len(system))
    return 0


def cmd_backtest(args) -> int:
    from bess_paper.backtest import run_backtest, summary
    from bess_paper.data import load_prices, load_system, make_client
    client = make_client(args.data_dir, offline=args.offline)
    end = args.end or (datetime.now(timezone.utc) - timedelta(days=1)).date()
    prices = load_prices(client, WARMUP_START, end)
    system = load_system(client, WARMUP_START, end)
    daily, detail = run_backtest(prices, system, start=args.start, end=end, workers=args.workers)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "backtest_daily.csv", float_format="%.4f", lineterminator="\n")
    detail = detail.assign(start_utc=pd.DatetimeIndex(detail["start_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           day=detail["day"].astype(str))
    num = detail.select_dtypes("number").columns
    detail[num] = detail[num].astype("float32")
    detail.to_parquet(out / "backtest_intervals.parquet", index=False)
    s = summary(daily)
    s.update(generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
             data_until=str(end), model={k: getattr(MODEL, k) for k in MODEL.__dataclass_fields__})
    (out / "backtest_summary.json").write_text(json.dumps(s, indent=1, default=float) + "\n", encoding="utf-8",
                                               newline="\n")
    log.info("backtest %s..%s: model %.0f PLN (%.0f%% of perfect foresight), yesterday %.0f, fixed %.0f",
             s["first_day"], s["last_day"], s["pnl_model"], 100 * s["capture_model"], s["pnl_yesterday"],
             s["pnl_fixed"])
    return 0


def cmd_daily(args) -> int:
    from bess_paper.data import load_prices, load_system, make_client
    from bess_paper.live import (LateError, ensure_readme, load_plan, local_time, make_plan, settle_pending,
                                 write_plan, write_snapshot)
    archive = Path(args.archive_dir)
    ensure_readme(archive)
    now = _now(args.now)
    today = pd.Timestamp(now).tz_convert(TZ).date()
    delivery = today + timedelta(days=1)
    client = make_client(args.data_dir, offline=args.offline)
    start = max(FIRST_DAY, delivery - timedelta(days=MODEL.train_days + 10))
    prices = load_prices(client, start, delivery)
    system = load_system(client, start, today)
    if args.now is not None:
        # replay: drop what was published after --now (prices of day X appear about 13:30-14:45 on X-1; 15:00 is used)
        published = {d: local_time(d - timedelta(days=1), time(15, 0)) for d in set(prices["business_date"])}
        prices = prices[[published[d] <= now for d in prices["business_date"]]]
        system = system[system.index + pd.Timedelta(hours=1) <= pd.Timestamp(now)]

    settled = settle_pending(archive, prices, now)
    log.info("settled %d day(s)", len(settled))

    if load_plan(archive, delivery) is not None:
        log.info("a plan for %s already exists; nothing to do", delivery)
        return 0
    try:
        plan = make_plan(prices, system, delivery, now)
    except LateError as e:
        log.warning("no plan for %s: %s", delivery, e)
        return 0
    write_plan(archive, plan)
    if not args.offline and args.now is None:          # a replay cannot see the PK5L version of that moment
        try:
            records = client.records("pk5l-wp", filter=f"business_date eq '{delivery.isoformat()}'")
            write_snapshot(archive, delivery, "pk5l", records)
        except Exception as e:      # the snapshot is an extra; the plan stands without it
            log.warning("PK5L snapshot failed: %s", e)
    log.info("plan for %s: expected P&L %.0f PLN at the median forecast, %.2f MWh sold", delivery,
             plan["expected_pnl_pln"], sum(r["discharge_mw"] for r in plan["intervals"]) * 0.25)
    return 0


def cmd_figures(args) -> int:
    from bess_paper.figures import write_figures
    write_figures(Path(args.results_dir), Path(args.out), args.day)
    return 0


def cmd_site(args) -> int:
    from bess_paper.site import build_site
    build_site(Path(args.results_dir), Path(args.archive_dir), Path(args.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bess_paper", description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--data-dir", default="data", help="cache of PSE downloads (default: data)")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download or update the PSE data cache")
    p.add_argument("--start", type=_date, default=WARMUP_START)
    p.add_argument("--end", type=_date)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("backtest", help="walk-forward backtest with as-of inputs")
    p.add_argument("--start", type=_date, default=FIRST_DAY)
    p.add_argument("--end", type=_date, help="last delivery day (default: yesterday)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--offline", action="store_true", help="use the cache only")
    p.add_argument("--workers", type=int, default=1, help="parallel processes for the weekly blocks")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("daily", help="settle published days and plan tomorrow (before the gate closure)")
    p.add_argument("--archive-dir", default="archive")
    p.add_argument("--now", help="pretend the current time is this (ISO, Polish time if no offset); for tests")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_daily)

    p = sub.add_parser("figures", help="README figures from the backtest results")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="docs/img")
    p.add_argument("--day", help="example day (default: a typical day of the 15-minute period)")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("site", help="build the static website")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--archive-dir", default="archive")
    p.add_argument("--out", default="site")
    p.set_defaults(func=cmd_site)

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    np.seterr(all="ignore")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
