"""Command line interface.

Usage:
    python -m bess_paper fetch                         # download/update the PSE cache in data/
    python -m bess_paper backtest --offline            # walk-forward backtest -> results/
    python -m bess_paper daily --archive-dir archive   # settle published days, plan tomorrow (before 12:00)
    python -m bess_paper site --archive-dir archive --out site
    python -m bess_paper figures                       # README figures -> docs/img/
    python -m bess_paper stats                         # comparisons of the backtest -> results/backtest_stats.json
    python -m bess_paper live-summary --archive-dir archive --update-readme README.md
    python -m bess_paper shadow --archive-dir archive  # backtest code re-run on the live days -> results/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from bess_paper.config import FIRST_DAY, MODEL, TZ, WARMUP_START

log = logging.getLogger("bess_paper")


def _date(text: str) -> date:
    return date.fromisoformat(text)


def _clock() -> datetime:
    return datetime.now(UTC)


def _now(text: str | None) -> datetime:
    if not text:
        return _clock()
    ts = pd.Timestamp(text)
    ts = ts.tz_localize(TZ) if ts.tzinfo is None else ts
    return ts.tz_convert("UTC").to_pydatetime()


def cmd_fetch(args) -> int:
    from bess_paper.data import load_prices, load_system, make_client
    client = make_client(args.data_dir)
    end = args.end or (datetime.now(UTC) + timedelta(days=1)).date()
    prices = load_prices(client, args.start, end)
    system = load_system(client, args.start, min(end, datetime.now(UTC).date()))
    log.info("prices: %d quarter-hours %s..%s; system: %d hours", len(prices), prices["business_date"].min(),
             prices["business_date"].max(), len(system))
    return 0


def cmd_backtest(args) -> int:
    from bess_paper.backtest import run_backtest, summary
    from bess_paper.data import load_prices, load_system, make_client
    client = make_client(args.data_dir, offline=args.offline)
    end = args.end or (datetime.now(UTC) - timedelta(days=1)).date()
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
    s.update(generated_utc=datetime.now(UTC).isoformat(timespec="seconds"),
             data_until=str(end), model={k: getattr(MODEL, k) for k in MODEL.__dataclass_fields__})
    (out / "backtest_summary.json").write_text(json.dumps(s, indent=1, default=float) + "\n", encoding="utf-8",
                                               newline="\n")
    log.info("backtest %s..%s: model %.0f PLN (%.0f%% of perfect foresight), yesterday %.0f, fixed %.0f",
             s["first_day"], s["last_day"], s["pnl_model"], 100 * s["capture_model"], s["pnl_yesterday"],
             s["pnl_fixed"])
    return 0


def cmd_daily(args) -> int:
    from bess_paper.data import load_prices, load_system, make_client
    from bess_paper.live import (
        LateError,
        ensure_readme,
        gate_closure,
        load_plan,
        local_time,
        make_plan,
        run_context,
        settle_pending,
        write_plan,
        write_snapshot,
    )
    if args.now is not None and os.environ.get("GITHUB_ACTIONS") == "true":
        log.error("--now replays a past moment and is refused in GitHub Actions, where the live record is written")
        return 2
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
        plan = make_plan(prices, system, delivery, now, run=run_context(replay=args.now is not None))
    except LateError as e:
        log.warning("no plan for %s: %s", delivery, e)
        return 0
    # the plan is issued when it is finished, which must still be before the gate (the push step checks once more)
    finished = now if args.now is not None else _clock()
    if finished >= gate_closure(delivery):
        log.warning("no plan for %s: finished at %s UTC, after the gate closure", delivery, f"{finished:%H:%M}")
        return 0
    plan["issued_utc"] = finished.isoformat(timespec="seconds")
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


def cmd_prepush(args) -> int:
    from bess_paper.live import drop_late_plans
    archive = Path(args.archive_dir)
    out = subprocess.run(["git", "-C", str(archive), "ls-files", "--others", "--exclude-standard", "plans"],
                         capture_output=True, text=True)
    if out.returncode == 0:
        new = [archive / line for line in out.stdout.splitlines() if line]
    else:                       # not a repository yet (first push): every plan is new
        new = sorted((archive / "plans").glob("*/*.json"))
    dropped = drop_late_plans(archive, new, _clock())
    log.info("%d new plan file(s), %d late file(s) removed", len(new), len(dropped))
    return 0


def cmd_live_summary(args) -> int:
    from bess_paper.live import live_markdown, live_summary, read_ledger
    summary = live_summary(read_ledger(Path(args.archive_dir)), Path(args.archive_dir))
    text = live_markdown(summary)
    if args.update_readme:
        path = Path(args.update_readme)
        readme = path.read_text(encoding="utf-8")
        start, end = "<!-- live-summary:start -->", "<!-- live-summary:end -->"
        if start not in readme or end not in readme:
            log.error("%s has no %s ... %s block", path, start, end)
            return 2
        head, rest = readme.split(start, 1)
        path.write_text(f"{head}{start}\n{text}{end}{rest.split(end, 1)[1]}", encoding="utf-8", newline="\n")
        log.info("live summary written to %s", path)
    else:
        print(text)
    return 0


def cmd_stats(args) -> int:
    from bess_paper.backtest import comparisons
    out = Path(args.results_dir)
    daily = pd.read_csv(out / "backtest_daily.csv", dtype={"date": str}).set_index("date")
    stats = comparisons(daily)
    (out / "backtest_stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8", newline="\n")
    m = stats["model_vs"]["yesterday"]
    log.info("model vs yesterday's prices: %+.0f PLN/MW/year (95 %% interval %+.0f..%+.0f)", m["pln_per_mw_year"],
             *m["ci95_pln_per_mw_year"])
    return 0


def cmd_shadow(args) -> int:
    from bess_paper.data import load_prices, load_system, make_client
    from bess_paper.live import read_ledger
    from bess_paper.shadow import shadow_live
    archive, out = Path(args.archive_dir), Path(args.results_dir)
    days = sorted(date.fromisoformat(d) for d in read_ledger(archive)["date"])
    if not days:
        log.error("the ledger in %s is empty", archive)
        return 2
    client = make_client(args.data_dir, offline=args.offline)
    start = days[0] - timedelta(days=MODEL.train_days + 10)
    prices = load_prices(client, start, days[-1])
    system = load_system(client, start, days[-1])
    per_day, summary = shadow_live(archive, prices, system)
    out.mkdir(parents=True, exist_ok=True)
    per_day.to_csv(out / "shadow_live.csv", index=False, float_format="%.4f", lineterminator="\n")
    (out / "shadow_live.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    log.info("shadow run of %d live days written to %s", len(per_day), out)
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
    p.add_argument("--now", help="replay this moment (ISO, Polish time if no offset); for tests only: the plan is "
                                 "marked as a replay, never counts as a trade and the option is refused in Actions")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_daily)

    p = sub.add_parser("prepush", help="drop new plans whose gate has closed (run just before pushing the archive)")
    p.add_argument("--archive-dir", default="archive")
    p.set_defaults(func=cmd_prepush)

    p = sub.add_parser("live-summary", help="live record from the ledger as a Markdown table")
    p.add_argument("--archive-dir", default="archive")
    p.add_argument("--update-readme", help="replace the live-summary block of this README")
    p.set_defaults(func=cmd_live_summary)

    p = sub.add_parser("stats", help="paired comparisons of the backtest strategies (bootstrap over weeks)")
    p.add_argument("--results-dir", default="results")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("shadow", help="re-run the forecast on the live days with today's data (daily and weekly refit)")
    p.add_argument("--archive-dir", default="archive")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_shadow)

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
