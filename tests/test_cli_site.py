"""The daily command (with the PSE download replaced by synthetic data) and the website build."""
import json
import re
from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

import bess_paper.cli as cli
import bess_paper.data as data_mod
import bess_paper.live as live_mod
from bess_paper.cli import main
from bess_paper.live import load_plan, local_time, planned_days, read_ledger
from bess_paper.site import build_site

from conftest import synthetic_prices, synthetic_system

FIRST = date(2025, 11, 1)
LAST = date(2026, 1, 31)


class Clock:
    """The time the daily command sees; ``times`` can queue several readings (start, finish)."""
    times: list[datetime] = []

    def __call__(self) -> datetime:
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


@pytest.fixture
def clock(monkeypatch):
    for name in ("GITHUB_ACTIONS", "GITHUB_EVENT_NAME", "GITHUB_RUN_ID"):    # CI itself runs in GitHub Actions
        monkeypatch.delenv(name, raising=False)
    c = Clock()
    monkeypatch.setattr(cli, "_clock", c)
    return c


@pytest.fixture
def offline_market(monkeypatch, fast_model, clock):
    prices = synthetic_prices(FIRST, LAST, seed=31)
    system = synthetic_system(FIRST, LAST, seed=32)

    def fake_prices(client, start, end):
        # what PSE has published at the clock's time (prices of day X from 15:00 on X-1)
        p = prices[(prices["business_date"] >= start) & (prices["business_date"] <= end)]
        seen = [local_time(d - timedelta(days=1), time(15, 0)) <= clock.times[0] for d in p["business_date"]]
        return p[seen].reset_index(drop=True)

    def fake_system(client, start, end):
        return system[system.index + pd.Timedelta(hours=1) <= clock.times[0]]

    monkeypatch.setattr(data_mod, "load_prices", fake_prices)
    monkeypatch.setattr(data_mod, "load_system", fake_system)
    monkeypatch.setattr(live_mod, "MODEL", fast_model)
    real = live_mod.make_plan
    monkeypatch.setattr(live_mod, "make_plan",
                        lambda p, s, d, now, params=fast_model, run=None: real(p, s, d, now, params, run=run))
    return prices


def at(when: str) -> datetime:
    return pd.Timestamp(when).tz_localize("Europe/Warsaw").tz_convert("UTC").to_pydatetime()


def run_live(archive, clock, *when):
    """A run in real time: ``when`` is the start (and, if given, the finish) in Polish time."""
    clock.times = [at(w) for w in when]
    return main(["--data-dir", str(archive / "cache"), "daily", "--archive-dir", str(archive), "--offline"])


def run_replay(archive, clock, when):
    clock.times = [at(when)]
    return main(["--data-dir", str(archive / "cache"), "daily", "--archive-dir", str(archive), "--now", when,
                 "--offline"])


def test_daily_runs_plan_then_settle(tmp_path, offline_market, clock):
    assert run_live(tmp_path, clock, "2026-01-10T09:30") == 0
    assert planned_days(tmp_path) == [date(2026, 1, 11)]
    plan = load_plan(tmp_path, date(2026, 1, 11))
    assert plan["run"]["trigger"] == "local" and plan["issued_utc"] == "2026-01-10T08:30:00+00:00"
    assert run_live(tmp_path, clock, "2026-01-10T10:15") == 0          # second run the same morning: nothing new
    assert planned_days(tmp_path) == [date(2026, 1, 11)]
    assert run_live(tmp_path, clock, "2026-01-11T09:30") == 0          # next morning: settle the 11th, plan the 12th
    assert planned_days(tmp_path) == [date(2026, 1, 11), date(2026, 1, 12)]
    ledger = read_ledger(tmp_path)
    assert list(ledger["date"]) == ["2026-01-11"] and ledger["status"].iloc[0] == "traded"
    assert (tmp_path / "settled" / "2026" / "2026-01-11_prices.csv").exists()


def test_daily_after_the_gate_makes_no_plan(tmp_path, offline_market, clock):
    assert run_live(tmp_path, clock, "2026-01-10T12:30") == 0
    assert planned_days(tmp_path) == []


def test_a_plan_finished_after_the_gate_is_not_written(tmp_path, offline_market, clock):
    assert run_live(tmp_path, clock, "2026-01-10T11:59", "2026-01-10T12:01") == 0
    assert planned_days(tmp_path) == []


def test_replayed_plans_are_marked_and_never_count(tmp_path, offline_market, clock):
    assert run_replay(tmp_path, clock, "2026-01-10T09:30") == 0
    plan = load_plan(tmp_path, date(2026, 1, 11))
    assert plan["run"] == {"trigger": "replay"}
    assert run_replay(tmp_path, clock, "2026-01-11T09:30") == 0
    row = read_ledger(tmp_path).iloc[0]
    assert row["status"] == "replay" and row["pnl_model"] == 0 and row["pnl_perfect"] > 0


def test_now_is_refused_in_github_actions(tmp_path, offline_market, clock, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert run_replay(tmp_path, clock, "2026-01-10T09:30") == 2
    assert planned_days(tmp_path) == []


def test_plans_record_the_github_run(tmp_path, offline_market, clock, monkeypatch):
    for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_RUN_ID": "123",
                 "GITHUB_RUN_ATTEMPT": "1", "GITHUB_REPOSITORY": "owner/repo", "GITHUB_ACTOR": "someone",
                 "GITHUB_SERVER_URL": "https://github.com"}.items():
        monkeypatch.setenv(k, v)
    assert run_live(tmp_path, clock, "2026-01-10T09:30") == 0
    run = load_plan(tmp_path, date(2026, 1, 11))["run"]
    assert run["trigger"] == "workflow_dispatch" and run["run_url"] == "https://github.com/owner/repo/actions/runs/123"


def test_prepush_removes_new_plans_whose_gate_has_closed(tmp_path, offline_market, clock):
    assert run_live(tmp_path, clock, "2026-01-10T09:30") == 0
    assert (tmp_path / "inputs" / "2026" / "2026-01-11_features.csv").exists()
    clock.times = [at("2026-01-10T11:00")]
    assert main(["prepush", "--archive-dir", str(tmp_path)]) == 0
    assert planned_days(tmp_path) == [date(2026, 1, 11)]           # still before the gate: kept
    clock.times = [at("2026-01-10T12:00")]
    assert main(["prepush", "--archive-dir", str(tmp_path)]) == 0
    assert planned_days(tmp_path) == []
    assert not (tmp_path / "inputs" / "2026" / "2026-01-11_features.csv").exists()


def test_site_embeds_live_and_backtest(tmp_path, offline_market, clock):
    archive, results, site = tmp_path / "archive", tmp_path / "results", tmp_path / "site"
    for when in ("2026-01-09T09:00", "2026-01-10T09:00", "2026-01-11T09:00"):
        run_live(archive, clock, when)
    results.mkdir()
    days = [date(2025, 12, 1) + timedelta(days=i) for i in range(10)]
    daily = pd.DataFrame({"date": [d.isoformat() for d in days]})
    for s in ("model", "perfect", "yesterday", "fixed", "last_week", "week_mean"):
        daily[f"pnl_pln_{s}"] = range(10)
        daily[f"cycles_{s}"] = 1.0
    for f in ("model", "yesterday", "last_week", "week_mean"):
        daily[f"mae_{f}"], daily[f"rmse_{f}"], daily[f"rank_{f}"] = 50.0, 70.0, 0.8
    daily["cover80"], daily["spread"] = 0.8, 300.0
    daily.to_csv(results / "backtest_daily.csv", index=False)
    (results / "backtest_summary.json").write_text(json.dumps({"days": 10, "first_day": "2025-12-01",
                                                               "last_day": "2025-12-10"}))
    page = build_site(results, archive, site).read_text(encoding="utf-8")
    payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1))
    assert payload["live"]["start"] == "2026-01-10"
    assert payload["live"]["ledger"]["date"] == ["2026-01-10", "2026-01-11"]
    assert payload["live"]["next_plan"]["delivery_date"] == "2026-01-12"
    assert "actual" in payload["live"]["last_settled"]
    assert payload["backtest"]["table"]["all"]["days"] == 10
    assert "Paper trading only" in page
    assert (site / "ledger.csv").exists() and (site / "backtest_daily.csv").exists()
    summary = json.loads((site / "live_summary.json").read_text(encoding="utf-8"))
    assert summary["days"] == 2 and summary["traded"] == 2 and summary["badge"].startswith("2 of 2 days traded")
    assert summary["plan_triggers"] == {"local": 2}


def test_live_summary_block_in_readme(tmp_path, offline_market, clock):
    for when in ("2026-01-09T09:00", "2026-01-11T09:00"):         # no run on the 10th: the 11th has no plan
        run_live(tmp_path, clock, when)
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- live-summary:start -->\nold\n<!-- live-summary:end -->\nrest\n", encoding="utf-8")
    assert main(["live-summary", "--archive-dir", str(tmp_path), "--update-readme", str(readme)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and text.startswith("intro\n") and text.endswith("rest\n")
    assert "2 settled delivery days, 1 with a plan published before the gate, 1 without" in text
    assert "| days with a plan | 1 |" in text
