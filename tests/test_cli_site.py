"""The daily command (with the PSE download replaced by synthetic data) and the website build."""
import json
import re
from datetime import date, timedelta

import pandas as pd
import pytest

import bess_paper.data as data_mod
import bess_paper.live as live_mod
from bess_paper.cli import main
from bess_paper.live import planned_days, read_ledger
from bess_paper.site import build_site

from conftest import synthetic_prices, synthetic_system

FIRST = date(2025, 11, 1)
LAST = date(2026, 1, 31)


@pytest.fixture
def offline_market(monkeypatch, fast_model):
    prices = synthetic_prices(FIRST, LAST, seed=31)
    system = synthetic_system(FIRST, LAST, seed=32)

    def fake_prices(client, start, end):
        return prices[(prices["business_date"] >= start) & (prices["business_date"] <= end)].reset_index(drop=True)

    def fake_system(client, start, end):
        return system

    monkeypatch.setattr(data_mod, "load_prices", fake_prices)
    monkeypatch.setattr(data_mod, "load_system", fake_system)
    monkeypatch.setattr(live_mod, "MODEL", fast_model)
    real = live_mod.make_plan
    monkeypatch.setattr(live_mod, "make_plan", lambda p, s, d, now, params=fast_model: real(p, s, d, now, params))
    return prices


def run_daily(archive, when):
    return main(["--data-dir", str(archive / "cache"), "daily", "--archive-dir", str(archive), "--now", when,
                 "--offline"])


def test_daily_runs_plan_then_settle(tmp_path, offline_market):
    assert run_daily(tmp_path, "2026-01-10T09:30") == 0
    assert planned_days(tmp_path) == [date(2026, 1, 11)]
    assert run_daily(tmp_path, "2026-01-10T10:15") == 0          # second run the same morning: nothing new
    assert planned_days(tmp_path) == [date(2026, 1, 11)]
    assert run_daily(tmp_path, "2026-01-11T09:30") == 0          # next morning: settle the 11th, plan the 12th
    assert planned_days(tmp_path) == [date(2026, 1, 11), date(2026, 1, 12)]
    ledger = read_ledger(tmp_path)
    assert list(ledger["date"]) == ["2026-01-11"] and ledger["status"].iloc[0] == "traded"
    assert (tmp_path / "settled" / "2026" / "2026-01-11_prices.csv").exists()


def test_daily_after_the_gate_makes_no_plan(tmp_path, offline_market):
    assert run_daily(tmp_path, "2026-01-10T12:30") == 0
    assert planned_days(tmp_path) == []


def test_site_embeds_live_and_backtest(tmp_path, offline_market):
    archive, results, site = tmp_path / "archive", tmp_path / "results", tmp_path / "site"
    for when in ("2026-01-09T09:00", "2026-01-10T09:00", "2026-01-11T09:00"):
        run_daily(archive, when)
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
