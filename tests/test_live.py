"""Live paper trading: gate closure, as-of inputs, plan archive and the append-only ledger."""
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bess_paper.config import FUND_CUTOFF_LOCAL
from bess_paper.live import (LateError, append_ledger, gate_closure, load_plan, local_time, make_plan, read_ledger,
                             settle_pending, write_plan)
from bess_paper.optimizer import Plan, check_plan, settle, soc_path
from bess_paper.strategies import fixed_plan, plan_day

from conftest import synthetic_prices, synthetic_system

START = date(2025, 11, 1)
TODAY = date(2026, 1, 20)


@pytest.fixture(scope="module")
def market():
    prices = synthetic_prices(START, TODAY + timedelta(days=3), seed=11)
    system = synthetic_system(START, TODAY + timedelta(days=3), seed=12)
    return prices, system


def known_at(prices, system, now):
    """What PSE has published by ``now``: prices of days up to today, system data up to now."""
    today = pd.Timestamp(now).tz_convert("Europe/Warsaw").date()
    return prices[prices["business_date"] <= today], system[system.index + pd.Timedelta(hours=1) <= now]


def morning(day: date, hour: int = 9) -> datetime:
    return local_time(day, datetime.min.time().replace(hour=hour))


def test_gate_closure_is_noon_polish_time_on_the_day_before():
    assert gate_closure(date(2026, 1, 21)) == datetime(2026, 1, 20, 11, 0, tzinfo=timezone.utc)   # CET
    assert gate_closure(date(2026, 7, 21)) == datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)   # CEST


def test_no_plan_after_the_gate_closure(market, fast_model):
    prices, system = market
    late = local_time(TODAY, datetime.min.time().replace(hour=12, minute=1))
    with pytest.raises(LateError):
        make_plan(*known_at(prices, system, late), TODAY + timedelta(days=1), late, fast_model)


def test_no_plan_when_the_prices_are_already_published(market, fast_model):
    prices, system = market
    with pytest.raises(LateError):
        make_plan(prices, system, TODAY + timedelta(days=1), morning(TODAY), fast_model)


def test_plan_ignores_data_after_the_decision(market, fast_model):
    prices, system = market
    now = morning(TODAY, 11)
    p, s = known_at(prices, system, now)
    plan_a = make_plan(p, s, TODAY + timedelta(days=1), now, fast_model)
    # system data between 06:00 and the run, which a late morning run would see, must not matter
    s2 = s.copy()
    cutoff = pd.Timestamp(local_time(TODAY, FUND_CUTOFF_LOCAL))
    s2.loc[s2.index >= cutoff, "wind"] = 1e6
    plan_b = make_plan(p, s2, TODAY + timedelta(days=1), now, fast_model)
    assert plan_a["intervals"] == plan_b["intervals"]


def test_plan_is_feasible_and_complete(market, fast_model):
    prices, system = market
    now = morning(TODAY)
    plan = make_plan(*known_at(prices, system, now), TODAY + timedelta(days=1), now, fast_model)
    iv = pd.DataFrame(plan["intervals"])
    assert len(iv) == 96 and (iv["p10"] <= iv["p50"]).all() and (iv["p50"] <= iv["p90"]).all()
    c, d = iv["charge_mw"].to_numpy(), iv["discharge_mw"].to_numpy()
    check_plan(Plan(c, d, soc_path(c, d)), tol=1e-5)
    assert plan["inputs"]["last_price_day"] == TODAY.isoformat()
    assert datetime.fromisoformat(plan["issued_utc"]) < datetime.fromisoformat(plan["gate_closure_utc"])


def test_a_plan_is_never_overwritten(tmp_path, market, fast_model):
    prices, system = market
    now = morning(TODAY)
    plan = make_plan(*known_at(prices, system, now), TODAY + timedelta(days=1), now, fast_model)
    write_plan(tmp_path, plan)
    with pytest.raises(FileExistsError):
        write_plan(tmp_path, plan)
    assert load_plan(tmp_path, TODAY + timedelta(days=1))["issued_utc"] == plan["issued_utc"]
    assert (tmp_path / "inputs" / "2026" / "2026-01-21_features.csv").exists()


def _archive_with_plans(tmp_path, prices, system, days, fast_model):
    for day in days:
        now = morning(day - timedelta(days=1))
        write_plan(tmp_path, make_plan(*known_at(prices, system, now), day, now, fast_model))


def test_settlement_uses_the_archived_plan_and_published_prices(tmp_path, market, fast_model):
    prices, system = market
    day = TODAY - timedelta(days=2)
    _archive_with_plans(tmp_path, prices, system, [day], fast_model)
    rows = settle_pending(tmp_path, prices[prices["business_date"] <= day])
    assert [r["date"] for r in rows] == [day.isoformat()]
    plan = load_plan(tmp_path, day)
    iv = pd.DataFrame(plan["intervals"])
    actual = prices.loc[prices["business_date"] == day, "price"].to_numpy()
    c, d = iv["charge_mw"].to_numpy(), iv["discharge_mw"].to_numpy()
    expected = settle(Plan(c, d, soc_path(c, d)), actual)["pnl_pln"]
    assert rows[0]["pnl_model"] == pytest.approx(expected)
    assert rows[0]["pnl_perfect"] >= rows[0]["pnl_model"] - 1e-6
    assert rows[0]["pnl_fixed"] == pytest.approx(settle(fixed_plan(day), actual)["pnl_pln"])
    yest = prices.loc[prices["business_date"] == day - timedelta(days=1), "price"].to_numpy()
    assert rows[0]["pnl_yesterday"] == pytest.approx(settle(plan_day(yest, day), actual)["pnl_pln"])
    assert rows[0]["mae_model"] == pytest.approx(np.abs(iv["p50"].to_numpy() - actual).mean(), rel=1e-6)


def test_ledger_is_append_only_and_never_recomputed(tmp_path, market, fast_model):
    prices, system = market
    days = [TODAY - timedelta(days=3), TODAY - timedelta(days=2)]
    _archive_with_plans(tmp_path, prices, system, days, fast_model)
    first = settle_pending(tmp_path, prices[prices["business_date"] <= days[0]])
    assert len(first) == 1
    text_before = (tmp_path / "ledger.csv").read_text(encoding="utf-8")

    # prices of the settled day change afterwards (e.g. a correction): the settled row must stay as it was
    changed = prices.copy()
    changed.loc[changed["business_date"] == days[0], "price"] += 500.0
    second = settle_pending(tmp_path, changed[changed["business_date"] <= days[1]])
    assert [r["date"] for r in second] == [days[1].isoformat()]
    text_after = (tmp_path / "ledger.csv").read_text(encoding="utf-8")
    assert text_after.startswith(text_before)
    assert settle_pending(tmp_path, changed[changed["business_date"] <= days[1]]) == []
    assert (tmp_path / "ledger.csv").read_text(encoding="utf-8") == text_after
    with pytest.raises(ValueError):
        append_ledger(tmp_path, [{"date": days[0].isoformat(), "status": "traded"}])


def test_a_day_without_a_plan_counts_as_no_trade(tmp_path, market, fast_model):
    prices, system = market
    first, missing = TODAY - timedelta(days=4), TODAY - timedelta(days=3)
    _archive_with_plans(tmp_path, prices, system, [first], fast_model)
    rows = settle_pending(tmp_path, prices[prices["business_date"] <= missing])
    by_day = {r["date"]: r for r in rows}
    assert by_day[missing.isoformat()]["status"] == "no_plan"
    assert by_day[missing.isoformat()]["pnl_model"] == 0.0
    assert by_day[missing.isoformat()]["pnl_perfect"] > 0
    ledger = read_ledger(tmp_path)
    assert list(ledger["date"]) == [first.isoformat(), missing.isoformat()]


def test_plan_file_is_plain_json(tmp_path, market, fast_model):
    prices, system = market
    _archive_with_plans(tmp_path, prices, system, [TODAY], fast_model)
    body = json.loads((tmp_path / "plans" / "2026" / f"{TODAY.isoformat()}.json").read_text(encoding="utf-8"))
    assert {"delivery_date", "issued_utc", "gate_closure_utc", "code_version", "intervals"} <= set(body)
    assert "_features" not in body
