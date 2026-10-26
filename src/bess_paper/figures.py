"""README figures from the backtest results (matplotlib, light theme).

Usage:
    write_figures(Path("results"), Path("docs/img"))
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bess_paper.config import TZ  # noqa: E402

log = logging.getLogger(__name__)

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"model": "#2a78d6", "yesterday": "#eb6834", "fixed": "#1baf7a", "perfect": INK2}
NAMES = {"model": "Forecast + MILP", "perfect": "Perfect foresight (upper bound)", "yesterday": "Yesterday's prices",
         "fixed": "Fixed hours"}


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK2)


def cumulative(daily: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    x = pd.to_datetime(daily.index)
    for s in ("perfect", "model", "yesterday", "fixed"):
        y = daily[f"pnl_pln_{s}"].cumsum() / 1000
        ax.plot(x, y, color=COLORS[s], lw=2, ls="--" if s == "perfect" else "-", label=NAMES[s])
        ax.annotate(f"{y.iloc[-1]:,.0f}k", (x[-1], y.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_ylabel("cumulative P&L, thousand PLN")
    ax.axvline(pd.Timestamp("2025-10-01"), color=AXIS, lw=1)
    ax.text(pd.Timestamp("2025-10-04"), ax.get_ylim()[1] * 0.97, "15-minute products", fontsize=8.5, color=MUTED,
            va="top")
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    ax.margins(x=0.06)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def monthly_capture(daily: pd.DataFrame, path: Path) -> None:
    m = daily.groupby(daily.index.str[:7])[[f"pnl_pln_{s}" for s in COLORS]].sum()
    x = pd.to_datetime(m.index + "-15")
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=130)
    for s in ("model", "yesterday", "fixed"):
        ax.plot(x, 100 * m[f"pnl_pln_{s}"] / m["pnl_pln_perfect"], color=COLORS[s], lw=2, marker="o", ms=4.5,
                mec="white", mew=1.2, label=NAMES[s])
    ax.axhline(100, color=INK2, lw=1, ls="--")
    ax.set_ylabel("% of perfect foresight")
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def example_day(intervals: pd.DataFrame, day: str, path: Path) -> None:
    d = intervals[intervals["day"] == day]
    t = pd.to_datetime(d["start_utc"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), dpi=130, sharex=True, height_ratios=[3, 2])
    a1.fill_between(t, d["p10"], d["p90"], step="post", color=COLORS["model"], alpha=0.15, lw=0,
                    label="10-90 % forecast range")
    a1.step(t, d["p50"], where="post", color=COLORS["model"], lw=2, label="median forecast")
    a1.step(t, d["actual"], where="post", color=INK, lw=1.6, label="SDAC clearing price")
    a1.set_ylabel("PLN/MWh")
    a1.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK2)
    net = d["net_model"].to_numpy()
    a2.bar(t, net, width=1 / 96 * 0.9, align="edge", color=[COLORS["model"] if v >= 0 else "#e34948" for v in net],
           label="plan: sell (+) / buy (-)")
    a2.step(t, d["net_perfect"], where="post", color=INK2, lw=1.2, ls=":", label="perfect-foresight plan")
    a2.set_ylabel("MW")
    a2.set_ylim(-1.15, 1.15)
    a2.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK2)
    for ax in (a1, a2):
        _style(ax)
    a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    fig.suptitle(f"Backtest day {day}", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_figures(results: Path, out: Path, day: str | None = None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(results / "backtest_daily.csv", dtype={"date": str}).set_index("date")
    paths = [out / "backtest_cumulative.png", out / "monthly_capture.png"]
    cumulative(daily, paths[0])
    monthly_capture(daily, paths[1])
    intervals = pd.read_parquet(results / "backtest_intervals.parquet")
    if day is None:
        recent = daily[daily.index >= "2025-10-01"]
        day = (recent["pnl_pln_perfect"] - recent["pnl_pln_model"]).abs().sort_values().index[len(recent) // 2]
    paths.append(out / "example_day.png")
    example_day(intervals, day, paths[-1])
    log.info("figures written to %s", out)
    return paths
