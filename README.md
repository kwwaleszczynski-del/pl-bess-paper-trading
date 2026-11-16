# Track record of pl-bess-paper-trading

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
