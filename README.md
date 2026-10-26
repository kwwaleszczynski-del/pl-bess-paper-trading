# Track record of pl-bess-paper-trading

Written only by the daily GitHub Actions run of the main branch. Paper trading: no orders are sent to any market.

- `plans/YYYY/YYYY-MM-DD.json`: the plan for delivery day YYYY-MM-DD (price forecast and schedule), committed
  before 12:00 Polish time on the day before. A plan is never replaced.
- `inputs/YYYY/`: the model inputs at the decision (`_features.csv`) and the PSE 5-day coordination plan as
  published at that moment (`_pk5l.json`).
- `settled/YYYY/YYYY-MM-DD_prices.csv`: the SDAC clearing prices the day was settled on.
- `ledger.csv`: one row per settled delivery day, appended once and never rewritten. P&L in PLN for a
  1 MW / 2 MWh battery, net of a degradation cost of 30 PLN/MWh discharged.

The commit history of this branch is the audit trail: commit times show when each plan was published.
