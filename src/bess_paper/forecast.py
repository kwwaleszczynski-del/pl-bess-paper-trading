"""Price forecasters: quantile gradient boosting and three simple benchmarks.

The benchmarks need no fitting: `yesterday` repeats the prices of D-1 at the same local time, `last_week` those
of D-7 and `week_mean` averages D-1..D-7. The model is one `HistGradientBoostingRegressor` per quantile
(10%, 50%, 90%) on the as-of features; the median is the point forecast used for the schedule.

Usage:
    models = fit(X_train, y_train)
    q = predict(models, X_test)            # columns p10, p50, p90
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from bess_paper.config import MODEL, QH_FROM, ModelParams
from bess_paper.features import FEATURES

log = logging.getLogger(__name__)

BENCHMARKS = {"yesterday": "lag1", "last_week": "lag7", "week_mean": "lag_week"}


def qname(q: float) -> str:
    return f"p{round(q * 100):02d}"


def fit(X: pd.DataFrame, y: np.ndarray, params: ModelParams = MODEL) -> dict[float, HistGradientBoostingRegressor]:
    models = {}
    for q in params.quantiles:
        model = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_iter=params.max_iter, learning_rate=params.learning_rate,
            max_leaf_nodes=params.max_leaf_nodes, min_samples_leaf=params.min_samples_leaf,
            l2_regularization=params.l2_regularization, early_stopping=False, random_state=params.random_state)
        models[q] = model.fit(X[FEATURES], y)
    return models


def predict(models: dict[float, HistGradientBoostingRegressor], X: pd.DataFrame) -> pd.DataFrame:
    """Quantile forecasts, sorted per row so that they never cross."""
    qs = sorted(models)
    values = np.sort(np.column_stack([models[q].predict(X[FEATURES]) for q in qs]), axis=1)
    return pd.DataFrame(values, columns=[qname(q) for q in qs], index=X.index)


def train_mask(X: pd.DataFrame, y: np.ndarray, first_test_day: date, params: ModelParams = MODEL) -> np.ndarray:
    """Training rows for a model used from ``first_test_day`` on: the preceding ``train_days`` delivery days.

    The prices of the day before ``first_test_day`` are known at its decision (published on the day before that).
    """
    lo = first_test_day - timedelta(days=params.train_days)
    day = X["day"]
    return ((day >= lo) & (day < first_test_day)).to_numpy() & np.isfinite(y) & X["lag7"].notna().to_numpy()


def to_product(values, day: date) -> np.ndarray:
    """Average quarter-hour values within each hour on days with hourly products (before 2025-10-01)."""
    v = np.asarray(values, dtype=float)
    if day >= QH_FROM:
        return v
    return np.repeat(v.reshape(-1, 4).mean(axis=1), 4)
