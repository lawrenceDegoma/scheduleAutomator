"""
demand_forecasting_model.py

XGBoost-based demand forecasting model.

Predicts for each 30-minute slot:
  - transactions  (regression)
  - sales         (regression)
  - required_headcount (regression, rounded to int)

Usage:
    model = DemandForecastingModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    model.save("demand_model.pkl")

    # Later:
    model = DemandForecastingModel.load("demand_model.pkl")
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class DemandForecastingModel:
    """
    Multi-output XGBoost regression model for demand forecasting.

    One XGBRegressor is trained per target (transactions, sales,
    required_headcount) sharing the same feature matrix.

    Hyperparameters are tuned for coffee-shop transaction scale but are
    easily overridable via the `params` constructor argument.
    """

    TARGETS = ["transactions", "sales", "required_headcount"]

    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.models: dict[str, xgb.XGBRegressor] = {}
        self.feature_importances_: Optional[pd.DataFrame] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame,
            eval_set_frac: float = 0.10) -> "DemandForecastingModel":
        """
        Fit one XGBRegressor per target.

        Parameters
        ----------
        X : feature matrix (time-ordered)
        y : target DataFrame with columns matching TARGETS
        eval_set_frac : fraction of tail data used for early stopping
        """
        assert all(t in y.columns for t in self.TARGETS), (
            f"y must contain columns: {self.TARGETS}"
        )

        n_eval = max(1, int(len(X) * eval_set_frac))
        X_tr, X_val = X.iloc[:-n_eval], X.iloc[-n_eval:]

        importances = {}
        for target in self.TARGETS:
            y_tr = y[target].iloc[:-n_eval]
            y_val = y[target].iloc[-n_eval:]

            model = xgb.XGBRegressor(**self.params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.models[target] = model
            importances[target] = model.feature_importances_

        self.feature_importances_ = pd.DataFrame(
            importances, index=X.columns
        ).sort_values("transactions", ascending=False)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return predictions as a DataFrame with one column per target."""
        self._check_fitted()
        preds = {}
        for target in self.TARGETS:
            raw = self.models[target].predict(X)
            if target == "required_headcount":
                preds[target] = np.round(raw).astype(int).clip(min=1)
            elif target in ("transactions",):
                preds[target] = np.round(raw).astype(int).clip(min=0)
            else:
                preds[target] = np.round(raw, 2).clip(min=0)
        return pd.DataFrame(preds, index=X.index)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Return MAE, RMSE, R² per target."""
        self._check_fitted()
        preds = self.predict(X)
        rows = []
        for target in self.TARGETS:
            mae  = mean_absolute_error(y[target], preds[target])
            rmse = mean_squared_error(y[target], preds[target]) ** 0.5
            r2   = r2_score(y[target], preds[target])
            rows.append({"target": target, "MAE": mae, "RMSE": rmse, "R2": r2})
        return pd.DataFrame(rows).set_index("target").round(4)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    def cross_validate(self, X: pd.DataFrame, y: pd.DataFrame,
                       n_splits: int = 5) -> pd.DataFrame:
        """
        Time-series cross-validation.
        Returns per-fold MAE for each target.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = {t: [] for t in self.TARGETS}

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            fold_model = DemandForecastingModel(self.params)
            fold_model.fit(
                X_tr,
                y.iloc[tr_idx],
                eval_set_frac=0.0,
            )
            preds = fold_model.predict(X_val)
            for target in self.TARGETS:
                mae = mean_absolute_error(y[target].iloc[val_idx], preds[target])
                results[target].append(mae)
            print(f"  Fold {fold+1}/{n_splits} done.")

        return pd.DataFrame(results,
                            index=[f"fold_{i+1}" for i in range(n_splits)])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "DemandForecastingModel":
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
