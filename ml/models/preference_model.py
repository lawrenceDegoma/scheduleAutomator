"""
preference_model.py

Barista preferred-hours learning model.

Learns latent shift preferences per employee from historical schedule data.
Two sub-models:

  1. PreferenceScoreModel  — predicts a 0–1 preference score for any
                             (employee, shift_type, day) combination.
                             Used by the optimizer as a soft-constraint weight.

  2. NoShowRiskModel       — predicts the probability that a specific
                             employee no-shows on a given shift assignment.
                             Used by the optimizer to hedge coverage.

Both use XGBoost (classifier for no-show, regressor for preference score).

Usage:
    pref_model = PreferenceScoreModel()
    pref_model.fit(X_train, y_train)
    scores = pref_model.predict(X_new)
    pref_model.save("preference_model.pkl")

    risk_model = NoShowRiskModel()
    risk_model.fit(X_train, y_binary_train)
    probs = risk_model.predict_proba(X_new)
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Optional

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (mean_absolute_error, r2_score,
                              roc_auc_score, classification_report)
import xgboost as xgb


# ===========================================================================
# 1. PREFERENCE SCORE MODEL
# ===========================================================================

class PreferenceScoreModel:
    """
    Predicts preference_score (0.0–1.0) for a (employee, shift) pair.

    Higher score → employee more likely to work this shift happily.
    The optimizer uses these scores as soft-constraint weights.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 10,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_columns_: Optional[list] = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PreferenceScoreModel":
        self.feature_columns_ = list(X.columns)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        raw = self.model.predict(X)
        return np.clip(raw, 0.0, 1.0)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self._check_fitted()
        preds = self.predict(X)
        return {
            "MAE":  round(mean_absolute_error(y, preds), 4),
            "R2":   round(r2_score(y, preds), 4),
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       cv: int = 5) -> dict:
        model = xgb.XGBRegressor(**self.params)
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="neg_mean_absolute_error")
        return {
            "cv_mae_mean": round(-scores.mean(), 4),
            "cv_mae_std":  round(scores.std(), 4),
        }

    def get_feature_importance(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns_,
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"PreferenceScoreModel saved → {path}")

    @classmethod
    def load(cls, path: str) -> "PreferenceScoreModel":
        return joblib.load(path)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")


# ===========================================================================
# 2. NO-SHOW RISK MODEL
# ===========================================================================

class NoShowRiskModel:
    """
    Binary classifier: predicts probability of no-show for a given assignment.

    Output: float in [0, 1] — probability the employee does not show up.

    The optimizer can use this to:
      - Add buffer coverage on high-risk assignments
      - Prefer reliable employees for critical roles (opener, supervisor)
    """

    DEFAULT_PARAMS = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "scale_pos_weight": 5,   # class imbalance: no-shows are rare
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns_: Optional[list] = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NoShowRiskModel":
        """y must be binary: 1 = no-show, 0 = showed up."""
        self.feature_columns_ = list(X.columns)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probability of no-show (class 1)."""
        self._check_fitted()
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Returns binary prediction."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self._check_fitted()
        probs = self.predict_proba(X)
        preds = self.predict(X)
        auc = roc_auc_score(y, probs) if y.nunique() > 1 else float("nan")
        report = classification_report(y, preds, output_dict=True)
        return {
            "ROC_AUC": round(auc, 4),
            "precision_noshow": round(report.get("1", {}).get("precision", 0), 4),
            "recall_noshow":    round(report.get("1", {}).get("recall", 0), 4),
        }

    def get_feature_importance(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            self.model.feature_importances_,
            index=self.feature_columns_,
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        self._check_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"NoShowRiskModel saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NoShowRiskModel":
        return joblib.load(path)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
