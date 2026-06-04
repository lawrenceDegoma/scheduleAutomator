"""
inference.py

Production inference interface — wraps all trained models into a single
SchedulingInference object that the scheduling engine calls at runtime.

Usage:
    inference = SchedulingInference.load("models/artifacts/")

    # Forecast demand for a given week
    demand_forecast = inference.forecast_demand(feature_rows_df)
    # -> DataFrame: [datetime, transactions, sales, required_headcount]

    # Score shift preferences for a list of assignments
    pref_scores = inference.score_preferences(assignment_features_df)
    # -> ndarray of shape (n,) in [0, 1]

    # Predict no-show risk
    risk_scores = inference.predict_noshow_risk(assignment_features_df)
    # -> ndarray of shape (n,) — probability of no-show per assignment
"""

import os
import pandas as pd
import numpy as np
from typing import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.demand_forecasting_model import DemandForecastingModel
from models.preference_model import PreferenceScoreModel, NoShowRiskModel
from features.feature_engineering import DemandFeatureBuilder, PreferenceFeatureBuilder


class SchedulingInference:
    """
    Unified inference interface for the AutomatedScheduling ML layer.

    This is the only class the scheduling-engine service needs to import.
    """

    def __init__(self,
                 demand_model: DemandForecastingModel,
                 pref_model: PreferenceScoreModel,
                 noshow_model: NoShowRiskModel):
        self._demand_model  = demand_model
        self._pref_model    = pref_model
        self._noshow_model  = noshow_model
        self._demand_builder = DemandFeatureBuilder()
        self._pref_builder   = PreferenceFeatureBuilder()

    # ------------------------------------------------------------------
    # Demand forecasting
    # ------------------------------------------------------------------
    def forecast_demand(self, raw_demand_df: pd.DataFrame) -> pd.DataFrame:
        """
        Accept raw demand rows (same schema as demand_history.csv) and
        return a forecast DataFrame.

        If the raw data has a 'datetime' column, it is carried through
        to the output for easy joining with the schedule.

        Returns
        -------
        DataFrame with columns:
            datetime, transactions, sales, required_headcount
        """
        feature_df = self._demand_builder.build(raw_demand_df)
        feature_cols = self._demand_builder.get_feature_columns()

        # Fill missing lags with available data (cold start case)
        feature_df = feature_df.fillna(method="bfill").fillna(0)

        X = feature_df[feature_cols]
        preds = self._demand_model.predict(X)

        if "datetime" in feature_df.columns:
            preds.insert(0, "datetime", feature_df["datetime"].values)

        return preds

    # ------------------------------------------------------------------
    # Preference scoring
    # ------------------------------------------------------------------
    def score_preferences(self,
                           feature_df: pd.DataFrame) -> np.ndarray:
        """
        Score a set of (employee, shift) candidate assignments.

        Parameters
        ----------
        feature_df : pre-built preference feature rows
                     (output of PreferenceFeatureBuilder.build())

        Returns
        -------
        ndarray of shape (n,) — preference score in [0, 1]
        """
        feature_cols = self._pref_builder.get_feature_columns()
        X = feature_df[feature_cols].fillna(0)
        return self._pref_model.predict(X)

    # ------------------------------------------------------------------
    # No-show risk
    # ------------------------------------------------------------------
    def predict_noshow_risk(self,
                             feature_df: pd.DataFrame) -> np.ndarray:
        """
        Predict no-show probability for a set of (employee, shift) pairs.

        Returns
        -------
        ndarray of shape (n,) — probability in [0, 1]
        """
        feature_cols = self._pref_builder.get_feature_columns()
        X = feature_df[feature_cols].fillna(0)
        return self._noshow_model.predict_proba(X)

    # ------------------------------------------------------------------
    # Convenience: combined scoring for optimizer
    # ------------------------------------------------------------------
    def score_assignments(self,
                           assignment_feature_df: pd.DataFrame
                           ) -> pd.DataFrame:
        """
        Returns a DataFrame with both preference_score and noshow_risk
        for each candidate assignment row. Designed to be passed directly
        into the scheduling optimizer as soft-constraint weights.

        Columns returned:
            preference_score  — how much the employee wants this shift [0,1]
            noshow_risk       — probability of no-show              [0,1]
            assignment_value  — composite score: pref * (1 - noshow_risk)
        """
        feature_cols = self._pref_builder.get_feature_columns()
        X = assignment_feature_df[feature_cols].fillna(0)

        pref_scores  = self._pref_model.predict(X)
        noshow_probs = self._noshow_model.predict_proba(X)

        result = assignment_feature_df[
            ["employee_id", "shift_type", "day_of_week", "shift_start_hour",
             "shift_end_hour"]
        ].copy()
        result["preference_score"]  = pref_scores
        result["noshow_risk"]       = noshow_probs
        result["assignment_value"]  = pref_scores * (1 - noshow_probs)
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, artifacts_dir: str) -> "SchedulingInference":
        """Load all models from a directory of .pkl artifacts."""
        demand_model = DemandForecastingModel.load(
            os.path.join(artifacts_dir, "demand_forecasting_model.pkl")
        )
        pref_model = PreferenceScoreModel.load(
            os.path.join(artifacts_dir, "preference_score_model.pkl")
        )
        noshow_model = NoShowRiskModel.load(
            os.path.join(artifacts_dir, "noshow_risk_model.pkl")
        )
        return cls(demand_model, pref_model, noshow_model)

    def save(self, artifacts_dir: str) -> None:
        """Convenience: save all underlying models."""
        os.makedirs(artifacts_dir, exist_ok=True)
        self._demand_model.save(
            os.path.join(artifacts_dir, "demand_forecasting_model.pkl")
        )
        self._pref_model.save(
            os.path.join(artifacts_dir, "preference_score_model.pkl")
        )
        self._noshow_model.save(
            os.path.join(artifacts_dir, "noshow_risk_model.pkl")
        )
