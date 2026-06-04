"""
training_pipeline.py

End-to-end training orchestrator.

Steps:
  1. Load raw CSV data (or accept DataFrames directly)
  2. Run feature engineering for both pipelines
  3. Train & evaluate the DemandForecastingModel
  4. Train & evaluate the PreferenceScoreModel
  5. Train & evaluate the NoShowRiskModel
  6. Save all artifacts to the given output directory

Can be run as a script:
    python training_pipeline.py --data-dir data/ --output-dir models/artifacts/

Or called programmatically:
    pipeline = TrainingPipeline(data_dir="data/")
    pipeline.run()
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Adjust these imports when running as part of the package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.feature_engineering import DemandFeatureBuilder, PreferenceFeatureBuilder
from models.demand_forecasting_model import DemandForecastingModel
from models.preference_model import PreferenceScoreModel, NoShowRiskModel


class TrainingPipeline:
    """
    Orchestrates the full ML training workflow.

    Parameters
    ----------
    data_dir    : directory containing demand_history.csv, employees.csv,
                  schedule_history.csv
    output_dir  : where to persist trained model artifacts
    run_cv      : whether to run cross-validation (slower but informative)
    """

    def __init__(self,
                 data_dir: str = "data",
                 output_dir: str = "models/artifacts",
                 run_cv: bool = False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.run_cv = run_cv
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data(self):
        demand_path = os.path.join(self.data_dir, "demand_history.csv")
        emp_path    = os.path.join(self.data_dir, "employees.csv")
        sched_path  = os.path.join(self.data_dir, "schedule_history.csv")

        print(f"Loading demand data from {demand_path} ...")
        demand_df = pd.read_csv(demand_path, parse_dates=["datetime", "date"])

        print(f"Loading employee data from {emp_path} ...")
        emp_df = pd.read_csv(emp_path)

        print(f"Loading schedule data from {sched_path} ...")
        sched_df = pd.read_csv(sched_path, parse_dates=["date", "week_start"])

        return demand_df, emp_df, sched_df

    # ------------------------------------------------------------------
    # Demand forecasting pipeline
    # ------------------------------------------------------------------
    def _run_demand_pipeline(self, demand_df: pd.DataFrame):
        print("\n── Demand Forecasting Pipeline ──────────────────────────")
        builder = DemandFeatureBuilder()
        demand_df = builder.build(demand_df)

        feature_cols = builder.get_feature_columns()
        target_cols  = builder.get_target_columns()

        # Drop rows missing lag features (first few weeks)
        demand_df = demand_df.dropna(subset=feature_cols).reset_index(drop=True)

        X = demand_df[feature_cols]
        y = demand_df[target_cols]

        # Temporal split: last 8 weeks as hold-out test set
        # (roughly 8 * 7 * 34 = ~1904 rows)
        HOLD_OUT_ROWS = 8 * 7 * 34
        X_train, X_test = X.iloc[:-HOLD_OUT_ROWS], X.iloc[-HOLD_OUT_ROWS:]
        y_train, y_test = y.iloc[:-HOLD_OUT_ROWS], y.iloc[-HOLD_OUT_ROWS:]

        print(f"  Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

        model = DemandForecastingModel()

        if self.run_cv:
            print("  Running 5-fold time-series CV ...")
            cv_results = model.cross_validate(X_train, y_train, n_splits=5)
            print("  CV MAE per fold:")
            print(cv_results.to_string())

        print("  Training final model ...")
        model.fit(X_train, y_train)

        print("  Evaluation on hold-out test set:")
        metrics = model.evaluate(X_test, y_test)
        print(metrics.to_string())

        print("\n  Top 10 Feature Importances (by transactions):")
        print(model.feature_importances_.head(10).to_string())

        path = os.path.join(self.output_dir, "demand_forecasting_model.pkl")
        model.save(path)
        return model, metrics

    # ------------------------------------------------------------------
    # Preference & no-show pipeline
    # ------------------------------------------------------------------
    def _run_preference_pipeline(self, sched_df: pd.DataFrame,
                                  emp_df: pd.DataFrame):
        print("\n── Preference & No-Show Pipeline ────────────────────────")
        builder = PreferenceFeatureBuilder()
        feature_df = builder.build(sched_df, emp_df)

        feature_cols = builder.get_feature_columns()
        target_col   = builder.get_target_column()

        # Drop nulls
        feature_df = feature_df.dropna(
            subset=feature_cols + [target_col, "no_show"]
        ).reset_index(drop=True)

        X = feature_df[feature_cols]
        y_pref    = feature_df[target_col]
        y_noshow  = feature_df["no_show"].astype(int)

        # Stratified split (no-show labels are sparse)
        X_tr, X_te, yp_tr, yp_te, yn_tr, yn_te = train_test_split(
            X, y_pref, y_noshow,
            test_size=0.20, random_state=42
        )
        print(f"  Train rows: {len(X_tr):,}  |  Test rows: {len(X_te):,}")
        print(f"  No-show rate in train: {yn_tr.mean():.3f}")

        # --- Preference score model ---
        print("  Training PreferenceScoreModel ...")
        pref_model = PreferenceScoreModel()
        pref_model.fit(X_tr, yp_tr)
        pref_metrics = pref_model.evaluate(X_te, yp_te)
        print(f"  Preference model metrics: {pref_metrics}")
        print("  Top 10 preference features:")
        print(pref_model.get_feature_importance().head(10).to_string())

        if self.run_cv:
            cv = pref_model.cross_validate(X_tr, yp_tr)
            print(f"  CV preference MAE: {cv}")

        pref_path = os.path.join(self.output_dir, "preference_score_model.pkl")
        pref_model.save(pref_path)

        # --- No-show risk model ---
        print("  Training NoShowRiskModel ...")
        noshow_model = NoShowRiskModel()
        noshow_model.fit(X_tr, yn_tr)
        noshow_metrics = noshow_model.evaluate(X_te, yn_te)
        print(f"  No-show model metrics: {noshow_metrics}")
        print("  Top 10 no-show risk features:")
        print(noshow_model.get_feature_importance().head(10).to_string())

        noshow_path = os.path.join(self.output_dir, "noshow_risk_model.pkl")
        noshow_model.save(noshow_path)

        return pref_model, noshow_model, pref_metrics, noshow_metrics

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self):
        print("=" * 60)
        print("  AutomatedScheduling — ML Training Pipeline")
        print("=" * 60)

        demand_df, emp_df, sched_df = self._load_data()

        demand_model, demand_metrics = self._run_demand_pipeline(demand_df)
        pref_model, noshow_model, pref_metrics, noshow_metrics = (
            self._run_preference_pipeline(sched_df, emp_df)
        )

        print("\n" + "=" * 60)
        print("  Training Complete")
        print("=" * 60)
        print(f"\nArtifacts saved to: {self.output_dir}/")
        print("\nSummary:")
        print(f"  Demand Model  — {demand_metrics.to_dict(orient='index')}")
        print(f"  Pref Model    — {pref_metrics}")
        print(f"  NoShow Model  — {noshow_metrics}")

        return {
            "demand_model":  demand_model,
            "pref_model":    pref_model,
            "noshow_model":  noshow_model,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AutomatedScheduling ML Training Pipeline"
    )
    parser.add_argument("--data-dir",   default="data",             type=str)
    parser.add_argument("--output-dir", default="models/artifacts", type=str)
    parser.add_argument("--run-cv",     action="store_true",
                        help="Enable cross-validation (slower)")
    args = parser.parse_args()

    pipeline = TrainingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_cv=args.run_cv,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
