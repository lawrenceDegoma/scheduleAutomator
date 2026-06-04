"""
feature_engineering.py

Transforms raw demand and schedule CSV data into model-ready feature matrices.

Two feature pipelines:
  1. DemandFeatureBuilder  — for the sales/transaction forecasting model
  2. PreferenceFeatureBuilder — for the barista preferred-hours learning model
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ===========================================================================
# 1. DEMAND FEATURE BUILDER
# ===========================================================================

class DemandFeatureBuilder:
    """
    Converts raw demand history rows into an ML feature matrix.

    Expected input columns (matches data_generator output):
        location_id, date, datetime, hour, minute, day_of_week,
        month, is_holiday, is_promotion, is_rain,
        transactions, sales

    Output features:
        Temporal:
            hour, minute, day_of_week, month, week_of_year,
            is_weekend, is_monday, is_friday, quarter

        Cyclical encodings (sin/cos):
            hour_sin, hour_cos, dow_sin, dow_cos,
            month_sin, month_cos, week_sin, week_cos

        Contextual:
            is_holiday, is_promotion, is_rain

        Lag features (transactions):
            lag_1w_same_slot   — same slot 1 week ago
            lag_2w_same_slot   — same slot 2 weeks ago
            lag_4w_same_slot   — same slot 4 weeks ago
            rolling_4w_mean    — 4-week rolling mean of same slot
            rolling_4w_std     — 4-week rolling std  of same slot

    Targets:
        transactions, sales, required_headcount (derived)
    """

    # Transactions-per-barista threshold used to derive headcount
    TX_PER_BARISTA_PER_30MIN = 18

    def __init__(self):
        self._fitted = False

    # ------------------------------------------------------------------
    # Cyclical encoding helper
    # ------------------------------------------------------------------
    @staticmethod
    def _cyclical(series: pd.Series, period: float):
        sin = np.sin(2 * np.pi * series / period)
        cos = np.cos(2 * np.pi * series / period)
        return sin, cos

    # ------------------------------------------------------------------
    # Lag feature builder
    # ------------------------------------------------------------------
    @staticmethod
    def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("datetime")
        # Create a slot key: day_of_week + hour + minute
        df["slot_key"] = (
            df["day_of_week"].astype(str) + "_"
            + df["hour"].astype(str) + "_"
            + df["minute"].astype(str)
        )
        # Group by slot to compute rolling stats per slot type
        df["lag_1w_same_slot"] = (
            df.groupby("slot_key")["transactions"]
              .shift(1)
        )
        df["lag_2w_same_slot"] = (
            df.groupby("slot_key")["transactions"]
              .shift(2)
        )
        df["lag_4w_same_slot"] = (
            df.groupby("slot_key")["transactions"]
              .shift(4)
        )
        df["rolling_4w_mean"] = (
            df.groupby("slot_key")["transactions"]
              .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        )
        df["rolling_4w_std"] = (
            df.groupby("slot_key")["transactions"]
              .transform(lambda x: x.shift(1).rolling(4, min_periods=1).std())
        )
        df["rolling_4w_std"] = df["rolling_4w_std"].fillna(0)
        return df

    # ------------------------------------------------------------------
    # Required headcount derivation
    # ------------------------------------------------------------------
    def _derive_headcount(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Baristas needed based on transaction volume
        baristas_needed = np.ceil(
            df["transactions"] / self.TX_PER_BARISTA_PER_30MIN
        ).clip(lower=1).astype(int)
        # Always need 1 supervisor + floor barista count + 1 register
        df["required_headcount"] = baristas_needed + 2
        return df

    # ------------------------------------------------------------------
    # Main transform
    # ------------------------------------------------------------------
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = pd.to_datetime(df["date"])

        # --- Temporal ---
        df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["datetime"].dt.quarter
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        # --- Cyclical ---
        # Slot index within day (0–33 for 30-min intervals 5am–10pm)
        df["slot_index"] = (df["hour"] - 5) * 2 + (df["minute"] // 30)
        df["hour_sin"], df["hour_cos"] = self._cyclical(df["slot_index"], 34)
        df["dow_sin"],  df["dow_cos"]  = self._cyclical(df["day_of_week"], 7)
        df["month_sin"], df["month_cos"] = self._cyclical(df["month"], 12)
        df["week_sin"], df["week_cos"]  = self._cyclical(df["week_of_year"], 52)

        # --- Lag features ---
        df = self._add_lag_features(df)

        # --- Derived headcount target ---
        df = self._derive_headcount(df)

        return df

    def get_feature_columns(self) -> list:
        return [
            # temporal
            "hour", "minute", "day_of_week", "month", "week_of_year",
            "quarter", "is_weekend", "is_monday", "is_friday",
            # cyclical
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "week_sin", "week_cos",
            # contextual
            "is_holiday", "is_promotion", "is_rain",
            # lag
            "lag_1w_same_slot", "lag_2w_same_slot", "lag_4w_same_slot",
            "rolling_4w_mean", "rolling_4w_std",
        ]

    def get_target_columns(self) -> list:
        return ["transactions", "sales", "required_headcount"]


# ===========================================================================
# 2. PREFERENCE FEATURE BUILDER
# ===========================================================================

class PreferenceFeatureBuilder:
    """
    Builds a per-employee feature matrix to learn latent shift preferences
    from historical schedule assignments.

    One row = one (employee, shift_type) combination per week.

    Input DataFrames:
      - schedules_df : output of EmployeeScheduleGenerator.generate()[1]
      - employees_df : output of EmployeeScheduleGenerator.generate()[0]

    Features (per row):
        Employee attributes:
            tenure_months, pay_rate, employment_type_enc,
            productivity_score, no_show_risk, fatigue_sensitivity,
            is_probationary, desired_weekly_hours

        Historical assignment patterns (rolling 8-week window):
            pct_morning_shifts    — fraction of assigned shifts that were opens/short_am
            pct_evening_shifts    — fraction that were close/short_pm
            pct_midday_shifts     — fraction that were mid
            pct_weekend_shifts    — fraction of shifts on Sat/Sun
            avg_weekly_hours      — average actual scheduled hours
            no_show_rate          — rolling no-show rate
            consecutive_days_max  — max consecutive days worked (fatigue proxy)

        Shift attributes:
            shift_start_hour, shift_end_hour, shift_hours,
            is_weekend, day_of_week, shift_type_enc

    Target:
        preference_score — 1.0 if the employee has never had this shift
                           type overridden by manager AND no-show was 0,
                           0.5 if no-show only once in rolling window,
                           0.0 if shift was manager-overridden (implicit dislike)
    """

    MORNING_SHIFTS = {"open", "short_am"}
    EVENING_SHIFTS = {"close", "short_pm"}

    def __init__(self):
        self._emp_enc = LabelEncoder()
        self._shift_enc = LabelEncoder()
        self._type_enc = LabelEncoder()
        self._fitted = False

    # ------------------------------------------------------------------
    # Rolling per-employee aggregates (8-week lookback)
    # ------------------------------------------------------------------
    def _build_rolling_features(self, schedules: pd.DataFrame) -> pd.DataFrame:
        schedules = schedules.copy()
        schedules["date"] = pd.to_datetime(schedules["date"])
        schedules["week_start"] = pd.to_datetime(schedules["week_start"])

        agg_list = []
        for emp_id, grp in schedules.groupby("employee_id"):
            grp = grp.sort_values("date")
            # Weekly aggregation
            weekly = (
                grp.groupby("week_start")
                   .agg(
                       total_hours=("scheduled_hours", "sum"),
                       n_shifts=("scheduled_hours", "count"),
                       n_morning=("shift_type",
                                  lambda x: x.isin(self.MORNING_SHIFTS).sum()),
                       n_evening=("shift_type",
                                  lambda x: x.isin(self.EVENING_SHIFTS).sum()),
                       n_weekend=("is_weekend" if "is_weekend" in grp.columns
                                  else "day_of_week",
                                  lambda x: (x >= 5).sum()),
                       n_no_show=("no_show", "sum"),
                   )
                   .reset_index()
            )
            # day_of_week weekends
            if "n_weekend" not in weekly.columns:
                weekly = weekly.rename(columns={"day_of_week": "n_weekend"})

            # 8-week rolling means
            for col in ["total_hours", "n_morning", "n_evening",
                        "n_weekend", "n_no_show", "n_shifts"]:
                weekly[f"roll8_{col}"] = (
                    weekly[col].rolling(8, min_periods=1).mean()
                )

            weekly["employee_id"] = emp_id
            agg_list.append(weekly)

        rolling_df = pd.concat(agg_list, ignore_index=True)

        # Compute percentages
        rolling_df["pct_morning_shifts"] = (
            rolling_df["roll8_n_morning"] /
            rolling_df["roll8_n_shifts"].clip(lower=1)
        )
        rolling_df["pct_evening_shifts"] = (
            rolling_df["roll8_n_evening"] /
            rolling_df["roll8_n_shifts"].clip(lower=1)
        )
        rolling_df["pct_midday_shifts"] = (
            1 - rolling_df["pct_morning_shifts"] - rolling_df["pct_evening_shifts"]
        ).clip(lower=0)
        rolling_df["pct_weekend_shifts"] = (
            rolling_df["roll8_n_weekend"] /
            rolling_df["roll8_n_shifts"].clip(lower=1)
        )
        rolling_df["avg_weekly_hours"] = rolling_df["roll8_total_hours"]
        rolling_df["no_show_rate"] = (
            rolling_df["roll8_n_no_show"] /
            rolling_df["roll8_n_shifts"].clip(lower=1)
        )
        return rolling_df

    # ------------------------------------------------------------------
    # Consecutive days worked (fatigue proxy)
    # ------------------------------------------------------------------
    @staticmethod
    def _max_consecutive_days(dates: pd.Series) -> int:
        if len(dates) == 0:
            return 0
        dates = sorted(dates.dt.date.unique())
        max_run, run = 1, 1
        for i in range(1, len(dates)):
            if (dates[i] - dates[i - 1]).days == 1:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        return max_run

    # ------------------------------------------------------------------
    # Build target: preference_score
    # ------------------------------------------------------------------
    @staticmethod
    def _build_target(row) -> float:
        if row["manager_override"] == 1:
            return 0.0      # implicit dislike
        if row["no_show"] == 1:
            return 0.5      # uncertain
        return 1.0          # accepted without friction

    # ------------------------------------------------------------------
    # Main build
    # ------------------------------------------------------------------
    def build(self, schedules_df: pd.DataFrame,
              employees_df: pd.DataFrame) -> pd.DataFrame:

        schedules_df = schedules_df.copy()
        schedules_df["date"] = pd.to_datetime(schedules_df["date"])
        schedules_df["is_weekend"] = (
            schedules_df["day_of_week"] >= 5
        ).astype(int)

        # Support both real data (partner_id) and legacy (employee_id)
        id_col = "partner_id" if "partner_id" in schedules_df.columns else "employee_id"
        emp_id_col = "partner_id" if "partner_id" in employees_df.columns else "employee_id"

        # Normalise to a single join key name for internal use
        schedules_df = schedules_df.rename(columns={id_col: "_pid"})
        employees_df = employees_df.rename(columns={emp_id_col: "_pid"})

        # Rolling employee-level features
        rolling = self._build_rolling_features(schedules_df.rename(columns={"_pid": "employee_id"}))
        rolling = rolling.rename(columns={"employee_id": "_pid"})

        # Merge rolling features onto individual shift rows
        schedules_df["week_start"] = pd.to_datetime(schedules_df["week_start"])
        rolling["week_start"] = pd.to_datetime(rolling["week_start"])
        merged = schedules_df.merge(
            rolling[[
                "_pid", "week_start",
                "pct_morning_shifts", "pct_evening_shifts",
                "pct_midday_shifts", "pct_weekend_shifts",
                "avg_weekly_hours", "no_show_rate",
            ]],
            on=["_pid", "week_start"],
            how="left",
        )

        # Merge employee attributes
        merged = merged.merge(
            employees_df[[
                "_pid", "tenure_months", "pay_rate",
                "employment_type", "productivity_score",
                "no_show_risk", "fatigue_sensitivity",
                "is_probationary", "desired_weekly_hours",
            ]],
            on="_pid",
            how="left",
        )

        # Restore public-facing id column
        merged = merged.rename(columns={"_pid": "employee_id"})

        # Encode categoricals
        merged["employment_type_enc"] = self._emp_enc.fit_transform(
            merged["employment_type"].fillna("unknown")
        )
        merged["shift_type_enc"] = self._shift_enc.fit_transform(
            merged["shift_type"].fillna("unknown")
        )

        # Shift duration
        merged["shift_hours"] = (
            merged["shift_end_hour"] - merged["shift_start_hour"]
        )

        # Target
        merged["preference_score"] = merged.apply(self._build_target, axis=1)

        self._fitted = True
        return merged

    def get_feature_columns(self) -> list:
        return [
            # employee
            "tenure_months", "pay_rate", "employment_type_enc",
            "productivity_score", "no_show_risk", "fatigue_sensitivity",
            "is_probationary", "desired_weekly_hours",
            # rolling history
            "pct_morning_shifts", "pct_evening_shifts",
            "pct_midday_shifts", "pct_weekend_shifts",
            "avg_weekly_hours", "no_show_rate",
            # shift
            "shift_start_hour", "shift_end_hour", "shift_hours",
            "is_weekend", "day_of_week", "shift_type_enc",
        ]

    def get_target_column(self) -> str:
        return "preference_score"
