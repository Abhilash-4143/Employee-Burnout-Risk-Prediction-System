"""Feature engineering for the Employee Burnout Risk Prediction System.

Creates domain-driven composite features:
    * Workload Index   = weekly_work_hours + meetings_per_week + emails_sent_per_day
    * Health Index     = (sleep_hours + exercise_hours_week) / stress_level
    * Productivity Efficiency = productivity_score / weekly_work_hours

Also exposes a correlation-based feature importance helper.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

_EPS = 1e-9  # Guards against division by zero.


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append engineered composite features to a copy of the DataFrame.

    Missing source columns are skipped gracefully so the function works on
    partial schemas (e.g. during inference with a subset of inputs).

    Args:
        df: Input DataFrame containing the raw workplace/lifestyle features.

    Returns:
        A new DataFrame with added engineered columns.
    """
    out = df.copy()

    if {"weekly_work_hours", "meetings_per_week", "emails_sent_per_day"}.issubset(out.columns):
        out["workload_index"] = (
            out["weekly_work_hours"]
            + out["meetings_per_week"]
            + out["emails_sent_per_day"]
        )

    if {"sleep_hours", "exercise_hours_week", "stress_level"}.issubset(out.columns):
        out["health_index"] = (
            out["sleep_hours"] + out["exercise_hours_week"]
        ) / (out["stress_level"].replace(0, np.nan).fillna(_EPS))

    if {"productivity_score", "weekly_work_hours"}.issubset(out.columns):
        out["productivity_efficiency"] = out["productivity_score"] / (
            out["weekly_work_hours"].replace(0, np.nan).fillna(_EPS)
        )

    return out


def feature_importance_by_correlation(
    df: pd.DataFrame, target: str
) -> Dict[str, float]:
    """Rank features by absolute Pearson correlation with the target.

    Args:
        df: DataFrame including the target column.
        target: Name of the target column.

    Returns:
        Mapping of feature name to absolute correlation, sorted descending.
    """
    numeric = df.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        raise KeyError(f"Target '{target}' is not numeric or not present.")
    corr = numeric.corr()[target].drop(labels=[target]).abs()
    return dict(corr.sort_values(ascending=False))


if __name__ == "__main__":
    from src.data_preprocessing import detect_target_column, load_data

    frame = add_engineered_features(load_data())
    tgt = detect_target_column(frame)
    for name, score in feature_importance_by_correlation(frame, tgt).items():
        print(f"{name:<28} {score:.3f}")
