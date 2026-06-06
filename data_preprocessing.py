"""Data loading and preprocessing for the Employee Burnout Risk Prediction System.

Responsibilities:
    * Auto-detect and load the dataset (repo root or data/ folder).
    * Auto-detect the target column even if the name differs slightly.
    * Handle missing values and duplicate records.
    * Drop identifier columns.
    * Scale features with StandardScaler.
    * Produce a reproducible 80/20 train-test split.
    * Emit a human-readable preprocessing report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Candidate locations for the dataset, checked in order.
_CANDIDATE_PATHS: Tuple[str, ...] = (
    "Employee_Burnout.csv",
    os.path.join("data", "Employee Burnout.csv"),
    os.path.join("data", "Employee_Burnout.csv"),
    "Employee Burnout.csv",
)

# Identifier columns that must never be used as features.
_ID_COLUMNS: Tuple[str, ...] = ("employee_id", "id", "emp_id")

# Tokens used to fuzzy-detect the target column.
_TARGET_TOKENS: Tuple[str, ...] = ("burnout_risk_score", "burnout", "risk_score", "burnout_risk")

RANDOM_STATE = 42
TEST_SIZE = 0.2


def find_dataset(explicit_path: Optional[str] = None) -> str:
    """Return the path to the dataset CSV.

    Args:
        explicit_path: Optional path provided by the caller. Checked first.

    Returns:
        The first existing path among the candidates.

    Raises:
        FileNotFoundError: If no candidate path exists.
    """
    candidates = ([explicit_path] if explicit_path else []) + list(_CANDIDATE_PATHS)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not locate the dataset. Looked in: "
        + ", ".join(p for p in candidates if p)
    )


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load the dataset into a DataFrame.

    Args:
        path: Optional explicit CSV path.

    Returns:
        The loaded DataFrame.
    """
    resolved = find_dataset(path)
    try:
        df = pd.read_csv(resolved)
    except Exception as exc:  # pragma: no cover - defensive
        raise IOError(f"Failed to read dataset at '{resolved}': {exc}") from exc
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """Auto-detect the burnout target column.

    Strategy:
        1. Exact match against known target tokens.
        2. Substring match (case-insensitive).
        3. Fall back to the last column.

    Args:
        df: Input DataFrame.

    Returns:
        The detected target column name.
    """
    lowered = {col.lower(): col for col in df.columns}
    for token in _TARGET_TOKENS:
        if token in lowered:
            return lowered[token]
    for token in _TARGET_TOKENS:
        for low, original in lowered.items():
            if token in low:
                return original
    return df.columns[-1]


@dataclass
class PreprocessResult:
    """Container for preprocessing outputs."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    scaler: StandardScaler
    feature_names: List[str]
    target_name: str
    report: str = field(default="")


def preprocess(
    df: pd.DataFrame,
    target: Optional[str] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreprocessResult:
    """Run the full preprocessing pipeline.

    Steps: detect target, drop IDs, handle missing values (median impute),
    drop duplicates, split 80/20, scale features with StandardScaler.

    Args:
        df: Raw input DataFrame.
        target: Optional explicit target column; auto-detected if None.
        test_size: Fraction held out for testing.
        random_state: Reproducibility seed.

    Returns:
        A populated PreprocessResult.
    """
    lines: List[str] = ["=" * 60, "PREPROCESSING REPORT", "=" * 60]
    work = df.copy()
    initial_rows = len(work)
    lines.append(f"Initial shape: {work.shape}")

    target_name = target or detect_target_column(work)
    lines.append(f"Detected target column: '{target_name}'")

    # Drop identifier columns.
    dropped_ids = [c for c in work.columns if c.lower() in _ID_COLUMNS]
    if dropped_ids:
        work = work.drop(columns=dropped_ids)
        lines.append(f"Dropped identifier columns: {dropped_ids}")

    # Handle duplicates.
    dup_count = int(work.duplicated().sum())
    if dup_count:
        work = work.drop_duplicates().reset_index(drop=True)
    lines.append(f"Duplicate rows removed: {dup_count}")

    # Handle missing values via median imputation on numeric columns.
    missing_total = int(work.isna().sum().sum())
    if missing_total:
        for col in work.columns:
            if work[col].isna().any() and pd.api.types.is_numeric_dtype(work[col]):
                median = work[col].median()
                work[col] = work[col].fillna(median)
        # Any remaining (non-numeric) NaNs dropped.
        work = work.dropna().reset_index(drop=True)
    lines.append(f"Missing values handled (median impute): {missing_total}")

    if target_name not in work.columns:
        raise KeyError(f"Target column '{target_name}' not present after cleaning.")

    # Note the zero-inflated nature of the target if applicable.
    zero_frac = float((work[target_name] == 0).mean())
    if zero_frac > 0.05:
        lines.append(
            f"NOTE: target is zero-inflated ({zero_frac:.1%} exact zeros). "
            "Linear Regression may underperform at the lower bound; documented in EDA."
        )

    feature_names = [c for c in work.columns if c != target_name]
    X = work[feature_names]
    y = work[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    lines.append(
        f"Train/test split: {len(X_train)} train / {len(X_test)} test "
        f"({int((1 - test_size) * 100)}/{int(test_size * 100)}, seed={random_state})"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lines.append("Features scaled with StandardScaler (fit on train only).")
    lines.append(f"Final feature count: {len(feature_names)}")
    lines.append(f"Rows retained: {len(work)} / {initial_rows}")
    lines.append("=" * 60)

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
        feature_names=feature_names,
        target_name=target_name,
        report="\n".join(lines),
    )


if __name__ == "__main__":
    frame = load_data()
    result = preprocess(frame)
    print(result.report)
