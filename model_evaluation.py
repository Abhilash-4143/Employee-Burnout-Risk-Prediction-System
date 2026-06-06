"""Model evaluation for the Employee Burnout Risk Prediction System.

Computes regression metrics, adjusted R-squared, and cross-validation scores,
and renders a reliability report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


@dataclass
class EvaluationResult:
    """Holds all evaluation metrics."""

    mae: float
    mse: float
    rmse: float
    r2: float
    adjusted_r2: float
    cv_mean: float
    cv_std: float
    n_test: int
    n_features: int


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Compute adjusted R-squared.

    Args:
        r2: R-squared score.
        n: Number of observations.
        p: Number of predictors.

    Returns:
        Adjusted R-squared (falls back to r2 when undefined).
    """
    denom = n - p - 1
    if denom <= 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / denom


def evaluate(
    model: LinearRegression,
    X_test_scaled: np.ndarray,
    y_test,
    X_train_scaled: np.ndarray,
    y_train,
    cv: int = 5,
) -> EvaluationResult:
    """Evaluate a fitted model on the held-out test set.

    Args:
        model: Fitted regression model.
        X_test_scaled: Scaled test features.
        y_test: Test target.
        X_train_scaled: Scaled training features (for cross-validation).
        y_train: Training target (for cross-validation).
        cv: Number of cross-validation folds.

    Returns:
        A populated EvaluationResult.
    """
    preds = model.predict(X_test_scaled)
    mae = float(mean_absolute_error(y_test, preds))
    mse = float(mean_squared_error(y_test, preds))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds))
    n, p = X_test_scaled.shape[0], X_test_scaled.shape[1]
    adj = float(adjusted_r2(r2, n, p))

    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=cv, scoring="r2"
    )
    return EvaluationResult(
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        adjusted_r2=adj,
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
        n_test=n,
        n_features=p,
    )


def reliability_report(res: EvaluationResult) -> str:
    """Render a textual evaluation + reliability report.

    Args:
        res: The evaluation result.

    Returns:
        A formatted multi-line report.
    """
    lines: List[str] = [
        "=" * 60,
        "MODEL EVALUATION REPORT",
        "=" * 60,
        f"{'Mean Absolute Error (MAE)':<32}{res.mae:>12.4f}",
        f"{'Mean Squared Error (MSE)':<32}{res.mse:>12.4f}",
        f"{'Root Mean Squared Error (RMSE)':<32}{res.rmse:>12.4f}",
        f"{'R-squared':<32}{res.r2:>12.4f}",
        f"{'Adjusted R-squared':<32}{res.adjusted_r2:>12.4f}",
        f"{'CV R2 (mean)':<32}{res.cv_mean:>12.4f}",
        f"{'CV R2 (std)':<32}{res.cv_std:>12.4f}",
        "-" * 60,
    ]
    if res.r2 >= 0.75:
        verdict = "Strong fit: the model explains most variance in burnout risk."
    elif res.r2 >= 0.5:
        verdict = "Moderate fit: useful for ranking risk, refine before high-stakes use."
    elif res.r2 >= 0.25:
        verdict = "Weak fit: directional signal only; consider non-linear models."
    else:
        verdict = (
            "Poor linear fit: likely due to zero-inflated target; "
            "consider Tweedie/two-part or tree-based models."
        )
    lines.append(f"Reliability: {verdict}")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    from src.data_preprocessing import load_data, preprocess
    from src.feature_engineering import add_engineered_features
    from src.model_training import train_model

    pre = preprocess(add_engineered_features(load_data()))
    tm = train_model(pre.X_train_scaled, pre.y_train, pre.feature_names)
    ev = evaluate(
        tm.model, pre.X_test_scaled, pre.y_test, pre.X_train_scaled, pre.y_train
    )
    print(reliability_report(ev))
