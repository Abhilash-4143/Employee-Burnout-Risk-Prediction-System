"""Model training for the Employee Burnout Risk Prediction System.

Trains a Linear Regression model, derives the human-readable model equation,
and persists the trained artifact (model + scaler + feature names) via pickle.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

DEFAULT_MODEL_PATH = os.path.join("models", "burnout_model.pkl")


@dataclass
class TrainedModel:
    """Bundle of everything needed to reproduce predictions."""

    model: LinearRegression
    feature_names: List[str]
    intercept: float
    coefficients: Dict[str, float]


def train_model(
    X_train_scaled: np.ndarray, y_train, feature_names: List[str]
) -> TrainedModel:
    """Fit a Linear Regression model.

    Args:
        X_train_scaled: Scaled training features.
        y_train: Training target.
        feature_names: Ordered feature names matching the columns.

    Returns:
        A TrainedModel bundle.
    """
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    coeffs = {name: float(c) for name, c in zip(feature_names, model.coef_)}
    return TrainedModel(
        model=model,
        feature_names=list(feature_names),
        intercept=float(model.intercept_),
        coefficients=coeffs,
    )


def model_equation(trained: TrainedModel, target: str = "burnout_risk_score") -> str:
    """Build a readable regression equation string.

    Args:
        trained: The trained model bundle.
        target: Target variable name for the left-hand side.

    Returns:
        A formatted equation string.
    """
    terms = [f"{trained.intercept:.3f}"]
    for name, coef in trained.coefficients.items():
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.3f}*{name}")
    return f"{target} = " + " ".join(terms)


def interpret_coefficients(trained: TrainedModel) -> List[str]:
    """Translate scaled coefficients into business-language statements.

    Note: coefficients are on standardized features, so each statement is
    expressed per one standard-deviation increase.

    Args:
        trained: The trained model bundle.

    Returns:
        A list of plain-language interpretation strings, strongest first.
    """
    ranked = sorted(
        trained.coefficients.items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    out: List[str] = []
    for name, coef in ranked:
        direction = "increases" if coef >= 0 else "decreases"
        out.append(
            f"A 1 std-dev rise in '{name}' {direction} predicted burnout risk "
            f"by {abs(coef):.2f} points."
        )
    return out


def save_model(
    trained: TrainedModel, scaler, path: str = DEFAULT_MODEL_PATH
) -> str:
    """Persist the model, scaler, and feature names with pickle.

    Args:
        trained: The trained model bundle.
        scaler: The fitted StandardScaler.
        path: Destination path for the pickle file.

    Returns:
        The path written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": trained.model,
        "scaler": scaler,
        "feature_names": trained.feature_names,
        "intercept": trained.intercept,
        "coefficients": trained.coefficients,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_model(path: str = DEFAULT_MODEL_PATH) -> dict:
    """Load a persisted model payload.

    Args:
        path: Pickle file path.

    Returns:
        The unpickled payload dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at '{path}'. Run main.py first.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def predict(payload: dict, features: np.ndarray) -> np.ndarray:
    """Run a prediction from a loaded payload on raw (unscaled) features.

    Args:
        payload: Output of load_model.
        features: 2D array of raw feature values in feature_names order.

    Returns:
        Array of predicted burnout risk scores, clipped to [0, 100].
    """
    scaled = payload["scaler"].transform(features)
    preds = payload["model"].predict(scaled)
    return np.clip(preds, 0, 100)


if __name__ == "__main__":
    from src.data_preprocessing import load_data, preprocess
    from src.feature_engineering import add_engineered_features

    pre = preprocess(add_engineered_features(load_data()))
    tm = train_model(pre.X_train_scaled, pre.y_train, pre.feature_names)
    print(model_equation(tm, pre.target_name))
    print(save_model(tm, pre.scaler))
