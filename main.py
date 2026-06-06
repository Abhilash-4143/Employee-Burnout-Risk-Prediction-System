from __future__ import annotations

import os

from src import visualization as viz
from src.data_preprocessing import load_data, preprocess
from src.feature_engineering import (
    add_engineered_features,
    feature_importance_by_correlation,
)
from src.model_evaluation import evaluate, reliability_report
from src.model_training import (
    interpret_coefficients,
    model_equation,
    save_model,
    train_model,
)

REPORTS_DIR = "reports"


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def run() -> None:
    """Execute the full training and reporting pipeline."""
    print("[1/7] Loading dataset...")
    raw = load_data()

    print("[2/7] Engineering features...")
    engineered = add_engineered_features(raw)

    print("[3/7] Preprocessing...")
    pre = preprocess(engineered)
    print(pre.report)
    _write(os.path.join(REPORTS_DIR, "preprocessing_report.txt"), pre.report)

    print("[4/7] Training Linear Regression...")
    trained = train_model(pre.X_train_scaled, pre.y_train, pre.feature_names)
    equation = model_equation(trained, pre.target_name)
    interpretations = interpret_coefficients(trained)
    print(equation)

    print("[5/7] Evaluating...")
    ev = evaluate(
        trained.model, pre.X_test_scaled, pre.y_test,
        pre.X_train_scaled, pre.y_train,
    )
    report = reliability_report(ev)
    print(report)
    model_report = "\n\n".join(
        [report, "MODEL EQUATION", equation, "COEFFICIENT INTERPRETATION",
         "\n".join(f"- {line}" for line in interpretations)]
    )
    _write(os.path.join(REPORTS_DIR, "model_report.txt"), model_report)

    print("[6/7] Saving model...")
    path = save_model(trained, pre.scaler)
    print(f"  Saved -> {path}")

    print("[7/7] Generating visualizations...")
    importance = feature_importance_by_correlation(engineered, pre.target_name)
    preds = trained.model.predict(pre.X_test_scaled)
    viz.generate_all(engineered, pre.target_name)
    viz.actual_vs_predicted(pre.y_test, preds)
    viz.residual_plots(pre.y_test, preds)
    viz.coefficient_plot(trained.coefficients)
    viz.feature_importance_plot(importance)
    print("Done. See models/, reports/, and screenshots/.")


if __name__ == "__main__":
    run()
