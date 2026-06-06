"""Visualization utilities for the Employee Burnout Risk Prediction System.

All plots use professional formatting (titles, axis labels, grids) and are
saved to the screenshots/ directory by default. Each function focuses on one
chart so it can be reused by the notebook, main.py, and the dashboard.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")  # Safe for headless / CI environments.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid")
OUTPUT_DIR = "screenshots"


def _save(fig: plt.Figure, name: str, output_dir: str) -> str:
    """Persist a figure and close it."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def correlation_heatmap(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Plot a correlation heatmap of numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    return _save(fig, "correlation_heatmap.png", output_dir)


def histograms(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Plot histograms for every numeric feature."""
    numeric = df.select_dtypes(include=[np.number])
    cols = list(numeric.columns)
    n = len(cols)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        sns.histplot(numeric[col], kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"Distribution: {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    return _save(fig, "histograms.png", output_dir)


def boxplots(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Plot boxplots for outlier detection across numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    cols = list(numeric.columns)
    n = len(cols)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        sns.boxplot(y=numeric[col], ax=axes[i], color="salmon")
        axes[i].set_title(f"Boxplot: {col}")
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    return _save(fig, "boxplots.png", output_dir)


def scatter_vs_target(
    df: pd.DataFrame, target: str, features: Optional[Iterable[str]] = None,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """Scatter each feature against the target."""
    numeric = df.select_dtypes(include=[np.number])
    cols = list(features) if features else [c for c in numeric.columns if c != target]
    n = len(cols)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        sns.scatterplot(x=numeric[col], y=numeric[target], ax=axes[i],
                        alpha=0.5, color="darkgreen")
        axes[i].set_title(f"{col} vs {target}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target)
        axes[i].grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    return _save(fig, "scatter_vs_target.png", output_dir)


def target_distribution(
    df: pd.DataFrame, target: str, output_dir: str = OUTPUT_DIR
) -> str:
    """Plot the burnout risk distribution (highlights zero-inflation)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[target], kde=True, ax=ax, color="purple", bins=40)
    ax.set_title("Burnout Risk Score Distribution", fontweight="bold")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    return _save(fig, "target_distribution.png", output_dir)


def actual_vs_predicted(
    y_true, y_pred, output_dir: str = OUTPUT_DIR
) -> str:
    """Plot actual vs predicted values with the ideal diagonal."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, color="teal")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", label="Ideal")
    ax.set_title("Actual vs Predicted Burnout Risk", fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, "actual_vs_predicted.png", output_dir)


def residual_plots(y_true, y_pred, output_dir: str = OUTPUT_DIR) -> str:
    """Plot residuals vs predicted and the residual distribution."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(y_pred, residuals, alpha=0.5, color="orange")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Residuals vs Predicted", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, alpha=0.3)
    sns.histplot(residuals, kde=True, ax=axes[1], color="crimson")
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)
    return _save(fig, "residual_plots.png", output_dir)


def coefficient_plot(
    coefficients: dict, output_dir: str = OUTPUT_DIR
) -> str:
    """Plot model coefficients as a horizontal bar chart."""
    items = sorted(coefficients.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    colors = ["crimson" if v >= 0 else "steelblue" for v in values]
    ax.barh(names, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Linear Regression Coefficients", fontweight="bold")
    ax.set_xlabel("Coefficient (standardized)")
    ax.grid(True, alpha=0.3)
    return _save(fig, "coefficient_plot.png", output_dir)


def feature_importance_plot(
    importance: dict, output_dir: str = OUTPUT_DIR
) -> str:
    """Plot absolute-correlation feature importance."""
    items = sorted(importance.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    ax.barh(names, values, color="mediumseagreen")
    ax.set_title("Feature Importance (|correlation| with target)", fontweight="bold")
    ax.set_xlabel("|Pearson r|")
    ax.grid(True, alpha=0.3)
    return _save(fig, "feature_importance.png", output_dir)


def generate_all(
    df: pd.DataFrame, target: str, output_dir: str = OUTPUT_DIR
) -> List[str]:
    """Generate the core EDA plot set and return saved paths."""
    return [
        correlation_heatmap(df, output_dir),
        histograms(df, output_dir),
        boxplots(df, output_dir),
        scatter_vs_target(df, target, output_dir=output_dir),
        target_distribution(df, target, output_dir),
    ]
