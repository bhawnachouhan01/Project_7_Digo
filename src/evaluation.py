"""Model evaluation helpers: metrics, plots and error analysis.

Metric conventions used throughout the project
----------------------------------------------
* The **positive class is 1 = disaster-related** (:data:`src.config.POSITIVE_CLASS`).
  ``precision``, ``recall`` and ``f1`` below are all computed with respect to
  that class, because the operationally interesting question is "when the
  model shouts *disaster*, is it right, and does it miss real ones?".
* ``macro_f1`` averages the two classes equally, so a model that only does
  well on the majority class cannot hide behind it.
* ``weighted_f1`` weights by class support.

Why accuracy alone is misleading here: the dataset is 57.0% / 43.0%, so a
model that predicts "non-disaster" for everything scores 57% accuracy while
being completely useless for emergency monitoring - its recall on class 1 is
zero. Accuracy is reported, but never on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import LABEL_NAMES, POSITIVE_CLASS


def get_scores(estimator: Any, X: Any) -> np.ndarray | None:
    """Return continuous scores for the positive class, or ``None``.

    Handles both ``predict_proba`` models (Logistic Regression, Naive Bayes)
    and ``decision_function`` models (LinearSVC). Returning ``None`` rather
    than fabricating a score lets callers skip ROC/PR curves honestly when a
    model genuinely cannot produce ranked output.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: np.ndarray | None = None,
    model_name: str = "model",
) -> dict[str, Any]:
    """Compute the full metric set for one model.

    ``roc_auc`` and ``average_precision`` are included only when ``y_score``
    is provided; otherwise they are ``None`` rather than a placeholder number.
    """
    metrics: dict[str, Any] = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["average_precision"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def print_classification_report(y_true: Any, y_pred: Any, title: str = "") -> None:
    """Print a labelled sklearn classification report."""
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
            digits=4,
            zero_division=0,
        )
    )


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    title: str = "Confusion Matrix",
    save_path: Path | None = None,
) -> np.ndarray:
    """Plot a confusion matrix annotated with both counts and row percentages.

    Returns the raw matrix so the caller can quote exact cell counts.
    """
    matrix = confusion_matrix(y_true, y_pred)
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = matrix / np.maximum(row_totals, 1) * 100

    annotations = np.empty_like(matrix).astype(str)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            annotations[i, j] = f"{matrix[i, j]:,}\n({percentages[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1.2,
        linecolor="white",
        xticklabels=[LABEL_NAMES[0], LABEL_NAMES[1]],
        yticklabels=[LABEL_NAMES[0], LABEL_NAMES[1]],
        ax=ax,
        annot_kws={"fontsize": 11},
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(f"{title}\n(percentages are row-wise, i.e. per true class)", fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return matrix


def plot_roc_and_pr_curves(
    y_true: Any,
    y_score: np.ndarray,
    model_name: str = "model",
    save_path: Path | None = None,
) -> tuple[float, float]:
    """Plot ROC and precision-recall curves side by side.

    The PR curve's baseline is the positive-class prevalence, which is the
    honest reference for an imbalanced problem (a random ranker achieves it).

    Returns
    -------
    tuple
        ``(roc_auc, average_precision)``.
    """
    y_true_array = np.asarray(y_true)
    fpr, tpr, _ = roc_curve(y_true_array, y_score)
    roc_auc_value = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true_array, y_score)
    ap_value = average_precision_score(y_true_array, y_score)
    baseline = float((y_true_array == POSITIVE_CLASS).mean())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(fpr, tpr, linewidth=2.2, label=f"{model_name} (AUC = {roc_auc_value:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (recall)")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(recall, precision, linewidth=2.2, color="darkorange",
                 label=f"{model_name} (AP = {ap_value:.4f})")
    axes[1].axhline(baseline, color="k", linestyle="--", linewidth=1,
                    label=f"Baseline = prevalence ({baseline:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{model_name} - ranking quality (positive class = disaster)", fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return float(roc_auc_value), float(ap_value)


def plot_model_comparison(
    comparison: pd.DataFrame,
    metrics: tuple[str, ...] = ("accuracy", "f1", "macro_f1"),
    save_path: Path | None = None,
) -> None:
    """Grouped bar chart comparing models across the chosen metrics."""
    plot_data = comparison.set_index("model")[list(metrics)]

    ax = plot_data.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.78,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Model comparison on the held-out test set", fontsize=13)
    # Zoom the y-axis to where the differences actually are, but keep the
    # floor well below the lowest bar so no difference is visually inflated.
    lowest = float(plot_data.min().min())
    ax.set_ylim(max(0.0, lowest - 0.12), 1.0)
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2, rotation=90)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def get_misclassified(
    texts: Any,
    y_true: Any,
    y_pred: Any,
    y_score: np.ndarray | None = None,
    error_type: str = "false_positive",
    n: int = 10,
) -> pd.DataFrame:
    """Return the most confident misclassifications of a given type.

    Parameters
    ----------
    error_type:
        ``"false_positive"`` - predicted disaster, actually not.
        ``"false_negative"`` - predicted non-disaster, actually a disaster.
    n:
        Number of examples to return.

    Sorting by model score surfaces the errors the model was *most sure*
    about, which are the informative ones for error analysis.
    """
    frame = pd.DataFrame(
        {
            "text": np.asarray(texts),
            "true_label": np.asarray(y_true),
            "predicted_label": np.asarray(y_pred),
        }
    )
    if y_score is not None:
        frame["score"] = np.asarray(y_score)

    if error_type == "false_positive":
        mask = (frame["true_label"] == 0) & (frame["predicted_label"] == 1)
        ascending = False
    elif error_type == "false_negative":
        mask = (frame["true_label"] == 1) & (frame["predicted_label"] == 0)
        ascending = True
    else:
        raise ValueError(
            f"error_type must be 'false_positive' or 'false_negative', got {error_type!r}"
        )

    errors = frame[mask]
    if y_score is not None and not errors.empty:
        errors = errors.sort_values("score", ascending=ascending)
    return errors.head(n).reset_index(drop=True)
