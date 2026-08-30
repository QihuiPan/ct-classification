from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def _truth_for_class(targets: np.ndarray, task_type: str, index: int) -> np.ndarray:
    return (targets == index).astype(int) if task_type == "single_label" else targets[:, index].astype(int)


def save_roc_plot(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    task_type: str,
    path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(class_names):
        truth = _truth_for_class(targets, task_type, index)
        if np.unique(truth).size < 2:
            continue
        false_positive, true_positive, _ = roc_curve(truth, probabilities[:, index])
        axis.plot(false_positive, true_positive, label=name)
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_pr_plot(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    task_type: str,
    path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(class_names):
        truth = _truth_for_class(targets, task_type, index)
        if np.unique(truth).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(truth, probabilities[:, index])
        axis.plot(recall, precision, label=name)
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
    axis.legend(loc="lower left")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_calibration_plot(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    task_type: str,
    path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(class_names):
        truth = _truth_for_class(targets, task_type, index)
        if np.unique(truth).size < 2:
            continue
        observed, predicted = calibration_curve(truth, probabilities[:, index], n_bins=10, strategy="quantile")
        axis.plot(predicted, observed, marker="o", label=name)
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="Predicted probability", ylabel="Observed frequency", title="Calibration")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_confusion_plot(matrix: np.ndarray, class_names: list[str], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names))))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="Reference",
        title="Confusion matrix",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_diagnostic_plots(
    targets: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict,
    class_names: list[str],
    task_type: str,
    output_dir: str | Path,
    prefix: str,
) -> None:
    output = Path(output_dir)
    save_roc_plot(targets, probabilities, class_names, task_type, output / f"{prefix}_roc.png")
    save_pr_plot(targets, probabilities, class_names, task_type, output / f"{prefix}_pr.png")
    save_calibration_plot(targets, probabilities, class_names, task_type, output / f"{prefix}_calibration.png")
    if task_type == "single_label":
        save_confusion_plot(
            np.asarray(metrics["confusion_matrix"]), class_names, output / f"{prefix}_confusion_matrix.png"
        )

