from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def logits_to_probabilities(logits: np.ndarray, task_type: str) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if task_type == "single_label":
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


def choose_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    task_type: str,
    method: str = "youden",
) -> list[float]:
    classes = probabilities.shape[1]
    thresholds = [0.5] * classes
    if method == "fixed":
        return thresholds
    if method != "youden":
        raise ValueError("threshold_method must be 'youden' or 'fixed'")

    if task_type == "single_label":
        if classes != 2:
            return thresholds
        target_columns = [(targets == 1).astype(int)]
        probability_columns = [probabilities[:, 1]]
        indices = [1]
    else:
        target_columns = [targets[:, index].astype(int) for index in range(classes)]
        probability_columns = [probabilities[:, index] for index in range(classes)]
        indices = list(range(classes))

    for class_index, truth, probability in zip(indices, target_columns, probability_columns):
        if np.unique(truth).size < 2:
            continue
        false_positive_rate, true_positive_rate, candidates = roc_curve(truth, probability)
        finite = np.isfinite(candidates)
        if finite.any():
            scores = np.where(finite, true_positive_rate - false_positive_rate, -np.inf)
            thresholds[class_index] = float(np.clip(candidates[int(np.argmax(scores))], 0.0, 1.0))
    return thresholds


def predictions_from_probabilities(
    probabilities: np.ndarray,
    task_type: str,
    thresholds: list[float],
) -> np.ndarray:
    if task_type == "single_label":
        if probabilities.shape[1] == 2:
            return (probabilities[:, 1] >= thresholds[1]).astype(int)
        return probabilities.argmax(axis=1).astype(int)
    return (probabilities >= np.asarray(thresholds)[None, :]).astype(int)


def _safe_score(function: Callable[..., float], *args: Any, **kwargs: Any) -> float:
    try:
        value = float(function(*args, **kwargs))
        return value if np.isfinite(value) else float("nan")
    except ValueError:
        return float("nan")


def _binary_metrics(truth: np.ndarray, probability: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "support_positive": int(truth.sum()),
        "support_negative": int((1 - truth).sum()),
        "auroc": _safe_score(roc_auc_score, truth, probability),
        "auprc": _safe_score(average_precision_score, truth, probability),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": _safe_score(precision_score, truth, predicted, zero_division=0),
        "f1": _safe_score(f1_score, truth, predicted, zero_division=0),
        "brier": _safe_score(brier_score_loss, truth, probability),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _bootstrap_indices(patient_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(patient_ids)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    blocks = [np.flatnonzero(patient_ids == patient) for patient in sampled]
    return np.concatenate(blocks)


def _bootstrap_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    patient_ids: np.ndarray,
    task_type: str,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, list[float] | None]]:
    class_count = probabilities.shape[1]
    collected = [dict(auroc=[], auprc=[], sensitivity=[], specificity=[]) for _ in range(class_count)]
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        indices = _bootstrap_indices(patient_ids, rng)
        for class_index in range(class_count):
            truth = (
                (targets[indices] == class_index).astype(int)
                if task_type == "single_label"
                else targets[indices, class_index].astype(int)
            )
            predicted = (
                (predictions[indices] == class_index).astype(int)
                if task_type == "single_label"
                else predictions[indices, class_index].astype(int)
            )
            values = _binary_metrics(truth, probabilities[indices, class_index], predicted)
            for metric in collected[class_index]:
                value = float(values[metric])
                if np.isfinite(value):
                    collected[class_index][metric].append(value)

    alpha = (1.0 - confidence_level) / 2.0
    intervals: list[dict[str, list[float] | None]] = []
    for class_values in collected:
        item: dict[str, list[float] | None] = {}
        for metric, values in class_values.items():
            item[metric] = (
                [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))]
                if values
                else None
            )
        intervals.append(item)
    return intervals


def evaluate_probabilities(
    targets: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: list[str] | np.ndarray,
    class_names: list[str],
    task_type: str,
    thresholds: list[float],
    bootstrap_iterations: int = 0,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 2026,
) -> tuple[dict[str, Any], np.ndarray]:
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    patient_ids_array = np.asarray(patient_ids, dtype=str)
    predictions = predictions_from_probabilities(probabilities, task_type, thresholds)

    class_results = []
    for class_index, class_name in enumerate(class_names):
        truth = (
            (targets == class_index).astype(int)
            if task_type == "single_label"
            else targets[:, class_index].astype(int)
        )
        predicted = (
            (predictions == class_index).astype(int)
            if task_type == "single_label"
            else predictions[:, class_index].astype(int)
        )
        result = {"class": class_name, "threshold": float(thresholds[class_index])}
        result.update(_binary_metrics(truth, probabilities[:, class_index], predicted))
        class_results.append(result)

    intervals = None
    if bootstrap_iterations > 0:
        intervals = _bootstrap_intervals(
            targets,
            probabilities,
            predictions,
            patient_ids_array,
            task_type,
            bootstrap_iterations,
            confidence_level,
            bootstrap_seed,
        )
        for result, interval in zip(class_results, intervals):
            result["confidence_intervals"] = interval

    macro_keys = ["auroc", "auprc", "sensitivity", "specificity", "precision", "f1", "brier"]
    macro = {
        key: float(np.nanmean([float(result[key]) for result in class_results]))
        for key in macro_keys
    }
    overall_accuracy = float(accuracy_score(targets, predictions))
    payload: dict[str, Any] = {
        "task_type": task_type,
        "samples": int(len(targets)),
        "patients": int(len(np.unique(patient_ids_array))),
        "accuracy": overall_accuracy,
        "macro": macro,
        "classes": class_results,
        "bootstrap": {
            "iterations": int(bootstrap_iterations),
            "confidence_level": float(confidence_level),
            "sampling_unit": "patient",
        },
    }
    if task_type == "single_label":
        payload["confusion_matrix"] = confusion_matrix(
            targets, predictions, labels=list(range(len(class_names)))
        ).tolist()
    else:
        payload["confusion_matrices"] = [
            confusion_matrix(targets[:, index], predictions[:, index], labels=[0, 1]).tolist()
            for index in range(len(class_names))
        ]
    return payload, predictions


def prediction_frame(
    patient_ids: list[str],
    study_ids: list[str],
    image_paths: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    task_type: str,
    subgroups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {"patient_id": patient_ids, "study_id": study_ids, "image_path": image_paths}
    )
    for column, values in (subgroups or {}).items():
        result[column] = values
    for index, name in enumerate(class_names):
        result[f"probability_{name}"] = probabilities[:, index]
    if task_type == "single_label":
        result["target_index"] = targets.astype(int)
        result["target_label"] = [class_names[int(index)] for index in targets]
        result["prediction_index"] = predictions.astype(int)
        result["prediction_label"] = [class_names[int(index)] for index in predictions]
    else:
        for index, name in enumerate(class_names):
            result[f"target_{name}"] = targets[:, index].astype(int)
            result[f"prediction_{name}"] = predictions[:, index].astype(int)
    return result
