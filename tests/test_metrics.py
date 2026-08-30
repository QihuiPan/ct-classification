from __future__ import annotations

import numpy as np

from ct_classifier.metrics import choose_thresholds, evaluate_probabilities


def test_binary_metrics_are_perfect_for_separable_predictions() -> None:
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    thresholds = choose_thresholds(targets, probabilities, "single_label", method="youden")
    metrics, predictions = evaluate_probabilities(
        targets,
        probabilities,
        ["P1", "P2", "P3", "P4"],
        ["negative", "positive"],
        "single_label",
        thresholds,
        bootstrap_iterations=20,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["macro"]["auroc"] == 1.0
    assert predictions.tolist() == targets.tolist()

