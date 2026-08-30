from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .dataset import encode_single_label


class FocalLoss(torch.nn.Module):
    def __init__(self, task_type: str, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.task_type = task_type
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type == "single_label":
            ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
            probability = torch.exp(-ce)
            return (((1.0 - probability) ** self.gamma) * ce).mean()
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=self.weight, reduction="none")
        probability = torch.exp(-bce)
        return (((1.0 - probability) ** self.gamma) * bce).mean()


def _weights(frame: pd.DataFrame, config: dict[str, Any]) -> torch.Tensor | None:
    if not config["training"].get("class_weighting", True):
        return None
    data = config["data"]
    classes = config["task"]["classes"]
    if config["task"]["type"] == "single_label":
        labels = np.array([encode_single_label(value, classes) for value in frame[data["label_columns"][0]]])
        counts = np.bincount(labels, minlength=len(classes)).astype(np.float32)
        if np.any(counts == 0):
            raise ValueError(f"Training split is missing classes: {np.array(classes)[counts == 0].tolist()}")
        weights = len(labels) / (len(classes) * counts)
        return torch.as_tensor(weights, dtype=torch.float32)
    matrix = frame[data["label_columns"]].astype(float).to_numpy()
    positives = matrix.sum(axis=0)
    if np.any(positives == 0):
        raise ValueError(f"Training split has labels with zero positives: {np.array(classes)[positives == 0].tolist()}")
    return torch.as_tensor((len(matrix) - positives) / positives, dtype=torch.float32)


def build_loss(frame: pd.DataFrame, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    task_type = config["task"]["type"]
    weight = _weights(frame, config)
    if weight is not None:
        weight = weight.to(device)
    requested = str(config["training"].get("loss", "auto")).lower()
    if requested == "focal":
        return FocalLoss(task_type, gamma=float(config["training"].get("focal_gamma", 2.0)), weight=weight)
    if requested not in {"auto", "cross_entropy", "bce"}:
        raise ValueError(f"Unsupported loss: {requested}")
    if task_type == "single_label":
        return torch.nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
        )
    return torch.nn.BCEWithLogitsLoss(pos_weight=weight)

