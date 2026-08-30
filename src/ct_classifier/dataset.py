from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, RandAffine, RandFlip, RandGaussianNoise
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .preprocessing import preprocess_ct


def encode_single_label(value: Any, classes: list[str]) -> int:
    text = str(value)
    if text in classes:
        return classes.index(text)
    try:
        index = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unknown label '{value}'. Expected one of {classes} or a class index") from error
    if not 0 <= index < len(classes):
        raise ValueError(f"Label index {index} is outside [0, {len(classes) - 1}]")
    return index


class CTVolumeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, config: dict[str, Any], training: bool = False):
        self.frame = frame.reset_index(drop=True).copy()
        self.config = config
        self.training = training
        aug = config["augmentation"]
        if training and aug.get("enabled", True):
            radians = math.radians(float(aug.get("rotate_degrees", 0)))
            self.augmentation = Compose(
                [
                    RandFlip(prob=float(aug.get("flip_probability", 0.5)), spatial_axis=2),
                    RandAffine(
                        prob=0.7,
                        rotate_range=(radians, radians, radians),
                        scale_range=(float(aug.get("scale_range", 0.0)),) * 3,
                        mode="bilinear",
                        padding_mode="border",
                    ),
                    RandGaussianNoise(prob=0.25, std=float(aug.get("noise_std", 0.0))),
                ]
            )
        else:
            self.augmentation = None

    def __len__(self) -> int:
        return len(self.frame)

    def _target(self, row: pd.Series) -> torch.Tensor:
        task = self.config["task"]
        columns = self.config["data"]["label_columns"]
        if task["type"] == "single_label":
            return torch.tensor(encode_single_label(row[columns[0]], task["classes"]), dtype=torch.long)
        values = row[columns].astype(float).to_numpy(dtype=np.float32)
        if not np.all(np.isin(values, [0.0, 1.0])):
            raise ValueError("multi_label targets must contain only 0/1 values")
        return torch.from_numpy(values)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        data = self.config["data"]
        image_path = str(Path(str(row[data["image_path_column"]])).expanduser())
        series_column = data.get("dicom_series_id_column")
        series_id = None
        if series_column and series_column in row.index and pd.notna(row[series_column]):
            series_id = str(row[series_column])
        image, _ = preprocess_ct(image_path, data, series_id=series_id)
        if self.augmentation is not None:
            image = self.augmentation(image)
        image = image.clamp(0.0, 1.0).float()
        result = {
            "image": image,
            "target": self._target(row),
            "patient_id": str(row[data["patient_id_column"]]),
            "study_id": str(row[data["study_id_column"]]),
            "image_path": image_path,
        }
        result["subgroups"] = {
            column: str(row[column]) for column in self.config.get("evaluation", {}).get("subgroup_columns", [])
        }
        return result


def _sampler_weights(frame: pd.DataFrame, config: dict[str, Any]) -> torch.Tensor:
    data = config["data"]
    if config["task"]["type"] == "single_label":
        labels = np.array(
            [encode_single_label(value, config["task"]["classes"]) for value in frame[data["label_columns"][0]]]
        )
        counts = np.bincount(labels, minlength=len(config["task"]["classes"])).clip(min=1)
        return torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
    matrix = frame[data["label_columns"]].astype(float).to_numpy()
    positives = matrix.sum(axis=0).clip(min=1)
    negatives = (len(matrix) - matrix.sum(axis=0)).clip(min=1)
    weights = (matrix / positives + (1.0 - matrix) / negatives).mean(axis=1)
    return torch.as_tensor(weights, dtype=torch.double)


def create_loader(frame: pd.DataFrame, config: dict[str, Any], training: bool) -> DataLoader:
    dataset = CTVolumeDataset(frame, config, training=training)
    use_sampler = training and bool(config["training"].get("balanced_sampler", False))
    sampler = None
    if use_sampler:
        weights = _sampler_weights(frame, config)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    workers = int(config["data"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
