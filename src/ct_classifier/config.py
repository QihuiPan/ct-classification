from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(value: str | None, root: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (root / path).resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    config = deepcopy(config)
    project_root = config_path.parent.parent
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root)
    config["data"]["manifest"] = _resolve_path(config["data"]["manifest"], project_root)
    config["data"]["cache_dir"] = _resolve_path(config["data"].get("cache_dir"), project_root)
    config["model"]["pretrained_checkpoint"] = _resolve_path(
        config["model"].get("pretrained_checkpoint"), project_root
    )
    config["output"]["run_dir"] = _resolve_path(config["output"]["run_dir"], project_root)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    task_type = config["task"]["type"]
    classes = config["task"]["classes"]
    label_columns = config["data"]["label_columns"]
    if task_type not in {"single_label", "multi_label"}:
        raise ValueError("task.type must be 'single_label' or 'multi_label'")
    if len(classes) < 2:
        raise ValueError("At least two class names are required")
    if len(set(classes)) != len(classes):
        raise ValueError("task.classes must contain unique names")
    if task_type == "single_label" and len(label_columns) != 1:
        raise ValueError("single_label requires exactly one data.label_columns entry")
    if task_type == "multi_label" and len(label_columns) != len(classes):
        raise ValueError("multi_label requires one label column per class")
    series_column = config["data"].get("dicom_series_id_column")
    if series_column is not None and not str(series_column).strip():
        raise ValueError("dicom_series_id_column must be null or a non-empty column name")
    if len(config["data"]["target_spacing"]) != 3 or len(config["data"]["target_size"]) != 3:
        raise ValueError("target_spacing and target_size must both contain z, y, x values")
    if any(float(x) <= 0 for x in config["data"]["target_spacing"]):
        raise ValueError("target_spacing values must be positive")
    if any(int(x) <= 0 for x in config["data"]["target_size"]):
        raise ValueError("target_size values must be positive")
    split = config["split"]
    if split["mode"] not in {"patient_random", "external_site", "existing"}:
        raise ValueError("split.mode must be patient_random, external_site, or existing")
    total = float(split["train_fraction"]) + float(split["val_fraction"]) + float(split["test_fraction"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1")


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in deepcopy(config).items() if not key.startswith("_")}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable_config(config), handle, allow_unicode=True, sort_keys=False)
