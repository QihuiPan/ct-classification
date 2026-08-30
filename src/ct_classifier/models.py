from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import DenseNet121, resnet18


def output_channels(config: dict[str, Any]) -> int:
    return len(config["task"]["classes"])


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    architecture = str(config["model"]["architecture"]).lower()
    in_channels = len(config["data"]["windows"])
    out_channels = output_channels(config)
    dropout = float(config["model"].get("dropout", 0.0))
    if architecture == "densenet121":
        model = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout,
        )
    elif architecture == "resnet18":
        model = resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=out_channels,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")

    checkpoint = config["model"].get("pretrained_checkpoint")
    if checkpoint:
        load_weights(model, checkpoint, strict=bool(config["model"].get("strict_checkpoint_loading", False)))
    return model


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary")
    return {str(key).removeprefix("module."): value for key, value in payload.items()}


def load_weights(model: torch.nn.Module, path: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = _unwrap_state_dict(payload)
    if strict:
        incompatible = model.load_state_dict(state, strict=True)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
    current = model.state_dict()
    compatible = {
        key: value for key, value in state.items() if key in current and current[key].shape == value.shape
    }
    skipped = [key for key, value in state.items() if key in current and current[key].shape != value.shape]
    incompatible = model.load_state_dict(compatible, strict=False)
    unexpected = list(incompatible.unexpected_keys) + [key for key in state if key not in current] + skipped
    return list(incompatible.missing_keys), unexpected


def freeze_backbone(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_names = ("class_layers", "fc")
    for name, parameter in model.named_parameters():
        if any(token in name for token in trainable_names):
            parameter.requires_grad = True


def unfreeze_all(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
