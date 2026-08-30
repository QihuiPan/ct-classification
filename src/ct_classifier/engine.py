from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .calibration import apply_temperature, fit_temperature
from .config import save_config, serializable_config
from .dataset import create_loader
from .losses import build_loss
from .metrics import (
    choose_thresholds,
    evaluate_probabilities,
    logits_to_probabilities,
    prediction_frame,
)
from .models import build_model, freeze_backbone, unfreeze_all
from .plots import save_diagnostic_plots
from .utils import ensure_dir, save_json


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    mixed_precision: bool = False,
    accumulation_steps: int = 1,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    patient_ids: list[str] = []
    study_ids: list[str] = []
    image_paths: list[str] = []
    subgroups: dict[str, list[str]] = {}
    if training:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, leave=False, desc="train" if training else "evaluate")
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with _autocast(device, mixed_precision):
                logits = model(images)
                loss = criterion(logits, targets)
                scaled_loss = loss / max(1, accumulation_steps)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
                if should_step:
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        logits_parts.append(logits.detach().float().cpu())
        target_parts.append(targets.detach().cpu())
        patient_ids.extend(str(value) for value in batch["patient_id"])
        study_ids.extend(str(value) for value in batch["study_id"])
        image_paths.extend(str(value) for value in batch["image_path"])
        for column, values in batch.get("subgroups", {}).items():
            subgroups.setdefault(column, []).extend(str(value) for value in values)
        progress.set_postfix(loss=f"{total_loss / max(1, total_samples):.4f}")

    if not logits_parts:
        raise ValueError("Data loader produced no batches")
    return {
        "loss": total_loss / total_samples,
        "logits": torch.cat(logits_parts),
        "targets": torch.cat(target_parts),
        "patient_ids": patient_ids,
        "study_ids": study_ids,
        "image_paths": image_paths,
        "subgroups": subgroups,
    }


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader, device: torch.device, mixed_precision: bool) -> dict[str, Any]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    patient_ids: list[str] = []
    study_ids: list[str] = []
    image_paths: list[str] = []
    subgroups: dict[str, list[str]] = {}
    for batch in tqdm(loader, leave=False, desc="predict"):
        images = batch["image"].to(device, non_blocking=True)
        with _autocast(device, mixed_precision):
            logits = model(images)
        logits_parts.append(logits.detach().float().cpu())
        target_parts.append(batch["target"].detach().cpu())
        patient_ids.extend(str(value) for value in batch["patient_id"])
        study_ids.extend(str(value) for value in batch["study_id"])
        image_paths.extend(str(value) for value in batch["image_path"])
        for column, values in batch.get("subgroups", {}).items():
            subgroups.setdefault(column, []).extend(str(value) for value in values)
    if not logits_parts:
        raise ValueError("Data loader produced no batches")
    return {
        "logits": torch.cat(logits_parts),
        "targets": torch.cat(target_parts),
        "patient_ids": patient_ids,
        "study_ids": study_ids,
        "image_paths": image_paths,
        "subgroups": subgroups,
    }


def _make_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    name = str(training.get("optimizer", "adamw")).lower()
    kwargs = {
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def _make_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    name = str(config["training"].get("scheduler", "cosine")).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(config["training"]["epochs"]))
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {name}")


def _selection_score(result: dict[str, Any], config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    probabilities = logits_to_probabilities(result["logits"].numpy(), config["task"]["type"])
    thresholds = [0.5] * len(config["task"]["classes"])
    metrics, _ = evaluate_probabilities(
        result["targets"].numpy(),
        probabilities,
        result["patient_ids"],
        config["task"]["classes"],
        config["task"]["type"],
        thresholds,
    )
    score = float(metrics["macro"]["auroc"])
    if not math.isfinite(score):
        score = -float(result["loss"])
    return score, metrics


def evaluate_and_save(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    config: dict[str, Any],
    output_dir: str | Path,
    split_name: str,
    temperature: float,
    thresholds: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = predict_loader(
        model,
        loader,
        device,
        mixed_precision=bool(config["training"].get("mixed_precision", True)),
    )
    calibrated_logits = apply_temperature(raw["logits"], temperature).numpy()
    targets = raw["targets"].numpy()
    probabilities = logits_to_probabilities(calibrated_logits, config["task"]["type"])
    evaluation = config["evaluation"]
    metrics, predictions = evaluate_probabilities(
        targets,
        probabilities,
        raw["patient_ids"],
        config["task"]["classes"],
        config["task"]["type"],
        thresholds,
        bootstrap_iterations=int(evaluation.get("bootstrap_iterations", 0)),
        confidence_level=float(evaluation.get("confidence_level", 0.95)),
        bootstrap_seed=int(evaluation.get("bootstrap_seed", config.get("seed", 2026))),
    )
    metrics["temperature"] = float(temperature)
    subgroup_metrics: dict[str, Any] = {}
    for column, values in raw.get("subgroups", {}).items():
        value_array = np.asarray(values, dtype=str)
        column_results: dict[str, Any] = {}
        for group_value in np.unique(value_array):
            mask = value_array == group_value
            group_metrics, _ = evaluate_probabilities(
                targets[mask],
                probabilities[mask],
                np.asarray(raw["patient_ids"], dtype=str)[mask],
                config["task"]["classes"],
                config["task"]["type"],
                thresholds,
                bootstrap_iterations=0,
            )
            column_results[str(group_value)] = group_metrics
        subgroup_metrics[column] = column_results
    if subgroup_metrics:
        metrics["subgroups"] = subgroup_metrics
    table = prediction_frame(
        raw["patient_ids"],
        raw["study_ids"],
        raw["image_paths"],
        targets,
        probabilities,
        predictions,
        config["task"]["classes"],
        config["task"]["type"],
        raw.get("subgroups", {}),
    )
    output = ensure_dir(output_dir)
    table.to_csv(output / f"{split_name}_predictions.csv", index=False)
    save_json(metrics, output / f"{split_name}_metrics.json")
    save_diagnostic_plots(
        targets,
        probabilities,
        metrics,
        config["task"]["classes"],
        config["task"]["type"],
        output,
        split_name,
    )
    return metrics, table


def train_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> Path:
    output = ensure_dir(config["output"]["run_dir"])
    save_config(config, output / "resolved_config.yaml")
    pd.concat([train_frame, val_frame, test_frame], ignore_index=True).to_csv(
        output / "manifest_with_splits.csv", index=False
    )

    train_loader = create_loader(train_frame, config, training=True)
    val_loader = create_loader(val_frame, config, training=False)
    test_loader = create_loader(test_frame, config, training=False)
    model = build_model(config).to(device)
    freeze_epochs = int(config["model"].get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        freeze_backbone(model)
    criterion = build_loss(train_frame, config, device)
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, config)
    mixed = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=mixed)
    accumulation = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    patience = 0
    checkpoint_path = output / "best.pt"
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            unfreeze_all(model)
        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed,
            accumulation_steps=accumulation,
        )
        val_result = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            mixed_precision=mixed,
        )
        score, val_metrics = _selection_score(val_result, config)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(train_result["loss"]),
            "val_loss": float(val_result["loss"]),
            "val_macro_auroc": float(val_metrics["macro"]["auroc"]),
            "val_macro_auprc": float(val_metrics["macro"]["auprc"]),
            "learning_rate": learning_rate,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(
            f"Epoch {epoch:03d}: train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_macro_auroc={row['val_macro_auroc']:.4f}"
        )

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": serializable_config(config),
                    "best_epoch": best_epoch,
                    "selection_score": best_score,
                },
                checkpoint_path,
            )
        else:
            patience += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()
        if patience >= int(config["training"].get("early_stopping_patience", 12)):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    validation_raw = predict_loader(model, val_loader, device, mixed)
    temperature = 1.0
    if config["evaluation"].get("calibrate_probabilities", True):
        temperature = fit_temperature(
            validation_raw["logits"], validation_raw["targets"], config["task"]["type"]
        )
    validation_probabilities = logits_to_probabilities(
        apply_temperature(validation_raw["logits"], temperature).numpy(), config["task"]["type"]
    )
    thresholds = choose_thresholds(
        validation_raw["targets"].numpy(),
        validation_probabilities,
        config["task"]["type"],
        method=str(config["evaluation"].get("threshold_method", "youden")),
    )
    best.update({"temperature": temperature, "thresholds": thresholds})
    torch.save(best, checkpoint_path)

    val_metrics, _ = evaluate_and_save(
        model, val_loader, device, config, output, "val", temperature, thresholds
    )
    test_metrics, _ = evaluate_and_save(
        model, test_loader, device, config, output, "test", temperature, thresholds
    )
    summary = {
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "temperature": temperature,
        "thresholds": thresholds,
        "val_macro": val_metrics["macro"],
        "test_macro": test_metrics["macro"],
    }
    save_json(summary, output / "summary.json")
    return checkpoint_path
