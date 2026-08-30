from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import load_ct_image, resample_ct


def _body_crop(volume: np.ndarray, threshold_hu: float) -> np.ndarray:
    mask = volume > threshold_hu
    if not np.any(mask):
        return volume
    coordinates = np.where(mask)
    starts = [int(axis.min()) for axis in coordinates]
    stops = [int(axis.max()) + 1 for axis in coordinates]
    margins = [max(2, int(round((stop - start) * 0.05))) for start, stop in zip(starts, stops)]
    starts = [max(0, start - margin) for start, margin in zip(starts, margins)]
    stops = [min(size, stop + margin) for size, stop, margin in zip(volume.shape, stops, margins)]
    return volume[starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]]


def center_crop_or_pad(volume: np.ndarray, target_size: list[int], fill: float) -> np.ndarray:
    output = volume
    slices = []
    for size, target in zip(output.shape, target_size):
        start = max(0, (size - target) // 2)
        slices.append(slice(start, start + min(size, target)))
    output = output[tuple(slices)]

    padding: list[tuple[int, int]] = []
    for size, target in zip(output.shape, target_size):
        total = max(0, target - size)
        before = total // 2
        padding.append((before, total - before))
    return np.pad(output, padding, mode="constant", constant_values=fill)


def apply_window(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("CT window width must be positive")
    lower = center - width / 2.0
    upper = center + width / 2.0
    clipped = np.clip(volume, lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def _cache_key(path: str | Path, data_config: dict[str, Any], series_id: str | None = None) -> str:
    source = Path(path).expanduser().resolve()
    stamp = source.stat().st_mtime_ns
    relevant = {
        "source": str(source),
        "stamp": stamp,
        "series_id": series_id,
        "target_spacing": data_config["target_spacing"],
        "target_size": data_config["target_size"],
        "windows": data_config["windows"],
        "body_crop": data_config.get("body_crop", False),
        "body_threshold_hu": data_config.get("body_threshold_hu", -900),
    }
    return hashlib.sha256(repr(relevant).encode("utf-8")).hexdigest()


def preprocess_ct(
    path: str | Path,
    data_config: dict[str, Any],
    series_id: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    cache_dir_value = data_config.get("cache_dir")
    cache_path: Path | None = None
    if cache_dir_value:
        cache_dir = Path(cache_dir_value)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{_cache_key(path, data_config, series_id)}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                tensor = torch.from_numpy(cached["image"].astype(np.float32, copy=False))
            return tensor, {
                "source": str(Path(path).resolve()),
                "series_id": series_id,
                "cached": True,
                "preprocessed_shape_czyx": list(tensor.shape),
            }

    image, metadata = load_ct_image(path, series_id=series_id)
    image = resample_ct(image, data_config["target_spacing"])
    import SimpleITK as sitk

    volume = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    volume = np.nan_to_num(volume, nan=-1024.0, posinf=3071.0, neginf=-1024.0)
    if data_config.get("body_crop", False):
        volume = _body_crop(volume, float(data_config.get("body_threshold_hu", -900)))

    channels = []
    for window in data_config["windows"]:
        windowed = apply_window(volume, float(window["center"]), float(window["width"]))
        channels.append(center_crop_or_pad(windowed, data_config["target_size"], fill=0.0))
    stacked = np.stack(channels, axis=0).astype(np.float32, copy=False)
    tensor = torch.from_numpy(stacked)

    metadata.update(
        {
            "cached": False,
            "resampled_spacing_zyx": list(data_config["target_spacing"]),
            "preprocessed_shape_czyx": list(stacked.shape),
        }
    )
    if cache_path is not None:
        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, image=stacked)
        os.replace(temporary, cache_path)
    return tensor, metadata
