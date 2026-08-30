from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


def _read_dicom_series(directory: Path, requested_series_id: str | None = None) -> tuple[sitk.Image, str]:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory))
    if not series_ids:
        raise ValueError(f"No readable DICOM series found in {directory}")

    candidates: list[tuple[int, str, list[str]]] = []
    for series_id in series_ids:
        names = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_id))
        candidates.append((len(names), series_id, names))
    if requested_series_id is not None:
        matches = [item for item in candidates if item[1] == requested_series_id]
        if not matches:
            raise ValueError(
                f"Requested DICOM SeriesInstanceUID '{requested_series_id}' was not found in {directory}"
            )
        _, selected_id, file_names = matches[0]
    elif len(candidates) == 1:
        _, selected_id, file_names = candidates[0]
    else:
        overview = ", ".join(f"{series_id} ({count} files)" for count, series_id, _ in candidates[:8])
        raise ValueError(
            "Multiple DICOM series found in one image_path. Point image_path to a single-series directory "
            f"or supply data.dicom_series_id_column. Found: {overview}"
        )
    if len(file_names) < 2:
        raise ValueError(f"Selected DICOM series contains fewer than two slices: {directory}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    return reader.Execute(), selected_id


def load_ct_image(path: str | Path, series_id: str | None = None) -> tuple[sitk.Image, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"CT input does not exist: {source}")

    if source.is_dir():
        image, selected_series_id = _read_dicom_series(source, requested_series_id=series_id)
        series_id = selected_series_id
        source_type = "dicom"
    else:
        lowered = source.name.lower()
        if not (lowered.endswith(".nii") or lowered.endswith(".nii.gz") or lowered.endswith(".mha") or lowered.endswith(".mhd")):
            raise ValueError(f"Unsupported CT file format: {source.name}")
        image = sitk.ReadImage(str(source), sitk.sitkFloat32)
        source_type = "volume"

    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D CT volume, got {image.GetDimension()}D: {source}")
    try:
        image = sitk.DICOMOrient(image, "LPS")
    except RuntimeError:
        pass
    image = sitk.Cast(image, sitk.sitkFloat32)
    array = sitk.GetArrayViewFromImage(image)
    if array.size == 0 or not np.isfinite(array).any():
        raise ValueError(f"CT volume is empty or non-finite: {source}")

    metadata = {
        "source": str(source),
        "source_type": source_type,
        "series_id": series_id,
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "direction": list(image.GetDirection()),
        "origin": list(image.GetOrigin()),
    }
    return image, metadata


def resample_ct(image: sitk.Image, target_spacing_zyx: list[float]) -> sitk.Image:
    target_spacing_xyz = tuple(float(x) for x in reversed(target_spacing_zyx))
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        max(1, int(round(size * spacing / new_spacing)))
        for size, spacing, new_spacing in zip(original_size, original_spacing, target_spacing_xyz)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_xyz)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(image)
