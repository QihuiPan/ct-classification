from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import SimpleITK as sitk


EXPECTED_COUNTS = {"covid19": 169, "cap": 60, "normal": 76}
LABEL_ALIASES = {
    "covid19": "covid19",
    "covid19cases": "covid19",
    "covid19subjects": "covid19",
    "cap": "cap",
    "capcases": "cap",
    "capsubjects": "cap",
    "communityacquiredpneumonia": "cap",
    "normal": "normal",
    "normalcases": "normal",
    "normalsubjects": "normal",
}


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def label_from_directory(directory: Path) -> str | None:
    return LABEL_ALIASES.get(_normalise_name(directory.name))


def _find_label_directories(root: Path) -> list[tuple[str, Path]]:
    matches: list[tuple[str, Path]] = []
    for directory in [root, *sorted(path for path in root.rglob("*") if path.is_dir())]:
        if any(part.upper() == "__MACOSX" for part in directory.parts):
            continue
        label = label_from_directory(directory)
        if label is not None:
            matches.append((label, directory))
    # Avoid matching a nested duplicate label directory twice.
    selected: list[tuple[str, Path]] = []
    for label, directory in matches:
        if not any(directory.is_relative_to(parent) for _, parent in selected):
            selected.append((label, directory))
    return selected


def _series_candidates(patient_directory: Path) -> list[tuple[int, str, Path, list[str]]]:
    candidates: list[tuple[int, str, Path, list[str]]] = []
    directories = [patient_directory, *sorted(path for path in patient_directory.rglob("*") if path.is_dir())]
    for directory in directories:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
        for series_id in series_ids:
            names = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_id))
            if len(names) >= 2:
                candidates.append((len(names), str(series_id), directory, names))
    return candidates


def _metadata(file_path: str) -> dict[str, str]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(file_path)
    reader.LoadPrivateTagsOff()
    reader.ReadImageInformation()

    def value(tag: str) -> str:
        return reader.GetMetaData(tag).strip() if reader.HasMetaDataKey(tag) else ""

    raw_sex = value("0010|0040").upper()
    sex = raw_sex if raw_sex in {"M", "F"} else "unknown"
    raw_age = value("0010|1010").upper()
    age_years = _age_in_years(raw_age)
    return {
        "sex": sex,
        "age_years": "" if age_years is None else f"{age_years:.3f}".rstrip("0").rstrip("."),
        "age_group": _age_group(age_years),
    }


def _age_in_years(raw_age: str) -> float | None:
    match = re.fullmatch(r"(\d{1,3})([DWMY])", raw_age.strip())
    if not match:
        return None
    value = float(match.group(1))
    return value * {"D": 1 / 365.25, "W": 7 / 365.25, "M": 1 / 12, "Y": 1}[match.group(2)]


def _age_group(age: float | None) -> str:
    if age is None:
        return "unknown"
    if age < 40:
        return "under_40"
    if age < 60:
        return "40_to_59"
    return "60_plus"


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_directories = _find_label_directories(dataset_root)
    if not label_directories:
        raise ValueError(f"Could not find COVID-19, CAP, or Normal class directories below {dataset_root}")

    for label, label_directory in label_directories:
        patient_directories = sorted(
            path for path in label_directory.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
        for patient_directory in patient_directories:
            candidates = _series_candidates(patient_directory)
            if not candidates:
                raise ValueError(f"No readable DICOM series found for {patient_directory}")
            slice_count, series_uid, series_directory, file_names = max(candidates, key=lambda item: item[0])
            patient_key = f"{label}_{patient_directory.name}"
            rows.append(
                {
                    "patient_id": patient_key,
                    "study_id": patient_key,
                    "image_path": str(series_directory.resolve()),
                    "series_instance_uid": series_uid,
                    "site": "COVID_CT_MD_single_centre",
                    "label": label,
                    "slice_count": slice_count,
                    **_metadata(file_names[0]),
                }
            )

    frame = pd.DataFrame(rows).sort_values(["label", "patient_id"]).reset_index(drop=True)
    duplicates = frame["patient_id"].duplicated(keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, "patient_id"].tolist()[:10]
        raise ValueError(f"Duplicate patient identifiers detected: {examples}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a patient-level manifest for COVID-CT-MD")
    parser.add_argument("--dataset-root", required=True, help="Extracted COVID-CT-MD directory")
    parser.add_argument("--output", required=True, help="Destination manifest CSV")
    parser.add_argument(
        "--allow-unexpected-counts",
        action="store_true",
        help="Write a manifest even when class counts differ from the published 169/60/76 cases",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {dataset_root}")
    frame = build_manifest(dataset_root)
    counts = Counter(frame["label"].astype(str))
    if not args.allow_unexpected_counts and dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Published class counts are {EXPECTED_COUNTS}, but discovered {dict(counts)}")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.groupby("label").agg(patients=("patient_id", "nunique"), slices=("slice_count", "sum")))
    print(f"\nSaved {len(frame)} patient-level studies to: {output}")


if __name__ == "__main__":
    main()
