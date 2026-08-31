from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


LABELS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

VOLUME_PATTERN = re.compile(
    r"^(?P<source_split>train|valid)_(?P<patient>\d+)_(?P<scan>[A-Za-z]+)_(?P<reconstruction>\d+)$"
)


def normalize_volume_name(value: str | Path) -> str:
    name = Path(str(value)).name
    return name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem


def parse_volume_name(value: str | Path) -> dict[str, str]:
    volume_name = normalize_volume_name(value)
    match = VOLUME_PATTERN.fullmatch(volume_name)
    if match is None:
        raise ValueError(
            f"Unexpected CT-RATE volume name '{value}'. Expected split_patient_scan_reconstruction."
        )
    parts = match.groupdict()
    return {
        **parts,
        "volume_name": volume_name,
        "patient_id": f"{parts['source_split']}_{parts['patient']}",
        "scan_id": f"{parts['source_split']}_{parts['patient']}_{parts['scan']}",
    }


def validation_split(patient_id: str) -> str:
    digest = hashlib.sha256(patient_id.encode("utf-8")).digest()
    return "val" if int.from_bytes(digest[:8], "big") % 2 == 0 else "test"


def _dataset_dir(root: Path) -> Path:
    nested = root / "dataset"
    return nested if nested.exists() else root


def _label_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CT-RATE labels not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in ["VolumeName", *LABELS] if column not in frame.columns]
    if missing:
        raise ValueError(f"Label file {path} is missing columns: {missing}")
    frame = frame[["VolumeName", *LABELS]].copy()
    frame["_volume_key"] = frame["VolumeName"].map(normalize_volume_name)
    if frame["_volume_key"].duplicated().any():
        duplicates = frame.loc[frame["_volume_key"].duplicated(), "_volume_key"].head().tolist()
        raise ValueError(f"Duplicate VolumeName values in {path}: {duplicates}")
    values = frame[LABELS].astype(float)
    if not values.isin([0.0, 1.0]).all().all():
        raise ValueError(f"CT-RATE labels must be binary 0/1 values: {path}")
    frame[LABELS] = values.astype(int)
    return frame.set_index("_volume_key", drop=True)


def _volume_paths(dataset_dir: Path, source_split: str) -> list[Path]:
    root = dataset_dir / source_split
    if not root.exists():
        return []
    return sorted(root.rglob(f"{source_split}_*.nii.gz"))


def build_manifest(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root).expanduser().resolve()
    dataset_dir = _dataset_dir(root)
    label_dir = dataset_dir / "multi_abnormality_labels"
    tables = {
        "train": _label_table(label_dir / "train_predicted_labels.csv"),
        "valid": _label_table(label_dir / "valid_predicted_labels.csv"),
    }

    rows: list[dict[str, object]] = []
    missing_labels: list[str] = []
    for source_split in ("train", "valid"):
        table = tables[source_split]
        for image_path in _volume_paths(dataset_dir, source_split):
            parsed = parse_volume_name(image_path)
            key = parsed["volume_name"]
            if key not in table.index:
                missing_labels.append(key)
                continue
            label_row = table.loc[key]
            split = "train" if source_split == "train" else validation_split(parsed["patient_id"])
            row: dict[str, object] = {
                "patient_id": parsed["patient_id"],
                "study_id": parsed["volume_name"],
                "scan_id": parsed["scan_id"],
                "reconstruction_id": parsed["reconstruction"],
                "image_path": str(image_path.resolve()),
                "site": "CT-RATE",
                "source_split": source_split,
                "split": split,
            }
            row.update({label: int(label_row[label]) for label in LABELS})
            rows.append(row)

    if missing_labels:
        examples = ", ".join(missing_labels[:5])
        raise ValueError(
            f"{len(missing_labels)} downloaded volumes have no matching label row; examples: {examples}"
        )
    if not rows:
        raise FileNotFoundError(
            f"No CT-RATE .nii.gz volumes found below {dataset_dir / 'train'} or {dataset_dir / 'valid'}"
        )
    frame = pd.DataFrame(rows).sort_values(["split", "patient_id", "study_id"]).reset_index(drop=True)
    leaking = frame.groupby("patient_id")["split"].nunique()
    if (leaking > 1).any():
        raise RuntimeError("Patient leakage detected while creating CT-RATE manifest")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a patient-isolated CT-RATE manifest")
    parser.add_argument(
        "--dataset-root",
        default="E:/Codex/ct-classification/datasets/CT-RATE",
        help="Hugging Face snapshot root or its dataset/ directory",
    )
    parser.add_argument(
        "--output",
        default="E:/Codex/ct-classification/datasets/CT-RATE/manifest.csv",
    )
    args = parser.parse_args()

    frame = build_manifest(args.dataset_root)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = frame.groupby("split").agg(patients=("patient_id", "nunique"), volumes=("study_id", "size"))
    print(summary.to_string())
    print("\nPositive labels by split:")
    print(frame.groupby("split")[LABELS].sum().to_string())
    print(f"\nManifest: {output}")


if __name__ == "__main__":
    main()
