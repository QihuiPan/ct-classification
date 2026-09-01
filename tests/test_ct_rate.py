from pathlib import Path

import pandas as pd

from scripts.download_ct_rate_pilot import patient_tree_paths
from scripts.prepare_ct_rate import LABELS, build_manifest, parse_volume_name, validation_split


def test_official_label_order_is_complete_and_unique() -> None:
    assert len(LABELS) == 18
    assert len(set(LABELS)) == 18
    assert LABELS[0] == "Medical material"
    assert LABELS[-1] == "Interlobular septal thickening"


def test_pilot_tree_paths_are_bounded_to_requested_patients() -> None:
    paths = patient_tree_paths(train_patients=2, valid_patients=3)
    assert paths[-5:] == [
        "dataset/train/train_1",
        "dataset/train/train_2",
        "dataset/valid/valid_1",
        "dataset/valid/valid_2",
        "dataset/valid/valid_3",
    ]
    assert "dataset/train" not in paths
    assert "dataset/valid" not in paths


def test_parse_volume_name_with_or_without_extension() -> None:
    parsed = parse_volume_name("train_53_a_2.nii.gz")
    assert parsed["patient_id"] == "train_53"
    assert parsed["scan_id"] == "train_53_a"
    assert parsed["reconstruction"] == "2"
    assert parse_volume_name(Path("valid_7_b_1"))["source_split"] == "valid"


def test_build_manifest_preserves_patient_isolation(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    label_dir = dataset / "multi_abnormality_labels"
    label_dir.mkdir(parents=True)

    val_ids = {"val": None, "test": None}
    candidate = 1
    while None in val_ids.values():
        patient_id = f"valid_{candidate}"
        val_ids[validation_split(patient_id)] = candidate
        candidate += 1

    names = ["train_1_a_1", *[f"valid_{number}_a_1" for number in val_ids.values()]]
    for name in names:
        source = name.split("_", 1)[0]
        path = dataset / source / name.rsplit("_", 2)[0] / name
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.nii.gz").touch()

    def row(name: str, positive_index: int) -> dict[str, object]:
        result: dict[str, object] = {"VolumeName": f"{name}.nii.gz"}
        result.update({label: int(index == positive_index) for index, label in enumerate(LABELS)})
        return result

    pd.DataFrame([row(names[0], 0)]).to_csv(label_dir / "train_predicted_labels.csv", index=False)
    pd.DataFrame([row(name, index + 1) for index, name in enumerate(names[1:])]).to_csv(
        label_dir / "valid_predicted_labels.csv", index=False
    )

    manifest = build_manifest(tmp_path)
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert manifest.groupby("patient_id")["split"].nunique().max() == 1
    assert manifest[LABELS].sum().sum() == 3


def test_build_manifest_can_bound_patient_directories(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    label_dir = dataset / "multi_abnormality_labels"
    label_dir.mkdir(parents=True)
    names = ["train_1_a_1", "train_2_a_1", "valid_1_a_1", "valid_2_a_1"]
    for name in names:
        source, patient = name.split("_")[:2]
        patient_dir = dataset / source / f"{source}_{patient}" / name
        patient_dir.mkdir(parents=True, exist_ok=True)
        (patient_dir / f"{name}.nii.gz").touch()

    def row(name: str) -> dict[str, object]:
        result: dict[str, object] = {"VolumeName": f"{name}.nii.gz"}
        result.update({label: int(index == 0) for index, label in enumerate(LABELS)})
        return result

    pd.DataFrame([row(name) for name in names[:2]]).to_csv(
        label_dir / "train_predicted_labels.csv", index=False
    )
    pd.DataFrame([row(name) for name in names[2:]]).to_csv(
        label_dir / "valid_predicted_labels.csv", index=False
    )

    manifest = build_manifest(tmp_path, train_patients=1, valid_patients=2)
    assert "train_1" in set(manifest["patient_id"])
    assert "train_2" not in set(manifest["patient_id"])
