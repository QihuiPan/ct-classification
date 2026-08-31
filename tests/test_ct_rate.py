from pathlib import Path

import pandas as pd

from scripts.prepare_ct_rate import LABELS, build_manifest, parse_volume_name, validation_split


def test_official_label_order_is_complete_and_unique() -> None:
    assert len(LABELS) == 18
    assert len(set(LABELS)) == 18
    assert LABELS[0] == "Medical material"
    assert LABELS[-1] == "Interlobular septal thickening"


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
