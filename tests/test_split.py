from __future__ import annotations

import pandas as pd

from ct_classifier.split import create_splits


def _config() -> dict:
    return {
        "seed": 7,
        "task": {"type": "single_label", "classes": ["negative", "positive"]},
        "data": {
            "patient_id_column": "patient_id",
            "study_id_column": "study_id",
            "image_path_column": "image_path",
            "site_column": "site",
            "split_column": "split",
            "label_columns": ["label"],
        },
        "split": {
            "mode": "patient_random",
            "train_fraction": 0.6,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "external_test_sites": [],
        },
    }


def test_patient_never_crosses_splits() -> None:
    rows = []
    for patient in range(20):
        for study in range(2):
            rows.append(
                {
                    "patient_id": f"P{patient:03d}",
                    "study_id": f"P{patient:03d}-S{study}",
                    "image_path": f"volume-{patient}-{study}.nii.gz",
                    "site": "A",
                    "label": "positive" if patient % 2 else "negative",
                }
            )
    result = create_splits(pd.DataFrame(rows), _config())
    assert result.groupby("patient_id")["split"].nunique().max() == 1
    assert set(result["split"]) == {"train", "val", "test"}


def test_external_site_is_held_out_by_patient() -> None:
    config = _config()
    config["split"]["mode"] = "external_site"
    config["split"]["external_test_sites"] = ["B"]
    rows = []
    for patient in range(20):
        rows.append(
            {
                "patient_id": f"P{patient:03d}",
                "study_id": f"S{patient:03d}",
                "image_path": f"volume-{patient}.nii.gz",
                "site": "B" if patient >= 16 else "A",
                "label": "positive" if patient % 2 else "negative",
            }
        )
    result = create_splits(pd.DataFrame(rows), config)
    assert set(result.loc[result["site"] == "B", "split"]) == {"test"}
    assert "test" not in set(result.loc[result["site"] == "A", "split"])

