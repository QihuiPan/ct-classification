from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_ct_rate import LABELS, validation_split
from scripts.prepare_ct_rate_expanded import build_expanded_manifest, select_cohort
from scripts.run_ct_rate_hf_expanded import expanded_config, main


def tables():
    result = {}
    for source, numbers in (("train", range(1, 13)), ("valid", range(1, 80))):
        names = [f"{source}_{n}_{scan}_{r}" for n in numbers for scan, r in (("a", 1), ("a", 2), ("b", 1))]
        result[source] = pd.DataFrame(0, index=names, columns=LABELS)
    return result


def test_selection_is_deterministic_label_independent_and_patient_isolated():
    data = tables()
    a = select_cohort(data, (4, 3, 3))
    b = select_cohort({key: frame.iloc[::-1] + 1 for key, frame in data.items()}, (4, 3, 3))
    assert a == b
    assert len({row["patient_id"] for row in a}) == 10
    assert all(row["scan"] == "a" and row["reconstruction"] == "1" for row in a)
    for row in a:
        if row["source_split"] == "valid":
            assert int(row["patient"]) > 16
            assert row["split"] == validation_split(row["patient_id"])
    assert a != select_cohort(data, (4, 3, 3), seed=2027)


def test_insufficient_patients_is_not_silently_reduced():
    with pytest.raises(ValueError, match="Not enough"):
        select_cohort(tables(), (100, 3, 3))


def test_manifest_only_uses_selected_volumes_and_fails_on_missing(tmp_path):
    data = tables()
    label_dir = tmp_path / "dataset/multi_abnormality_labels"
    label_dir.mkdir(parents=True)
    for source, frame in data.items():
        frame.rename_axis("VolumeName").reset_index().to_csv(label_dir / f"{source}_predicted_labels.csv", index=False)
    selected = select_cohort(data, (4, 3, 3))
    paths = []
    for row in selected:
        path = tmp_path / "dataset" / row["source_split"] / row["patient_id"] / row["scan_id"] / (row["volume_name"] + ".nii.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        paths.append(path)
    frame = build_expanded_manifest(tmp_path, (4, 3, 3))
    assert len(frame) == frame.patient_id.nunique() == 10
    assert frame.groupby("split").size().to_dict() == {"train": 4, "val": 3, "test": 3}
    paths[0].unlink()
    with pytest.raises(ValueError, match="missing or ambiguous"):
        build_expanded_manifest(tmp_path, (4, 3, 3))


def test_expanded_configuration_has_budget_and_privacy_guards():
    config = expanded_config(Path(__file__).resolve().parents[1], Path("/tmp/expanded"), Path("/outputs"))
    assert config["training"]["max_seconds"] == 36000
    assert config["data"]["cache_min_free_gb"] == 20
    assert config["output"]["save_patient_level_artifacts"] is False
    assert config["output"]["run_dir"].endswith("ct_rate_expanded_medicalnet")


def test_cloud_runner_rejects_local_execution(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setenv("CT_RATE_EXPANDED_CLOUD", "1")
    with pytest.raises(RuntimeError, match="Cloud-only"):
        main()
