"""Deterministic expanded cohort: one volume per patient, no label-based selection."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scripts.prepare_ct_rate import LABELS, _dataset_dir, _label_table, parse_volume_name, validation_split


def select_cohort(tables: dict[str, pd.DataFrame], counts=(512, 64, 64), seed=2026) -> list[dict]:
    if len(counts) != 3 or any(not isinstance(n, int) or n < 1 for n in counts):
        raise ValueError("Three positive patient counts are required")
    pools = {"train": {}, "val": {}, "test": {}}
    for source, table in tables.items():
        for name in table.index:
            parsed = parse_volume_name(name)
            if parsed["source_split"] != source:
                raise ValueError("Label table source split mismatch")
            # The previous pilot's validation/test patients have already been inspected.
            if source == "valid" and int(parsed["patient"]) <= 16:
                continue
            split = "train" if source == "train" else validation_split(parsed["patient_id"])
            pools[split].setdefault(parsed["patient_id"], []).append(parsed)
    selected = []
    for split, count in zip(("train", "val", "test"), counts):
        pool = pools[split]
        if len(pool) < count:
            raise ValueError(f"Not enough eligible patients in {split}: {len(pool)} < {count}")
        ranked = sorted(pool, key=lambda patient: hashlib.sha256(f"{seed}:{patient}".encode()).hexdigest())
        for patient in ranked[:count]:
            parsed = min(pool[patient], key=lambda p: (p["scan"], int(p["reconstruction"])))
            selected.append({**parsed, "split": split})
    return selected


def build_expanded_manifest(dataset_root: Path, counts=(512, 64, 64), seed=2026) -> pd.DataFrame:
    dataset = _dataset_dir(dataset_root.resolve())
    label_dir = dataset / "multi_abnormality_labels"
    tables = {source: _label_table(label_dir / f"{source}_predicted_labels.csv") for source in ("train", "valid")}
    rows = []
    for selected in select_cohort(tables, counts, seed):
        source, patient, name = (selected[key] for key in ("source_split", "patient_id", "volume_name"))
        # Traverse only a selected patient's directory, never decode other CTs.
        paths = list((dataset / source / patient).rglob(f"{name}.nii.gz"))
        if len(paths) != 1:
            raise ValueError("A selected CT volume is missing or ambiguous; cohort is not silently changed")
        path = paths[0].resolve()
        if not path.is_relative_to(dataset):
            raise ValueError("CT path escaped the dataset mount")
        labels = tables[source].loc[name]
        rows.append({
            "patient_id": patient, "study_id": name, "scan_id": selected["scan_id"],
            "image_path": str(path), "site": "CT-RATE", "source_split": source,
            "split": selected["split"], **{label: int(labels[label]) for label in LABELS},
        })
    frame = pd.DataFrame(rows)
    if frame["patient_id"].duplicated().any():
        raise ValueError("Expanded cohort must have exactly one volume per patient")
    return frame
