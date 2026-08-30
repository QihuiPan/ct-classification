from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


VALID_SPLITS = {"train", "val", "test"}


def read_manifest(config: dict[str, Any]) -> pd.DataFrame:
    manifest_path = Path(config["data"]["manifest"]).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Copy data/manifest_template.csv to data/manifest.csv first."
        )
    frame = pd.read_csv(manifest_path)
    image_column = config["data"]["image_path_column"]
    if image_column in frame.columns:
        def resolve_image(value: Any) -> str:
            path = Path(str(value)).expanduser()
            return str(path if path.is_absolute() else (manifest_path.parent / path).resolve())

        frame[image_column] = frame[image_column].map(resolve_image)
    return frame


def validate_manifest(df: pd.DataFrame, config: dict[str, Any], require_split: bool = False) -> None:
    data = config["data"]
    required = {
        data["patient_id_column"],
        data["study_id_column"],
        data["image_path_column"],
        *data["label_columns"],
    }
    required.update(config.get("evaluation", {}).get("subgroup_columns", []))
    if data.get("dicom_series_id_column"):
        required.add(data["dicom_series_id_column"])
    if config["split"]["mode"] == "external_site":
        required.add(data["site_column"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Manifest is empty")
    if df[list(required)].isnull().any().any():
        bad = df[list(required)].columns[df[list(required)].isnull().any()].tolist()
        raise ValueError(f"Manifest contains missing values in required columns: {bad}")
    if require_split:
        column = data["split_column"]
        if column not in df.columns:
            raise ValueError(f"Manifest does not contain split column '{column}'")
        unknown = set(df[column].astype(str).str.lower()).difference(VALID_SPLITS)
        if unknown:
            raise ValueError(f"Unknown split names: {sorted(unknown)}")


def _patient_table(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = config["data"]
    patient_col = data["patient_id_column"]
    label_cols = data["label_columns"]
    rows = []
    for patient_id, group in df.groupby(patient_col, sort=True):
        row: dict[str, Any] = {patient_col: patient_id}
        if config["task"]["type"] == "single_label":
            labels = group[label_cols[0]].astype(str).unique()
            row["_stratum"] = labels[0] if len(labels) == 1 else "mixed:" + "|".join(sorted(labels))
        else:
            values = group[label_cols].astype(int).max(axis=0).tolist()
            row["_stratum"] = "|".join(str(int(value)) for value in values)
        if data["site_column"] in group.columns:
            row[data["site_column"]] = str(group[data["site_column"]].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _usable_strata(strata: pd.Series, holdout_count: int) -> pd.Series | None:
    counts = strata.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2 or holdout_count < len(counts):
        return None
    return strata


def _split_ids(
    patient_table: pd.DataFrame,
    patient_col: str,
    holdout_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    ids = patient_table[patient_col].astype(str)
    if holdout_fraction <= 0:
        return set(ids), set()
    if holdout_fraction >= 1:
        return set(), set(ids)
    holdout_count = max(1, int(round(len(ids) * holdout_fraction)))
    strata = _usable_strata(patient_table["_stratum"], holdout_count)
    try:
        train_ids, holdout_ids = train_test_split(
            ids,
            test_size=holdout_fraction,
            random_state=seed,
            shuffle=True,
            stratify=strata,
        )
    except ValueError:
        train_ids, holdout_ids = train_test_split(
            ids,
            test_size=holdout_fraction,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return set(train_ids.astype(str)), set(holdout_ids.astype(str))


def create_splits(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    validate_manifest(df, config, require_split=config["split"]["mode"] == "existing")
    result = df.copy()
    data = config["data"]
    split_cfg = config["split"]
    patient_col = data["patient_id_column"]
    split_col = data["split_column"]
    result[patient_col] = result[patient_col].astype(str)

    if split_cfg["mode"] == "existing":
        result[split_col] = result[split_col].astype(str).str.strip().str.lower()
        _assert_patient_isolation(result, patient_col, split_col)
        return result

    patients = _patient_table(result, config)
    seed = int(config.get("seed", 2026))
    test_ids: set[str]

    if split_cfg["mode"] == "external_site":
        sites = {str(site) for site in split_cfg.get("external_test_sites", [])}
        if not sites:
            raise ValueError("external_site mode requires at least one external_test_sites entry")
        site_col = data["site_column"]
        test_ids = set(
            result.loc[result[site_col].astype(str).isin(sites), patient_col].astype(str)
        )
        if not test_ids:
            raise ValueError(f"No patients found for external test sites: {sorted(sites)}")
        remaining = patients[~patients[patient_col].astype(str).isin(test_ids)].copy()
    else:
        remaining_ids, test_ids = _split_ids(
            patients, patient_col, float(split_cfg["test_fraction"]), seed
        )
        remaining = patients[patients[patient_col].astype(str).isin(remaining_ids)].copy()

    train_fraction = float(split_cfg["train_fraction"])
    val_fraction = float(split_cfg["val_fraction"])
    relative_val = val_fraction / (train_fraction + val_fraction)
    train_ids, val_ids = _split_ids(remaining, patient_col, relative_val, seed + 1)

    result[split_col] = ""
    result.loc[result[patient_col].isin(train_ids), split_col] = "train"
    result.loc[result[patient_col].isin(val_ids), split_col] = "val"
    result.loc[result[patient_col].isin(test_ids), split_col] = "test"
    if (result[split_col] == "").any():
        raise RuntimeError("Some patients were not assigned to a split")
    _assert_patient_isolation(result, patient_col, split_col)
    return result


def _assert_patient_isolation(df: pd.DataFrame, patient_col: str, split_col: str) -> None:
    split_counts = df.groupby(patient_col)[split_col].nunique()
    leaking = split_counts[split_counts > 1].index.astype(str).tolist()
    if leaking:
        raise ValueError(f"Patient leakage across splits detected; examples: {leaking[:10]}")


def split_summary(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = config["data"]
    split_col = data["split_column"]
    patient_col = data["patient_id_column"]
    rows = []
    for split_name, group in df.groupby(split_col, sort=False):
        row = {
            "split": split_name,
            "patients": int(group[patient_col].nunique()),
            "studies": int(len(group)),
        }
        if config["task"]["type"] == "single_label":
            counts = group[data["label_columns"][0]].astype(str).value_counts()
            for label, count in counts.items():
                row[f"label_{label}"] = int(count)
        else:
            for label, column in zip(config["task"]["classes"], data["label_columns"]):
                row[f"positive_{label}"] = int(group[column].astype(int).sum())
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)
