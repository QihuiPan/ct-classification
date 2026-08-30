from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ct_classifier.config import load_config
from ct_classifier.preprocessing import preprocess_ct
from ct_classifier.split import create_splits, read_manifest, split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manifest, patient isolation, and CT readability")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Only decode this many studies; 0 checks all")
    parser.add_argument("--output", help="Optional per-study validation CSV")
    args = parser.parse_args()

    config = load_config(args.config)
    frame = create_splits(read_manifest(config), config)
    data = config["data"]
    study_col = data["study_id_column"]
    duplicate_studies = frame[study_col].astype(str).duplicated(keep=False)
    if duplicate_studies.any():
        examples = frame.loc[duplicate_studies, study_col].astype(str).unique()[:10].tolist()
        raise ValueError(f"Duplicate study_id values detected; examples: {examples}")

    rows = []
    subset = frame if args.limit <= 0 else frame.head(args.limit)
    for _, item in subset.iterrows():
        path = str(item[data["image_path_column"]])
        result = {
            "patient_id": str(item[data["patient_id_column"]]),
            "study_id": str(item[study_col]),
            "image_path": path,
            "ok": False,
            "shape_czyx": "",
            "error": "",
        }
        try:
            series_column = data.get("dicom_series_id_column")
            series_id = (
                str(item[series_column])
                if series_column and series_column in item.index and pd.notna(item[series_column])
                else None
            )
            image, _ = preprocess_ct(path, data, series_id=series_id)
            result["ok"] = True
            result["shape_czyx"] = "x".join(str(value) for value in image.shape)
        except Exception as error:  # keep auditing remaining studies
            result["error"] = f"{type(error).__name__}: {error}"
        rows.append(result)

    report = pd.DataFrame(rows)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(config["output"]["run_dir"]) / "data_validation.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(split_summary(frame, config).to_string(index=False))
    print(f"\nDecoded studies: {len(report)}; passed: {int(report['ok'].sum())}; failed: {int((~report['ok']).sum())}")
    print(f"Validation report: {output}")
    if not report["ok"].all():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
