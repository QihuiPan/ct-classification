from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded CT-RATE pilot on Hugging Face Jobs")
    parser.add_argument("--dataset-root", default="/mnt/ct-rate")
    parser.add_argument("--output-root", default="/outputs")
    parser.add_argument("--train-patients", type=int, default=48)
    parser.add_argument("--valid-patients", type=int, default=16)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest = Path("/tmp/ct-rate-pilot/manifest.csv")
    if not (dataset_root / "dataset" / "multi_abnormality_labels").exists():
        raise FileNotFoundError(f"CT-RATE mount is unavailable: {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    provenance = {
        "job_id": os.getenv("HF_JOB_ID"),
        "dataset_mount": str(dataset_root),
        "train_patients": args.train_patients,
        "valid_patients": args.valid_patients,
        "patient_level_artifacts_exported": False,
    }
    (output_root / "cloud_job.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    run(
        [
            sys.executable,
            str(repository / "scripts" / "prepare_ct_rate.py"),
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(manifest),
            "--train-patients",
            str(args.train_patients),
            "--valid-patients",
            str(args.valid_patients),
        ]
    )
    run(
        [
            sys.executable,
            str(repository / "scripts" / "train.py"),
            "--config",
            str(repository / "configs" / "ct_rate_hf_pilot.yaml"),
        ]
    )


if __name__ == "__main__":
    main()
