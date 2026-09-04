from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def runtime_config(repository: Path, manifest: Path, output_root: Path) -> dict:
    """Keep sensitive artifacts ephemeral and honor the selected output mount."""
    template = repository / "configs" / "ct_rate_hf_pilot.yaml"
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    config["data"]["manifest"] = str(manifest)
    config["output"]["run_dir"] = str(output_root / "ct_rate_pilot_medicalnet")
    config["output"]["save_patient_level_artifacts"] = False
    return config


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
    if args.train_patients < 1 or args.valid_patients < 2:
        parser.error("--train-patients must be >= 1 and --valid-patients must be >= 2")

    repository = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest = Path("/tmp/ct-rate-pilot/manifest.csv")
    if not (dataset_root / "dataset" / "multi_abnormality_labels").exists():
        raise FileNotFoundError(f"CT-RATE mount is unavailable: {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    config_path = manifest.parent / "runtime_config.yaml"
    config_path.write_text(
        yaml.safe_dump(runtime_config(repository, manifest, output_root), sort_keys=False),
        encoding="utf-8",
    )

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
            str(config_path),
        ]
    )


if __name__ == "__main__":
    main()
