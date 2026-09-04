"""Cloud-only expanded research run; submission must independently enforce a 12h timeout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import yaml

from scripts.prepare_ct_rate_expanded import build_expanded_manifest
from scripts.run_ct_rate_hf_cloud import runtime_config


def expanded_config(repository: Path, temporary: Path, output: Path) -> dict:
    config = runtime_config(repository, temporary / "manifest.csv", output)
    config["data"].update(cache_dir=str(temporary / "cache"), cache_min_free_gb=20, num_workers=2)
    config["output"]["run_dir"] = str(output / "ct_rate_expanded_medicalnet")
    config["training"].update(epochs=20, early_stopping_patience=5, max_seconds=36000)
    return config


def main() -> None:
    if platform.system() != "Linux" or os.getenv("CT_RATE_EXPANDED_CLOUD") != "1":
        raise RuntimeError("Cloud-only entrypoint; do not run CT-RATE training on the local computer")
    dataset, output, temporary = Path("/mnt/ct-rate"), Path("/outputs"), Path("/tmp/ct-rate-expanded")
    if not dataset.is_dir() or not output.is_dir():
        raise RuntimeError("Required cloud mounts are missing")
    if shutil.disk_usage("/tmp").free < 40 * 1024**3:
        raise RuntimeError("At least 40 GiB free ephemeral disk is required before starting")
    repository = Path(__file__).resolve().parents[1]
    temporary.mkdir(exist_ok=True)
    config = expanded_config(repository, temporary, output)
    config_path = temporary / "runtime_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    frame = build_expanded_manifest(dataset)
    frame.to_csv(temporary / "manifest.csv", index=False)
    counts = frame.groupby("split").size().to_dict()
    if counts != {"train": 512, "val": 64, "test": 64}:
        raise RuntimeError("Cohort count verification failed")
    if len(frame) * 2 * 96 * 192 * 192 * 4 > 20 * 1024**3:
        raise RuntimeError("Uncompressed preprocessing tensor budget exceeded")
    provenance = {
        "status": "prepared", "cohort_patients": counts, "volumes_per_patient": 1,
        "selection": "SHA256(seed:patient), seed 2026; first scan and lowest reconstruction index",
        "previous_pilot_valid_patients_excluded": 16,
        "manifest_sha256": hashlib.sha256((temporary / "manifest.csv").read_bytes()).hexdigest(),
        "patient_level_artifacts_exported": False,
        "job_timeout_seconds_required": 43200, "training_soft_limit_seconds": 36000,
        "python": platform.python_version(),
        "source_git_sha": os.getenv("CT_SOURCE_GIT_SHA", "unrecorded"),
        "dataset_revision": os.getenv("CT_DATASET_REVISION", "unrecorded"),
    }
    provenance_path = output / "expanded_job.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(repository / "scripts/train.py"), "--config", str(config_path)], check=True)
    provenance["status"] = "completed"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
