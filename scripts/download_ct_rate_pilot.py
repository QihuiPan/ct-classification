from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "ibrahimhamamci/CT-RATE"
METADATA_PATTERNS = [
    "README.md",
    "dataset/metadata/**",
    "dataset/multi_abnormality_labels/**",
    "dataset/radiology_text_reports/**",
]


def patient_patterns(train_patients: int, valid_patients: int) -> list[str]:
    patterns = list(METADATA_PATTERNS)
    patterns.extend(f"dataset/train/train_{patient}/**" for patient in range(1, train_patients + 1))
    patterns.extend(f"dataset/valid/valid_{patient}/**" for patient in range(1, valid_patients + 1))
    return patterns


def selected_files(repo_files, patterns: list[str]) -> list[dict[str, object]]:
    selected = []
    for item in repo_files:
        name = item.rfilename
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            selected.append({"name": name, "size": int(item.size or 0)})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or download a bounded CT-RATE pilot subset to E:"
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--destination", default="E:/Codex/ct-classification/datasets/CT-RATE")
    parser.add_argument("--train-patients", type=int, default=48)
    parser.add_argument("--valid-patients", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true", help="Download after showing and validating the plan")
    args = parser.parse_args()
    if args.train_patients < 1 or args.valid_patients < 2:
        raise ValueError("Use at least 1 train patient and 2 validation patients")

    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type="dataset", files_metadata=True)
    patterns = patient_patterns(args.train_patients, args.valid_patients)
    files = selected_files(info.siblings, patterns)
    volume_files = [item for item in files if str(item["name"]).endswith(".nii.gz")]
    if not volume_files:
        raise RuntimeError(
            "No NIfTI files matched the expected official CT-RATE folder layout. "
            "Recheck repository access and file paths before downloading."
        )

    required_bytes = sum(int(item["size"]) for item in files)
    destination = Path(args.destination).expanduser().resolve()
    existing = destination
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    free_bytes = shutil.disk_usage(existing).free
    plan = {
        "repo_id": args.repo_id,
        "destination": str(destination),
        "train_patients_requested": args.train_patients,
        "valid_patients_requested": args.valid_patients,
        "matched_files": len(files),
        "matched_nifti_volumes": len(volume_files),
        "download_gb": required_bytes / 1_000_000_000,
        "free_gb": free_bytes / 1_000_000_000,
        "headroom_factor": 1.20,
        "fits": free_bytes >= int(required_bytes * 1.20),
    }
    print(json.dumps(plan, indent=2))
    if not plan["fits"]:
        raise SystemExit("Pilot selection does not fit with 20% free-space headroom")
    if not args.execute:
        print("Plan only. Re-run with --execute to download this exact selection.")
        return

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=patterns,
        max_workers=args.workers,
    )
    print(f"Downloaded CT-RATE pilot to {destination}")


if __name__ == "__main__":
    main()
