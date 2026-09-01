from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "ibrahimhamamci/CT-RATE"
METADATA_PATTERNS = [
    "dataset/README.md",
    "dataset/data_correction_note.md",
    "dataset/metadata/**",
    "dataset/multi_abnormality_labels/**",
    "dataset/radiology_text_reports/**",
]
STATIC_FILES = [
    "dataset/README.md",
    "dataset/data_correction_note.md",
]
STATIC_TREE_PATHS = [
    "dataset/metadata",
    "dataset/multi_abnormality_labels",
    "dataset/radiology_text_reports",
]


def patient_patterns(train_patients: int, valid_patients: int) -> list[str]:
    patterns = list(METADATA_PATTERNS)
    patterns.extend(f"dataset/train/train_{patient}/**" for patient in range(1, train_patients + 1))
    patterns.extend(f"dataset/valid/valid_{patient}/**" for patient in range(1, valid_patients + 1))
    return patterns


def patient_tree_paths(train_patients: int, valid_patients: int) -> list[str]:
    paths = list(STATIC_TREE_PATHS)
    paths.extend(f"dataset/train/train_{patient}" for patient in range(1, train_patients + 1))
    paths.extend(f"dataset/valid/valid_{patient}" for patient in range(1, valid_patients + 1))
    return paths


def selected_files(repo_files, patterns: list[str]) -> list[dict[str, object]]:
    selected = []
    for item in repo_files:
        name = item.rfilename
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            selected.append({"name": name, "size": int(item.size or 0)})
    return selected


def targeted_repo_files(
    api: HfApi,
    repo_id: str,
    train_patients: int,
    valid_patients: int,
    workers: int,
) -> list[dict[str, object]]:
    """List only the bounded pilot paths instead of expanding the full 21.3 TB repository."""

    def list_tree(path: str):
        return list(
            api.list_repo_tree(
                repo_id,
                repo_type="dataset",
                path_in_repo=path,
                recursive=True,
                expand=False,
            )
        )

    paths = patient_tree_paths(train_patients, valid_patients)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        trees = list(pool.map(list_tree, paths))

    items = [item for tree in trees for item in tree]
    items.extend(api.get_paths_info(repo_id, STATIC_FILES, repo_type="dataset", expand=False))

    files: dict[str, dict[str, object]] = {}
    for item in items:
        name = getattr(item, "path", None) or getattr(item, "rfilename", None)
        size = getattr(item, "size", None)
        if name and size is not None:
            files[str(name)] = {"name": str(name), "size": int(size)}
    return [files[name] for name in sorted(files)]


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
    patterns = patient_patterns(args.train_patients, args.valid_patients)
    files = targeted_repo_files(
        api,
        args.repo_id,
        args.train_patients,
        args.valid_patients,
        args.workers,
    )
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
