from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError


REPO_ID = "ibrahimhamamci/CT-RATE"
DECIMAL_TB = 1_000_000_000_000


def nearest_existing_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"No existing parent for storage path: {path}")
    return candidate


def storage_report(path: str | Path, required_tb: float, headroom: float) -> dict[str, float | str | bool]:
    existing = nearest_existing_path(path)
    usage = shutil.disk_usage(existing)
    required_bytes = int(required_tb * DECIMAL_TB * headroom)
    return {
        "checked_path": str(existing),
        "free_tb": usage.free / DECIMAL_TB,
        "total_tb": usage.total / DECIMAL_TB,
        "required_tb_with_headroom": required_bytes / DECIMAL_TB,
        "sufficient": usage.free >= required_bytes,
        "shortfall_tb": max(0, required_bytes - usage.free) / DECIMAL_TB,
    }


def hugging_face_report(repo_id: str = REPO_ID) -> dict[str, object]:
    api = HfApi()
    result: dict[str, object] = {"repo_id": repo_id, "authenticated": False, "repository_visible": False}
    try:
        account = api.whoami()
        result["authenticated"] = True
        result["account"] = account.get("name", "authenticated")
    except LocalTokenNotFoundError:
        result["error"] = "Not logged in to Hugging Face. Run: hf auth login"
        return result
    except HfHubHTTPError as error:
        result["error"] = f"Hugging Face authentication check failed: {error.response.status_code}"
        return result

    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        result["repository_visible"] = True
        result["file_count"] = len(files)
        result["metadata_files_visible"] = any(
            name.endswith("train_predicted_labels.csv") for name in files
        )
    except HfHubHTTPError as error:
        result["error"] = (
            "Authenticated, but CT-RATE files are not accessible. Accept the dataset terms on "
            f"https://huggingface.co/datasets/{repo_id} and retry "
            f"(HTTP {error.response.status_code})."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CT-RATE authorization and full-storage readiness")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--storage-root", default="E:/Codex/ct-classification/datasets/CT-RATE")
    parser.add_argument("--dataset-size-tb", type=float, default=21.3)
    parser.add_argument("--headroom", type=float, default=1.15)
    args = parser.parse_args()

    disk = storage_report(args.storage_root, args.dataset_size_tb, args.headroom)
    access = hugging_face_report(args.repo_id)
    print("CT-RATE access")
    for key, value in access.items():
        print(f"  {key}: {value}")
    print("CT-RATE storage")
    for key, value in disk.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    ready = bool(access.get("repository_visible")) and bool(disk["sufficient"])
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
