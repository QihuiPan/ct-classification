from __future__ import annotations

import argparse
from pathlib import Path

from ct_classifier.config import load_config
from ct_classifier.engine import train_model
from ct_classifier.split import create_splits, read_manifest, split_summary
from ct_classifier.utils import choose_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a 3D CT classifier")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 2026)))
    frame = create_splits(read_manifest(config), config)
    split_column = config["data"]["split_column"]
    groups = {name: frame[frame[split_column] == name].copy() for name in ("train", "val", "test")}
    empty = [name for name, group in groups.items() if group.empty]
    if empty:
        raise ValueError(f"Empty data splits: {empty}")

    print(split_summary(frame, config).to_string(index=False))
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    split_summary(frame, config).to_csv(run_dir / "split_summary.csv", index=False)
    device = choose_device()
    print(f"\nTraining device: {device}")
    checkpoint = train_model(groups["train"], groups["val"], groups["test"], config, device)
    print(f"\nCompleted. Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
