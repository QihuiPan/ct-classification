from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from ct_classifier.dataset import create_loader
from ct_classifier.engine import evaluate_and_save
from ct_classifier.models import build_model
from ct_classifier.split import create_splits, read_manifest
from ct_classifier.utils import choose_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CT classifier")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--manifest", help="Optional replacement manifest")
    parser.add_argument("--output-dir", help="Optional output directory")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = deepcopy(payload["config"])
    if args.manifest:
        config["data"]["manifest"] = str(Path(args.manifest).expanduser().resolve())
    seed_everything(int(config.get("seed", 2026)))
    frame = create_splits(read_manifest(config), config)
    split_col = config["data"]["split_column"]
    selected = frame[frame[split_col] == args.split].copy()
    if selected.empty:
        raise ValueError(f"The requested split '{args.split}' is empty")

    model_config = deepcopy(config)
    model_config["model"]["pretrained_checkpoint"] = None
    device = choose_device()
    model = build_model(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    loader = create_loader(selected, config, training=False)
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint_path.parent / "reevaluation"
    metrics, _ = evaluate_and_save(
        model,
        loader,
        device,
        config,
        output,
        args.split,
        float(payload.get("temperature", 1.0)),
        [float(value) for value in payload.get("thresholds", [0.5] * len(config["task"]["classes"]))],
    )
    print(f"Macro metrics: {metrics['macro']}")
    print(f"Saved evaluation to: {output}")


if __name__ == "__main__":
    main()

