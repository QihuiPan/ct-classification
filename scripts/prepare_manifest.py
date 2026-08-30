from __future__ import annotations

import argparse
from pathlib import Path

from ct_classifier.config import load_config
from ct_classifier.split import create_splits, read_manifest, split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe patient-level train/val/test splits")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--output", help="Output CSV; defaults to <manifest>_with_split.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    frame = create_splits(read_manifest(config), config)
    source = Path(config["data"]["manifest"])
    output = Path(args.output).expanduser().resolve() if args.output else source.with_name(f"{source.stem}_with_split.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(split_summary(frame, config).to_string(index=False))
    print(f"\nSaved split manifest to: {output}")
    print("To reuse these exact assignments, set data.manifest to this file and split.mode to existing.")


if __name__ == "__main__":
    main()

