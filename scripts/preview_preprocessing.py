from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ct_classifier.config import load_config
from ct_classifier.preprocessing import preprocess_ct
from ct_classifier.split import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a visual QC montage after CT preprocessing")
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-id", help="Study to preview; defaults to the first manifest row")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = read_manifest(config)
    data = config["data"]
    study_column = data["study_id_column"]
    if args.study_id:
        selected = frame[frame[study_column].astype(str) == str(args.study_id)]
        if selected.empty:
            raise ValueError(f"study_id not found: {args.study_id}")
        row = selected.iloc[0]
    else:
        row = frame.iloc[0]

    series_column = data.get("dicom_series_id_column")
    series_id = (
        str(row[series_column])
        if series_column and series_column in row.index and pd.notna(row[series_column])
        else None
    )
    image, metadata = preprocess_ct(str(row[data["image_path_column"]]), data, series_id=series_id)
    array = image.numpy()
    depth = array.shape[1]
    slice_indices = np.linspace(max(0, depth * 0.15), max(0, depth * 0.85 - 1), 7).astype(int)
    channel_names = [window.get("name", f"window_{index}") for index, window in enumerate(data["windows"])]
    figure, axes = plt.subplots(len(channel_names), len(slice_indices), figsize=(3 * len(slice_indices), 3 * len(channel_names)), squeeze=False)
    for channel, name in enumerate(channel_names):
        for column, slice_index in enumerate(slice_indices):
            axes[channel, column].imshow(array[channel, slice_index], cmap="gray", vmin=0, vmax=1)
            axes[channel, column].set_title(f"{name}\nz={slice_index}")
            axes[channel, column].axis("off")
    figure.suptitle(
        f"Preprocessing QC — patient={row[data['patient_id_column']]} study={row[study_column]}\n"
        f"shape={tuple(array.shape)} spacing_zyx={tuple(data['target_spacing'])}"
    )
    figure.tight_layout()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"QC montage saved to: {output}")
    print(f"Source metadata: {metadata}")


if __name__ == "__main__":
    main()

