from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ct_classifier.calibration import apply_temperature
from ct_classifier.gradcam import GradCAM3D, save_gradcam_montage
from ct_classifier.metrics import logits_to_probabilities, predictions_from_probabilities
from ct_classifier.models import build_model
from ct_classifier.preprocessing import preprocess_ct
from ct_classifier.utils import choose_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one DICOM series or NIfTI CT volume")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--series-id", help="SeriesInstanceUID when the directory contains multiple DICOM series")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gradcam", help="Optional Grad-CAM montage PNG")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = deepcopy(payload["config"])
    model_config = deepcopy(config)
    model_config["model"]["pretrained_checkpoint"] = None
    device = choose_device()
    model = build_model(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    image, metadata = preprocess_ct(args.image, config["data"], series_id=args.series_id)
    batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch).float().cpu()
    temperature = float(payload.get("temperature", 1.0))
    probabilities = logits_to_probabilities(
        apply_temperature(logits, temperature).numpy(), config["task"]["type"]
    )
    thresholds = [float(value) for value in payload.get("thresholds", [0.5] * probabilities.shape[1])]
    predictions = predictions_from_probabilities(probabilities, config["task"]["type"], thresholds)
    classes = config["task"]["classes"]
    probability_map = {name: float(probabilities[0, index]) for index, name in enumerate(classes)}

    if config["task"]["type"] == "single_label":
        predicted_index = int(predictions[0])
        predicted = classes[predicted_index]
        confidence = float(probabilities[0, predicted_index])
        low_confidence = confidence < float(config["evaluation"].get("abstain_confidence", 0.60))
        prediction_payload = {
            "predicted_class": predicted,
            "predicted_index": predicted_index,
            "confidence": confidence,
            "low_confidence": low_confidence,
        }
        gradcam_index = predicted_index
    else:
        active = [classes[index] for index, value in enumerate(predictions[0]) if value == 1]
        margins = np.abs(probabilities[0] - np.asarray(thresholds))
        low_confidence = bool(np.any(margins < float(config["evaluation"].get("abstain_margin", 0.10))))
        prediction_payload = {"positive_classes": active, "low_confidence": low_confidence}
        gradcam_index = int(np.argmax(probabilities[0]))

    result = {
        "image": str(Path(args.image).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "task_type": config["task"]["type"],
        "probabilities": probability_map,
        "thresholds": {name: thresholds[index] for index, name in enumerate(classes)},
        "temperature": temperature,
        **prediction_payload,
        "preprocessing": metadata,
        "warning": "Research/decision-support output only; not a standalone clinical diagnosis.",
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output_path)

    if args.gradcam:
        gradcam_path = Path(args.gradcam).expanduser().resolve()
        gradcam_path.parent.mkdir(parents=True, exist_ok=True)
        cam_engine = GradCAM3D(model)
        cam, _ = cam_engine.generate(batch, gradcam_index)
        cam_engine.close()
        save_gradcam_montage(
            batch,
            cam,
            gradcam_path,
            title=f"Grad-CAM: {classes[gradcam_index]} (not a segmentation mask)",
        )
        result["gradcam"] = str(gradcam_path)
        save_json(result, output_path)
    print(f"Prediction saved to: {output_path}")


if __name__ == "__main__":
    main()
