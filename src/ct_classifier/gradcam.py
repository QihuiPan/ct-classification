from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def find_last_conv3d(model: torch.nn.Module) -> torch.nn.Conv3d:
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv3d)]
    if not layers:
        raise ValueError("The model does not contain a Conv3d layer for Grad-CAM")
    return layers[-1]


class GradCAM3D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or find_last_conv3d(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def generate(self, image: torch.Tensor, target_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[0, target_index].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[2:], mode="trilinear", align_corners=False)
        minimum = cam.amin(dim=(2, 3, 4), keepdim=True)
        maximum = cam.amax(dim=(2, 3, 4), keepdim=True)
        cam = (cam - minimum) / (maximum - minimum + 1e-8)
        return cam.detach(), logits.detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def save_gradcam_montage(
    image: torch.Tensor,
    cam: torch.Tensor,
    output_path: str | Path,
    title: str,
    slices: int = 9,
) -> None:
    base = image.detach().cpu().numpy()[0].mean(axis=0)
    heatmap = cam.detach().cpu().numpy()[0, 0]
    scores = heatmap.mean(axis=(1, 2))
    selected = np.sort(np.argsort(scores)[-min(slices, len(scores)) :])
    columns = 3
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), squeeze=False)
    for axis, index in zip(axes.ravel(), selected):
        axis.imshow(base[index], cmap="gray")
        axis.imshow(heatmap[index], cmap="jet", alpha=0.42, vmin=0, vmax=1)
        axis.set_title(f"z={int(index)}")
        axis.axis("off")
    for axis in axes.ravel()[len(selected) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

