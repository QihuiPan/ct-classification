from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(float(temperature), 1e-6)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, task_type: str) -> float:
    logits = logits.detach().float().cpu()
    targets = targets.detach().cpu()
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        calibrated = logits / temperature
        if task_type == "single_label":
            loss = F.cross_entropy(calibrated, targets.long())
        else:
            loss = F.binary_cross_entropy_with_logits(calibrated, targets.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())

