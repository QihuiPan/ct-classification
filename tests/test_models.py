import torch

import ct_classifier.models as model_module


class _FakeMedicalNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.constant_(self.conv1.weight, 2.0)
        self.in_planes = 16
        self.fc = None


def test_medicalnet_first_layer_is_adapted_for_two_windows(monkeypatch) -> None:
    received = {}

    def fake_resnet18(**kwargs):
        received.update(kwargs)
        return _FakeMedicalNet()

    monkeypatch.setattr(model_module, "resnet18", fake_resnet18)
    config = {
        "task": {"classes": ["normal", "cap", "covid19"]},
        "data": {"windows": [{"name": "lung"}, {"name": "mediastinum"}]},
        "model": {
            "architecture": "resnet18",
            "medicalnet_pretrained": True,
            "pretrained_checkpoint": None,
        },
    }

    model = model_module.build_model(config)

    assert received["pretrained"] is True
    assert received["n_input_channels"] == 1
    assert model.conv1.in_channels == 2
    assert torch.allclose(model.conv1.weight, torch.ones_like(model.conv1.weight))
    assert model.fc.in_features == 16
    assert model.fc.out_features == 3
