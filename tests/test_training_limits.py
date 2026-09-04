import time

import pytest
import torch

from ct_classifier.engine import run_epoch


def test_expired_deadline_stops_before_forward_or_parameter_update():
    model = torch.nn.Linear(2, 2)
    before = model.weight.detach().clone()
    loader = [{"image": torch.ones(1, 2), "target": torch.tensor([1])}]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(TimeoutError, match="soft time limit"):
        run_epoch(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"),
                  optimizer=optimizer, deadline=time.monotonic() - 1)
    assert torch.equal(before, model.weight)


def test_low_disk_stops_before_ct_read(tmp_path, monkeypatch):
    import ct_classifier.preprocessing as preprocessing
    from collections import namedtuple
    usage = namedtuple("Usage", "total used free")
    source = tmp_path / "synthetic.nii.gz"
    source.touch()
    config = dict(cache_dir=str(tmp_path / "cache"), cache_min_free_gb=20,
                  target_spacing=[3, 1.5, 1.5], target_size=[4, 4, 4], windows=[])
    monkeypatch.setattr(preprocessing.shutil, "disk_usage", lambda path: usage(100, 99, 1))
    monkeypatch.setattr(preprocessing, "load_ct_image", lambda *a, **k: pytest.fail("CT should not be read"))
    with pytest.raises(RuntimeError, match="disk headroom"):
        preprocessing.preprocess_ct(source, config)


def test_soft_limit_evaluates_only_best_completed_epoch(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import ct_classifier.engine as engine
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(engine, "build_model", lambda config: model)
    monkeypatch.setattr(engine, "create_loader", lambda *a, **k: [])
    monkeypatch.setattr(engine, "build_loss", lambda *a: torch.nn.CrossEntropyLoss())
    raw = {"logits": torch.tensor([[2., -2.], [-2., 2.]]), "targets": torch.tensor([0, 1]), "loss": 0.1}
    calls = 0

    def limited_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise TimeoutError("synthetic deadline")
        return raw

    monkeypatch.setattr(engine, "run_epoch", limited_epoch)
    monkeypatch.setattr(engine, "_selection_score", lambda *a: (0.75, {"macro": {"auroc": 0.75, "auprc": 0.7}}))
    monkeypatch.setattr(engine, "predict_loader", lambda *a: raw)
    monkeypatch.setattr(engine, "evaluate_and_save", lambda *a: ({"macro": {"auroc": 0.75}}, pd.DataFrame()))
    config = {
        "task": {"type": "single_label", "classes": ["a", "b"]}, "model": {},
        "training": {"learning_rate": 0.01, "scheduler": "none", "epochs": 5,
                     "mixed_precision": False, "max_seconds": 100},
        "evaluation": {"calibrate_probabilities": False, "threshold_method": "fixed"},
        "output": {"run_dir": str(tmp_path), "save_patient_level_artifacts": False},
    }
    engine.train_model(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), config, torch.device("cpu"))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["epochs_completed"] == summary["best_epoch"] == 1
    assert summary["stop_reason"] == "training_time_limit"
    assert not (tmp_path / "manifest_with_splits.csv").exists()
