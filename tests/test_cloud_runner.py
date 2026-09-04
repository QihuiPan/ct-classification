from pathlib import Path

import pytest
import yaml

from scripts import run_ct_rate_hf_cloud as cloud


def test_cloud_config_respects_output_mount_and_disables_patient_export(tmp_path):
    repo = tmp_path / "source"
    (repo / "configs").mkdir(parents=True)
    template = {
        "data": {"manifest": "/old/manifest.csv", "cache_dir": "/tmp/cache"},
        "output": {"run_dir": "/old/results", "save_patient_level_artifacts": True},
    }
    path = repo / "configs" / "ct_rate_hf_pilot.yaml"
    path.write_text(yaml.safe_dump(template), encoding="utf-8")
    manifest = tmp_path / "ephemeral" / "manifest.csv"
    output = tmp_path / "custom-output"
    config = cloud.runtime_config(repo, manifest, output)
    assert config["data"]["manifest"] == str(manifest)
    assert config["output"]["run_dir"] == str(output / "ct_rate_pilot_medicalnet")
    assert config["output"]["save_patient_level_artifacts"] is False
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == template


@pytest.mark.parametrize("args", [["--train-patients", "0"], ["--valid-patients", "1"]])
def test_invalid_patient_limits_stop_before_any_work(monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["cloud", *args])
    with pytest.raises(SystemExit) as error:
        cloud.main()
    assert error.value.code == 2


def test_repository_cloud_template_has_only_ephemeral_patient_paths():
    repo = Path(__file__).resolve().parents[1]
    config = cloud.runtime_config(repo, Path("/tmp/pilot/manifest.csv"), Path("/outputs"))
    assert config["data"]["cache_dir"].startswith("/tmp/")
    assert config["output"]["save_patient_level_artifacts"] is False
