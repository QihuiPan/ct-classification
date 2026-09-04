import hashlib
from pathlib import Path

import pytest

from scripts.reassemble_checkpoint import reassemble


@pytest.fixture
def parts(tmp_path: Path):
    chunks = [b"synthetic model part one", b"synthetic model part two"]
    entries = []
    for i, chunk in enumerate(chunks, 1):
        name = f"best.pt.part{i:02d}"
        (tmp_path / name).write_bytes(chunk)
        entries.append(f"{hashlib.sha256(chunk).hexdigest()}  {name}")
    (tmp_path / "best.pt.parts.sha256").write_text("\n".join(entries), encoding="utf-8")
    payload = b"".join(chunks)
    (tmp_path / "best.pt.sha256").write_text(f"{hashlib.sha256(payload).hexdigest()}  best.pt", encoding="utf-8")
    return tmp_path, payload


def test_verifies_and_reassembles_identical_bytes(parts):
    directory, payload = parts
    output = directory / "best.pt"
    assert reassemble(directory) == hashlib.sha256(payload).hexdigest()
    assert not output.exists()
    reassemble(directory, output)
    assert output.read_bytes() == payload
    assert not list(directory.glob("*.partial"))


def test_never_overwrites_an_existing_model(parts):
    directory, _ = parts
    output = directory / "best.pt"
    output.write_bytes(b"user-owned model")
    with pytest.raises(FileExistsError):
        reassemble(directory, output)
    assert output.read_bytes() == b"user-owned model"


@pytest.mark.parametrize("problem", ["corrupt", "missing", "wrong_full_hash", "reordered", "traversal", "duplicate"])
def test_bad_release_parts_never_publish_output(parts, problem):
    directory, _ = parts
    manifest = directory / "best.pt.parts.sha256"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if problem == "corrupt":
        (directory / "best.pt.part02").write_bytes(b"corrupt")
    elif problem == "missing":
        (directory / "best.pt.part02").unlink()
    elif problem == "wrong_full_hash":
        (directory / "best.pt.sha256").write_text("0" * 64 + "  best.pt", encoding="utf-8")
    elif problem == "reordered":
        manifest.write_text("\n".join(reversed(lines)), encoding="utf-8")
    elif problem == "traversal":
        manifest.write_text(lines[0].replace("best.pt.part01", "../best.pt.part01"), encoding="utf-8")
    elif problem == "duplicate":
        manifest.write_text("\n".join([lines[0], lines[0]]), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        reassemble(directory, directory / "best.pt")
    assert not (directory / "best.pt").exists()
    assert not list(directory.glob("*.partial"))
