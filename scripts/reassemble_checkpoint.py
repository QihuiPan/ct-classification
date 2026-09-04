"""Verify and join release model parts without network access or overwriting files."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import tempfile


def checksums(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^\s/\\]+)", line.strip())
        if not match:
            raise ValueError(f"Invalid checksum entry in {path.name}")
        entries.append((match[1].lower(), match[2]))
    if not entries or len({name for _, name in entries}) != len(entries):
        raise ValueError(f"Empty or duplicate checksum entries in {path.name}")
    return entries


def reassemble(parts_dir: Path, output: Path | None = None) -> str:
    parts_dir = parts_dir.resolve()
    parts = checksums(parts_dir / "best.pt.parts.sha256")
    full = checksums(parts_dir / "best.pt.sha256")
    if len(full) != 1 or full[0][1] != "best.pt":
        raise ValueError("The full checksum manifest must contain only best.pt")
    if [name for _, name in parts] != [f"best.pt.part{i:02d}" for i in range(1, len(parts) + 1)]:
        raise ValueError("Parts must be contiguous and ordered from best.pt.part01")
    if output is not None:
        output = output.absolute()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {output}")
        if not output.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {output.parent}")

    temporary = None
    handle = None
    full_digest = hashlib.sha256()
    try:
        if output is not None:
            handle = tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, suffix=".partial", delete=False)
            temporary = Path(handle.name)
        for expected, name in parts:
            part = (parts_dir / name).resolve()
            if part.parent != parts_dir:
                raise ValueError("Part path escaped the selected directory")
            digest = hashlib.sha256()
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    full_digest.update(chunk)
                    if handle is not None:
                        handle.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch: {name}")
        if full_digest.hexdigest() != full[0][0]:
            raise ValueError("Reconstructed checkpoint SHA-256 mismatch")
        if handle is not None:
            handle.close()
            handle = None
            # Both paths are on the same filesystem. link() atomically fails if the
            # destination exists; never replace an existing model, even in a race.
            os.link(temporary, output)
        return full_digest.hexdigest()
    finally:
        if handle is not None:
            handle.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Default: <parts-dir>/best.pt")
    parser.add_argument("--verify-only", action="store_true", help="Validate all parts and full hash without writing")
    args = parser.parse_args()
    output = None if args.verify_only else (args.output or args.parts_dir / "best.pt")
    digest = reassemble(args.parts_dir, output)
    print(f"Verified SHA-256: {digest}")
    if output is not None:
        print(f"Checkpoint: {output.resolve()}")
    print("Load only trusted checkpoints; these models are not for clinical use.")


if __name__ == "__main__":
    main()
