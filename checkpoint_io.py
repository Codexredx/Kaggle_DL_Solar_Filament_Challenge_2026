import hashlib
import os
from pathlib import Path
import tempfile
import zipfile

import torch


def is_unpacked_checkpoint(path):
    path = Path(path)
    return (path.is_dir() and (path / "data.pkl").is_file()
            and (path / "version").is_file() and (path / "data").is_dir())


def find_checkpoints(roots):
    found = set()
    for root in map(Path, roots):
        candidates = root.rglob("best.pt") if root.is_dir() else [root]
        for candidate in candidates:
            if candidate.name == "best.pt" and (candidate.is_file() or is_unpacked_checkpoint(candidate)):
                found.add(candidate.resolve())
    return sorted(found)


def _content_hash(members):
    digest = hashlib.sha256()
    for name, path in members:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as handle:
            for part in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(part)
    return digest.hexdigest()


def _validate(path):
    torch.load(path, map_location="cpu", weights_only=True)


def prepare_checkpoint(path, cache_dir):
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".zip":
        raise ValueError("Extract only the downloaded outer ZIP and select best.pt inside it. "
                         "Keep the best.pt checkpoint itself intact.")
    if source.is_file():
        return source
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint does not exist in this session: {source}")
    if not is_unpacked_checkpoint(source):
        raise ValueError(f"Checkpoint path is a folder, not an intact best.pt or recognized unpacked "
                         f"PyTorch checkpoint: {source}. Expected data.pkl, version, and data/.")

    entries = sorted(source.rglob("*"))
    if any(entry.is_symlink() for entry in entries):
        raise ValueError(f"Unpacked checkpoint contains symbolic links: {source}")
    members = [(entry.relative_to(source).as_posix(), entry) for entry in entries if entry.is_file()]
    digest = _content_hash(members)
    destination = Path(cache_dir).expanduser().resolve() / digest / "best.pt"
    if destination.is_file():
        try:
            _validate(destination)
            return destination
        except Exception:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, member in members:
                archive.write(member, f"archive/{name}")
        try:
            _validate(temporary)
        except Exception as error:
            raise ValueError(f"Could not reconstruct valid PyTorch weights from {source}. "
                             "Check that all files from the original checkpoint were preserved.") from error
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
