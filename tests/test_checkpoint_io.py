from collections import OrderedDict
from pathlib import Path
import zipfile

import pytest
import torch

from checkpoint_io import find_checkpoints, is_unpacked_checkpoint, prepare_checkpoint


def checkpoint_state(value=0.0):
    return OrderedDict([
        ("weight", torch.arange(12, dtype=torch.float32).reshape(3, 4) + value),
        ("running_mean", torch.tensor([0.2, -0.5, 1.25], dtype=torch.float64)),
        ("num_batches_tracked", torch.tensor(23, dtype=torch.int64)),
        ("mask", torch.tensor([True, False, True])),
    ])


def unpack_checkpoint(root, state=None):
    root.mkdir(parents=True, exist_ok=True)
    saved = root / "original.pt"
    torch.save(checkpoint_state() if state is None else state, saved)
    source = root / "best.pt"
    source.mkdir()
    with zipfile.ZipFile(saved) as archive:
        for member in archive.infolist():
            relative = Path(*Path(member.filename).parts[1:])
            destination = source / relative
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    saved.unlink()
    return source


def file_snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def assert_same_state(actual, expected):
    assert list(actual) == list(expected)
    for key, value in expected.items():
        assert actual[key].dtype == value.dtype
        assert actual[key].shape == value.shape
        assert torch.equal(actual[key], value)


def test_reconstructs_unpacked_pytorch_weights_without_changing_input(tmp_path):
    expected = checkpoint_state()
    source = unpack_checkpoint(tmp_path / "uploaded", expected)
    before = file_snapshot(source)
    assert source.is_dir() and not source.is_file()
    assert is_unpacked_checkpoint(source)
    output = prepare_checkpoint(source, tmp_path / "cache")
    assert output.is_file() and output.name == "best.pt"
    assert output.is_relative_to(tmp_path / "cache")
    assert_same_state(torch.load(output, map_location="cpu", weights_only=True), expected)
    assert file_snapshot(source) == before
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {f"archive/{name}" for name in before}
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())


def test_recovery_cache_is_reused_and_changes_when_tensor_contents_change(tmp_path):
    source = unpack_checkpoint(tmp_path / "uploaded")
    output = prepare_checkpoint(source, tmp_path / "cache")
    stamp = output.stat().st_mtime_ns
    assert prepare_checkpoint(source, tmp_path / "cache") == output
    assert output.stat().st_mtime_ns == stamp
    data_file = next(path for path in (source / "data").iterdir() if path.is_file())
    contents = data_file.read_bytes()
    replacement = bytes([contents[0] ^ 1]) + contents[1:]
    data_file.write_bytes(replacement)
    changed = prepare_checkpoint(source, tmp_path / "cache")
    assert changed != output
    assert changed.is_file() and output.is_file()


def test_corrupted_cache_is_rebuilt_atomically_from_original_members(tmp_path):
    source = unpack_checkpoint(tmp_path / "uploaded")
    output = prepare_checkpoint(source, tmp_path / "cache")
    output.write_bytes(b"damaged cached archive")
    assert prepare_checkpoint(source, tmp_path / "cache") == output
    assert_same_state(torch.load(output, weights_only=True), checkpoint_state())
    assert not list(output.parent.glob("*.tmp"))


def test_intact_pytorch_zip_checkpoint_is_returned_unchanged(tmp_path):
    checkpoint = tmp_path / "best.pt"
    torch.save(checkpoint_state(), checkpoint)
    before = checkpoint.read_bytes()
    assert zipfile.is_zipfile(checkpoint)
    assert prepare_checkpoint(checkpoint, tmp_path / "cache") == checkpoint
    assert checkpoint.read_bytes() == before
    assert not (tmp_path / "cache").exists()


def test_discovery_includes_intact_files_and_unpacked_checkpoint_directories(tmp_path):
    source = unpack_checkpoint(tmp_path / "uploaded")
    intact = tmp_path / "different_model" / "best.pt"
    intact.parent.mkdir()
    torch.save(checkpoint_state(4), intact)
    ordinary = tmp_path / "ordinary" / "best.pt"
    ordinary.mkdir(parents=True)
    (ordinary / "readme.txt").write_text("not model weights")
    found = find_checkpoints([tmp_path, tmp_path / "uploaded", tmp_path / "missing", intact])
    assert found == sorted([source, intact])
    assert not is_unpacked_checkpoint(ordinary)


def test_missing_path_reports_that_weights_are_unavailable(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        prepare_checkpoint(tmp_path / "missing.pt", tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_ordinary_directory_is_not_replaced_with_fake_weights(tmp_path):
    folder = tmp_path / "best.pt"
    folder.mkdir()
    with pytest.raises(ValueError, match="Expected data.pkl"):
        prepare_checkpoint(folder, tmp_path / "cache")
    assert folder.is_dir() and not list(folder.iterdir())
    assert not (tmp_path / "cache").exists()


def test_outer_download_zip_requires_selecting_the_actual_checkpoint(tmp_path):
    archive = tmp_path / "best.pt.zip"
    archive.write_bytes(b"an outer download archive")
    with pytest.raises(ValueError, match="outer ZIP"):
        prepare_checkpoint(archive, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_incomplete_unpacked_checkpoint_is_not_published(tmp_path):
    source = tmp_path / "best.pt"
    source.mkdir()
    (source / "data").mkdir()
    (source / "data.pkl").write_bytes(b"corrupt pickle")
    (source / "version").write_text("3\n")
    assert is_unpacked_checkpoint(source)
    with pytest.raises(ValueError, match="all files"):
        prepare_checkpoint(source, tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("best.pt"))
    assert not list((tmp_path / "cache").rglob("*.tmp"))
