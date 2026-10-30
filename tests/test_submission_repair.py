import json

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

from data import rle_decode, rle_encode
from repair_submission import repair_file


def encoded_counts(counts, shape=(8, 9)):
    return mask_util.frPyObjects(dict(size=list(shape), counts=counts), *shape)["counts"].decode("ascii")


def test_repairs_only_missing_final_background_and_preserves_source_and_other_rows(tmp_path):
    shape = (8, 9)
    second = np.zeros(shape, np.uint8)
    second[4:6, 6:8] = 1
    unaffected = rle_encode(second)
    source, output = tmp_path / "original.csv", tmp_path / "fixed.csv"
    pd.DataFrame([("obs_1", encoded_counts([10, 3, 7])), ("obs_2", unaffected)],
                 columns=["filament_id", "segmentation_rle"]).to_csv(source, index=False)
    original = source.read_bytes()
    result = repair_file(source, output, shape=shape)
    flat = np.zeros(72, np.uint8)
    flat[10:13] = 1
    expected = flat.reshape(shape, order="F")
    fixed = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert source.read_bytes() == original
    assert fixed.filament_id.tolist() == ["obs_1", "obs_2"]
    assert fixed.segmentation_rle.iloc[0] == rle_encode(expected)
    assert fixed.segmentation_rle.iloc[1] == unaffected
    np.testing.assert_array_equal(rle_decode(fixed.segmentation_rle.iloc[0], shape), expected)
    assert result["repaired_rows"] == [dict(filament_id="obs_1", encoded_pixels=20,
                                            restored_background_pixels=52, foreground_pixels=3)]
    assert json.loads(output.with_suffix(".audit.json").read_text()) == result


@pytest.mark.parametrize("counts", [[10, 3], [80, 3, 2]])
def test_refuses_incomplete_foreground_or_runs_beyond_the_image(tmp_path, counts):
    source, output = tmp_path / "original.csv", tmp_path / "fixed.csv"
    pd.DataFrame([("obs_1", encoded_counts(counts))],
                 columns=["filament_id", "segmentation_rle"]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Cannot safely repair obs_1"):
        repair_file(source, output, shape=(8, 9))
    assert not output.exists()


def test_refuses_to_publish_if_existing_masks_overlap(tmp_path):
    source, output = tmp_path / "original.csv", tmp_path / "fixed.csv"
    encoded = encoded_counts([10, 3, 59])
    pd.DataFrame([("obs_1", encoded), ("obs_2", encoded)],
                 columns=["filament_id", "segmentation_rle"]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Overlapping masks"):
        repair_file(source, output, shape=(8, 9))
    assert not output.exists()


def test_preserves_existing_files(tmp_path):
    source = tmp_path / "original.csv"
    source.write_text("original data", encoding="utf-8")
    with pytest.raises(ValueError, match="separate output path"):
        repair_file(source, source)
    output = tmp_path / "fixed.csv"
    output.write_text("existing data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        repair_file(source, output)
    assert source.read_text() == "original data"
    assert output.read_text() == "existing data"
