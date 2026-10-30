import numpy as np
import pandas as pd
import pytest

import infer
from data import rle_decode, rle_encode
from submission_check import validate_submission


@pytest.fixture
def predicted_rows():
    first = np.zeros((2048, 2048), dtype=np.uint8)
    first[2:9, 3:16] = 1
    second = np.zeros_like(first)
    second[40:56, 70:89] = 1
    return [("obs_1", rle_encode(first)), ("obs_2", rle_encode(second))]


def test_publishes_complete_model_predictions_without_changing_masks(tmp_path, predicted_rows):
    path = tmp_path / "submission.csv"
    infer.write_submission(predicted_rows, path)
    restored = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(restored.itertuples(index=False, name=None)) == predicted_rows
    assert validate_submission(restored)["decoded_masks"] == 2
    for (_, encoded), row in zip(predicted_rows, restored.itertuples(index=False)):
        np.testing.assert_array_equal(rle_decode(row.segmentation_rle, (2048, 2048)),
                                      rle_decode(encoded, (2048, 2048)))
    assert path.read_bytes().endswith(b"\n")
    assert not list(tmp_path.glob(".submission.csv.*.tmp"))


@pytest.mark.parametrize("encoded", ["913", "!", rle_encode(np.zeros((2048, 2048), np.uint8))])
def test_rejects_malformed_or_empty_predictions_without_replacing_output(tmp_path, encoded):
    path = tmp_path / "submission.csv"
    path.write_bytes(b"previous completed submission")
    with pytest.raises(ValueError, match="Invalid RLE for obs_1"):
        infer.write_submission([("obs_1", encoded)], path)
    assert path.read_bytes() == b"previous completed submission"
    assert not list(tmp_path.glob(".submission.csv.*.tmp"))


@pytest.mark.parametrize("failure", ["raise", "truncate"])
def test_interrupted_or_truncated_write_preserves_previous_output(tmp_path, predicted_rows, monkeypatch, failure):
    path = tmp_path / "submission.csv"
    path.write_bytes(b"previous completed submission")
    real_write = pd.DataFrame.to_csv

    def incomplete_write(frame, handle, **kwargs):
        real_write(frame.iloc[:1], handle, **kwargs)
        if failure == "raise":
            raise OSError("Interrupted export")

    monkeypatch.setattr(pd.DataFrame, "to_csv", incomplete_write)
    error = OSError if failure == "raise" else ValueError
    with pytest.raises(error):
        infer.write_submission(predicted_rows, path)
    assert path.read_bytes() == b"previous completed submission"
    assert not list(tmp_path.glob(".submission.csv.*.tmp"))
