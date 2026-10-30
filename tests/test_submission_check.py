import hashlib

import numpy as np
import pandas as pd
import pytest

from data import rle_counts, rle_decode, rle_encode
import submission_check


def frame_for(items, shape=(8, 9)):
    rows = []
    for filament_id, coordinates in items:
        mask = np.zeros(shape, dtype=np.uint8)
        for coordinate in coordinates:
            mask[coordinate] = 1
        rows.append((filament_id, rle_encode(mask)))
    return pd.DataFrame(rows, columns=submission_check.EXPECTED_COLUMNS)


def test_disjoint_masks_pass_and_missing_images_are_reported():
    frame = frame_for([("obs_a_1", [(1, 2), (1, 3)]), ("obs_a_2", [(2, 3)])])
    result = submission_check.validate_submission(frame, {"obs_a", "obs_b"}, (8, 9))
    assert result == dict(filament_rows=2, represented_images=1, test_images=2,
                          missing_images=["obs_b"], decoded_masks=2)


def test_same_coordinates_are_allowed_in_different_observations():
    frame = frame_for([("obs_a_1", [(1, 2)]), ("obs_b_1", [(1, 2)])])
    result = submission_check.validate_submission(frame, shape=(8, 9))
    assert result["decoded_masks"] == 2
    assert result["represented_images"] == 2


def test_overlap_beyond_old_five_row_sample_names_both_conflicting_ids():
    items = [(f"obs_a_{i + 1}", [(i, 0)]) for i in range(7)]
    items.append(("obs_a_8", [(0, 0)]))
    frame = frame_for(items)
    with pytest.raises(ValueError, match=r"obs_a_8 overlaps .*obs_a_1.*1 pixels"):
        submission_check.validate_submission(frame, shape=(8, 9))


def test_overlap_reports_every_prior_mask_involved():
    frame = frame_for([("obs_a_1", [(1, 1)]), ("obs_a_2", [(2, 2)]),
                       ("obs_a_3", [(1, 1), (2, 2), (3, 3)])])
    with pytest.raises(ValueError, match=r"obs_a_3 overlaps .*obs_a_1.*obs_a_2.*2 pixels"):
        submission_check.validate_submission(frame, shape=(8, 9))


@pytest.mark.parametrize("encoded", ["!", "P", "O", "0", "oooooooooooooo"])
def test_invalid_compressed_counts_are_rejected_before_decoder(encoded, monkeypatch):
    frame = pd.DataFrame([("obs_a_1", encoded)], columns=submission_check.EXPECTED_COLUMNS)

    def unexpected_decode(*args):
        pytest.fail("Malformed compressed counts must not reach the native decoder")

    monkeypatch.setattr(submission_check, "rle_decode", unexpected_decode)
    with pytest.raises(ValueError, match="Invalid RLE for obs_a_1"):
        submission_check.validate_submission(frame, shape=(8, 9))


def test_valid_rle_for_wrong_size_is_rejected():
    frame = frame_for([("obs_a_1", [(1, 1)])], shape=(7, 9))
    with pytest.raises(ValueError, match="expected 72"):
        submission_check.validate_submission(frame, shape=(8, 9))


def test_zero_area_prediction_is_rejected():
    frame = frame_for([("obs_a_1", [])])
    with pytest.raises(ValueError, match="empty decoded mask for obs_a_1"):
        submission_check.validate_submission(frame, shape=(8, 9))


@pytest.mark.parametrize("decoded", [np.ones((8, 8), np.uint8),
                                     np.full((8, 9), 2, np.uint8),
                                     np.full((8, 9), np.nan)])
def test_decoder_outputs_must_have_expected_shape_and_binary_values(decoded, monkeypatch):
    frame = frame_for([("obs_a_1", [(1, 1)])])
    monkeypatch.setattr(submission_check, "rle_decode", lambda *args: decoded)
    with pytest.raises(ValueError, match="Invalid or empty decoded mask for obs_a_1"):
        submission_check.validate_submission(frame, shape=(8, 9))


@pytest.mark.parametrize("filament_id", ["obs_a", "obs_a_x", "unknown_1"])
def test_malformed_and_unknown_ids_are_rejected(filament_id):
    frame = frame_for([(filament_id, [(1, 1)])])
    with pytest.raises(ValueError, match="unknown or malformed filament IDs"):
        submission_check.validate_submission(frame, {"obs_a"}, (8, 9))


def test_duplicate_ids_are_rejected():
    frame = frame_for([("obs_a_1", [(1, 1)]), ("obs_a_1", [(2, 2)])])
    with pytest.raises(ValueError, match="Duplicate filament_id"):
        submission_check.validate_submission(frame, shape=(8, 9))


@pytest.mark.parametrize("column,value,message", [("filament_id", "", "empty filament IDs"),
                                                 ("segmentation_rle", "", "empty RLE strings"),
                                                 ("segmentation_rle", np.nan, "empty RLE strings")])
def test_empty_fields_are_rejected(column, value, message):
    frame = frame_for([("obs_a_1", [(1, 1)])])
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        submission_check.validate_submission(frame, shape=(8, 9))


def test_columns_and_empty_submissions_are_rejected():
    frame = frame_for([("obs_a_1", [(1, 1)])])
    with pytest.raises(ValueError, match="Unexpected columns"):
        submission_check.validate_submission(frame.iloc[:, ::-1], shape=(8, 9))
    with pytest.raises(ValueError, match="no filament predictions"):
        submission_check.validate_submission(frame.iloc[:0], shape=(8, 9))


def test_check_file_keeps_samples_argument_but_checks_every_mask(tmp_path, capsys):
    frame = frame_for([("obs_a_1", [(1, 1)]), ("obs_a_2", [(2, 2)])], (2048, 2048))
    csv_path, manifest_path = tmp_path / "submission.csv", tmp_path / "manifest.csv"
    frame.to_csv(csv_path, index=False)
    pd.DataFrame([dict(obs_id="obs_a", split="test"), dict(obs_id="obs_b", split="test")]).to_csv(
        manifest_path, index=False)
    result = submission_check.check_file(csv_path, manifest_path, samples=0)
    assert result["decoded_masks"] == 2
    assert result["missing_images"] == ["obs_b"]
    assert result["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    output = capsys.readouterr().out
    assert "checked all 2 masks" in output
    assert f"CSV SHA256: {result['csv_sha256']}" in output


def test_count_parser_matches_real_coco_masks():
    rng = np.random.default_rng(19)
    for shape in [(1, 1), (8, 9), (100, 101), (2048, 2048)]:
        mask = rng.random(shape) < 0.05
        mask[0, 0] = True
        frame = pd.DataFrame([("obs_a_1", rle_encode(mask))], columns=submission_check.EXPECTED_COLUMNS)
        assert submission_check.validate_submission(frame, shape=shape)["decoded_masks"] == 1


def test_rle_counts_returns_background_and_foreground_runs():
    mask = np.zeros((3, 4), dtype=np.uint8, order="F")
    mask.ravel(order="F")[[0, 6, 7]] = 1
    assert rle_counts(rle_encode(mask)) == [0, 1, 5, 2, 4]


def test_rle_decode_rejects_short_background_run_before_native_decoder(monkeypatch):
    from pycocotools import mask as mu

    def unexpected_decode(*args):
        pytest.fail("A short RLE must not reach the native decoder")

    monkeypatch.setattr(mu, "decode", unexpected_decode)
    assert rle_counts("913") == [9, 1, 3]
    with pytest.raises(ValueError, match="RLE runs cover 13 pixels; expected 72"):
        rle_decode("913", (8, 9))
