

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data import rle_counts, rle_decode


EXPECTED_COLUMNS = ["filament_id", "segmentation_rle"]


def _validate_counts(encoded, pixels):
    counts = rle_counts(encoded)
    if sum(counts) != pixels:
        raise ValueError(f"RLE runs cover {sum(counts)} pixels; expected {pixels}")
    return counts


def validate_submission(frame, test_ids=None, shape=(2048, 2048)):
    if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in shape):
        raise ValueError("Mask shape must contain two positive integer dimensions")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}. Expected {EXPECTED_COLUMNS}")
    if frame.empty:
        raise ValueError("Submission has no filament predictions")
    if frame.filament_id.duplicated().any():
        raise ValueError("Duplicate filament_id detected")
    if not frame.segmentation_rle.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Submission contains empty RLE strings")
    if not frame.filament_id.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Submission contains empty filament IDs")

    test_ids = None if test_ids is None else set(map(str, test_ids))
    unknown = []
    groups = {}
    for filament_id, encoded in frame.itertuples(index=False, name=None):
        found = re.fullmatch(r"(.+)_(\d+)", filament_id)
        if found is None or (test_ids is not None and found.group(1) not in test_ids):
            unknown.append(filament_id)
        else:
            groups.setdefault(found.group(1), []).append((filament_id, encoded))
    if unknown:
        raise ValueError(f"Found unknown or malformed filament IDs, examples: {unknown[:3]}")

    pixels = int(shape[0]) * int(shape[1])
    for obs_id, items in groups.items():
        owners = np.zeros(shape, dtype=np.int32)
        for number, (filament_id, encoded) in enumerate(items, 1):
            try:
                _validate_counts(encoded, pixels)
                mask = np.asarray(rle_decode(encoded, shape))
            except Exception as error:
                raise ValueError(f"Invalid RLE for {filament_id}: {error}") from error
            if mask.shape != tuple(shape) or not np.isin(mask, [0, 1]).all() or not mask.any():
                raise ValueError(f"Invalid or empty decoded mask for {filament_id}")
            foreground = mask.astype(bool)
            overlaps = foreground & (owners > 0)
            if overlaps.any():
                conflicting = [items[i - 1][0] for i in np.unique(owners[overlaps])]
                raise ValueError(f"Overlapping masks in {obs_id}: {filament_id} overlaps "
                                 f"{conflicting} at {np.count_nonzero(overlaps)} pixels")
            owners[foreground] = number
    missing = [] if test_ids is None else sorted(test_ids - groups.keys())
    return dict(filament_rows=len(frame), represented_images=len(groups),
                test_images=None if test_ids is None else len(test_ids),
                missing_images=missing, decoded_masks=len(frame))


def check_file(csv_path, manifest_path, samples=5):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    csv_sha256 = digest.hexdigest()
    print(f"CSV SHA256: {csv_sha256}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    manifest = pd.read_csv(manifest_path, dtype={"obs_id": str}, keep_default_na=False)
    test_ids = set(manifest.loc[manifest.split == "test", "obs_id"].astype(str))
    result = validate_submission(frame, test_ids)
    result["csv_sha256"] = csv_sha256
    print(f"LOCAL FORMAT CHECK PASSED: {len(frame)} filament rows, "
          f"{result['represented_images']}/{len(test_ids)} test images represented")
    print(f"Decoded every RLE mask and checked all {len(frame)} masks for overlaps")
    if result["missing_images"]:
        print(f"Test images without predictions: {result['missing_images']}")
    print("Competition acceptance is checked separately by Kaggle; this check does not guarantee acceptance.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--sample", type=int, default=5)
    arguments = parser.parse_args()
    check_file(arguments.csv, arguments.manifest, arguments.sample)
