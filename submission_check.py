

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data import rle_decode


EXPECTED_COLUMNS = ["filament_id", "segmentation_rle"]


def check_file(csv_path, manifest_path, samples=5):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}. Expected {EXPECTED_COLUMNS}")
    if frame.empty:
        raise ValueError("Submission has no filament predictions")
    if frame.filament_id.duplicated().any():
        raise ValueError("Duplicate filament_id detected")
    if (frame.segmentation_rle.str.len() == 0).any():
        raise ValueError("Submission contains empty RLE strings")
    if (frame.filament_id.str.len() == 0).any():
        raise ValueError("Submission contains empty filament IDs")

    manifest = pd.read_csv(manifest_path, dtype={"obs_id": str}, keep_default_na=False)
    test_ids = set(manifest.loc[manifest.split == "test", "obs_id"].astype(str))
    unknown = []
    seen = set()
    for filament_id in frame.filament_id:
        found = re.fullmatch(r"(.+)_(\d+)", filament_id)
        if found is None or found.group(1) not in test_ids:
            unknown.append(filament_id)
        else:
            seen.add(found.group(1))
    if unknown:
        raise ValueError(f"Found unknown or malformed filament IDs, examples: {unknown[:3]}")

    n = min(len(frame), max(0, samples))
    if n:
        sample = frame.sample(n=n, random_state=0)
        for item in sample.itertuples(index=False):
            mask = rle_decode(item.segmentation_rle, (2048, 2048))
            if mask.shape != (2048, 2048) or not np.any(mask):
                raise ValueError(f"Invalid or empty decoded mask for {item.filament_id}")
    print(f"VALID: {len(frame)} filament rows, {len(seen)}/{len(test_ids)} test images represented")
    print(f"Decoded and checked {n} RLE masks successfully")
    print("Images without predictions are allowed to have no rows; Kaggle must still accept the file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--sample", type=int, default=5)
    arguments = parser.parse_args()
    check_file(arguments.csv, arguments.manifest, arguments.sample)
