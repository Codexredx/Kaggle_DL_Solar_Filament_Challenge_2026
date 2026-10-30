import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import rle_counts, rle_encode
from submission_check import validate_submission


def repair_file(csv_path, output_path, manifest_path=None, shape=(2048, 2048)):
    source, output = Path(csv_path), Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("Choose a separate output path to preserve the original CSV")
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    if list(frame.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError("Expected filament_id and segmentation_rle columns")
    pixels = int(shape[0]) * int(shape[1])
    repaired = []
    for index, row in frame.iterrows():
        counts = rle_counts(row.segmentation_rle)
        total = sum(counts)
        if total == pixels:
            continue
        if total > pixels or len(counts) % 2 == 0:
            raise ValueError(f"Cannot safely repair {row.filament_id}: "
                             "only a short final background run can be extended")
        flat = np.zeros(pixels, dtype=np.uint8)
        position = 0
        for number, count in enumerate(counts):
            if number % 2:
                flat[position:position + count] = 1
            position += count
        encoded = rle_encode(flat.reshape(shape, order="F"))
        fixed_counts = rle_counts(encoded)
        if counts[:-1] != fixed_counts[:-1] or fixed_counts[-1] != counts[-1] + pixels - total:
            raise ValueError(f"Cannot repair {row.filament_id} without changing other runs")
        frame.at[index, "segmentation_rle"] = encoded
        repaired.append(dict(filament_id=row.filament_id, encoded_pixels=total,
                             restored_background_pixels=pixels - total,
                             foreground_pixels=sum(counts[1::2])))
    test_ids = None
    if manifest_path is not None:
        manifest = pd.read_csv(manifest_path, dtype={"obs_id": str}, keep_default_na=False)
        test_ids = set(manifest.loc[manifest.split == "test", "obs_id"])
    result = validate_submission(frame, test_ids, shape)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, lineterminator="\n")
    report = dict(source=str(source.resolve()), output=str(output.resolve()),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  repaired_rows=repaired, validation=result)
    output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    repair_file(args.csv, args.out, args.manifest)
