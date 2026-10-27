


import argparse, time
import numpy as np
import pandas as pd
import torch

from data import load_image, load_instance_mask, time_block_split, rle_encode
from models import build_model, predict_prob, count_params, model_size_mb
from postprocess import prob_to_instances, tune, DEFAULTS
from pq_metric import PQAccumulator


def write_submission(rows, path):
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    if df["filament_id"].duplicated().any():
        raise ValueError("Duplicate filament_id values in Kaggle submission")
    if df.empty:
        raise ValueError("No filaments predicted for any test image; review model and thresholds")
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} filament predictions to {path}")


def main(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.model, a.encoder, None, a.base).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"params={count_params(model):,}; size={model_size_mb(model):.2f} MB")

    df = pd.read_csv(a.manifest, keep_default_na=False)
    params = dict(DEFAULTS)
    if not a.skip_tune or not a.skip_validation:
        _, va = time_block_split(df[df.split == "train"], seed=a.split_seed)
        if va.empty:
            raise ValueError("Validation subset is empty")

        if not a.skip_tune:
            subset = va.sample(n=min(len(va), a.tune_samples), random_state=a.split_seed)
            probs = [predict_prob(model, load_image(r.image_path), device, tta=bool(a.tta))
                     for _, r in subset.iterrows()]
            gts = [load_instance_mask(r.mask_path) for _, r in subset.iterrows()]
            tuned, subset_pq = tune(probs, gts, verbose=False)
            params.update(tuned)
            del probs, gts
            print(f"Post-processing search on {len(subset)} validation entries: PQ={subset_pq:.4f}")
            print("Selected parameters:", params)

        if not a.skip_validation:
            acc = PQAccumulator()
            for _, r in va.iterrows():
                prob = predict_prob(model, load_image(r.image_path), device, tta=bool(a.tta))
                inst = prob_to_instances(prob, **params)
                acc.update(load_instance_mask(r.mask_path), inst)
            print("Full validation:", acc.compute())

    test = df[df.split == "test"]
    if test.empty:
        raise ValueError("Manifest contains no test images")

    rows = []
    start = time.perf_counter()
    missing = 0
    for index, (_, r) in enumerate(test.iterrows(), start=1):
        prob = predict_prob(model, load_image(r.image_path), device, tta=bool(a.tta))
        inst = prob_to_instances(prob, **params)
        if inst.shape != (2048, 2048):
            raise ValueError(f"Expected 2048 x 2048 prediction for {r.obs_id}, got {inst.shape}")
        instance_ids = np.unique(inst)
        instance_ids = instance_ids[instance_ids > 0]
        if len(instance_ids) == 0:
            missing += 1
        for filament_num, instance_id in enumerate(instance_ids, start=1):
            rows.append((f"{r.obs_id}_{filament_num}", rle_encode(inst == instance_id)))
        if index % 30 == 0 or index == len(test):
            print(f"Predicted {index}/{len(test)} test images")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"End-to-end time per test image: {elapsed / len(test):.3f} s")
    print(f"Images without predicted filaments: {missing}")
    write_submission(rows, a.out)

    from pathlib import Path
    import json
    details = dict(checkpoint=a.ckpt, model=a.model, encoder=a.encoder,
                   output=a.out, postprocessing=params, test_images=int(len(test)),
                   predicted_filaments=len(rows), images_without_predictions=missing,
                   seconds_per_image=elapsed / len(test),
                   tta=bool(a.tta), split_seed=a.split_seed)
    Path(a.out).with_suffix(".metadata.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--model", default="smp"); p.add_argument("--encoder", default="resnet34")
    p.add_argument("--base", type=int, default=16); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tta", type=int, choices=[0, 1], default=0)
    p.add_argument("--skip-tune", action="store_true", help="Use default post-processing")
    p.add_argument("--skip-validation", action="store_true", help="Skip full validation to produce a first CSV quickly")
    p.add_argument("--tune-samples", type=int, default=4,
                   help="Limit RAM/time for hyperparameter search; increase if resources permit")
    p.add_argument("--split-seed", type=int, default=0,
                   help="Must match train.py --split-seed")
    main(p.parse_args())
