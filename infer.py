import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_image, load_instance_mask, time_block_split, rle_counts, rle_encode
from models import build_model, predict_prob, count_params, model_size_mb
from postprocess import prob_to_instances, tune, DEFAULTS
from pq_metric import PQAccumulator


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_submission(rows, path):
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    if df["filament_id"].duplicated().any():
        raise ValueError("Duplicate filament_id values in Kaggle submission")
    if df.empty:
        raise ValueError("No filaments predicted for any test image; review model and thresholds")
    for item in df.itertuples(index=False):
        try:
            counts = rle_counts(item.segmentation_rle)
        except ValueError as error:
            raise ValueError(f"Invalid RLE for {item.filament_id}: {error}") from error
        if sum(counts) != 2048 * 2048 or not sum(counts[1::2]):
            raise ValueError(f"Invalid RLE for {item.filament_id}: "
                             "expected a nonempty mask covering exactly 2048 x 2048 pixels")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        restored = pd.read_csv(staged, dtype=str, keep_default_na=False)
        if not restored.equals(df):
            raise ValueError("CSV readback differs from the generated predictions")
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    print(f"Wrote {len(df)} filament predictions to {path}", flush=True)


def validation_partition(val_df, tune_samples, seed=0, validation_samples=0):

    obs = val_df.obs_id.drop_duplicates().to_numpy()
    if len(obs) < 2:
        raise ValueError("Need at least two validation observations for calibration and audit")
    if tune_samples < 1 or validation_samples < 0:
        raise ValueError("tune_samples must be positive; validation_samples must be nonnegative")
    obs = np.random.default_rng(seed).permutation(np.sort(obs))
    n = min(tune_samples, len(obs) - 1)
    fit_ids, audit_ids = obs[:n], obs[n:]
    if validation_samples:
        audit_ids = audit_ids[:validation_samples]
    return (val_df[val_df.obs_id.isin(fit_ids)].reset_index(drop=True),
            val_df[val_df.obs_id.isin(audit_ids)].reset_index(drop=True))


class MaskSequence:

    def __init__(self, paths, probabilities=False):
        self.paths, self.probabilities = list(paths), probabilities

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return np.load(path, mmap_mode="r") if self.probabilities else load_instance_mask(path)


def predict_ensemble(models, img, device, *, tta=False, tile_size=512,
                     overlap=64, blend="uniform"):

    if not models:
        raise ValueError("Need at least one model for probability averaging")
    result = np.zeros(img.shape, dtype=np.float32)
    for model in models:
        result += predict_prob(model, img, device, tta=tta, tile_size=tile_size,
                               overlap=overlap, blend=blend) / len(models)
    return result


def cache_predictions(models, rows, device, args, signature):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    observations = rows.drop_duplicates("obs_id")
    for i, row in enumerate(observations.itertuples(), 1):
        content = dict(inference=signature, image_sha256=sha256(row.image_path))
        key = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:24]
        path = cache_dir / f"{row.obs_id}_{key}.npy"
        if not path.is_file():
            prob = predict_ensemble(models, load_image(row.image_path), device,
                                    tta=bool(args.tta), tile_size=args.tile_size,
                                    overlap=args.overlap, blend=args.blend)
            tmp = path.with_suffix(".npy.tmp")
            with tmp.open("wb") as handle:
                np.save(handle, prob)
            os.replace(tmp, path)
        paths[row.obs_id] = str(path.resolve())
        if i % 10 == 0 or i == len(observations):
            print(f"Validation probability maps: {i}/{len(observations)} (cached maps reused)", flush=True)
    return paths


def evaluate_rows(rows, paths, params):
    accumulator, records = PQAccumulator(), []
    for obs_id, annotations in rows.groupby("obs_id", sort=False):
        prob = np.load(paths[obs_id], mmap_mode="r")
        instances = prob_to_instances(prob, **params)
        for row in annotations.itertuples():
            gt = load_instance_mask(row.mask_path)
            accumulator.update(gt, instances)
            single = PQAccumulator()
            single.update(gt, instances)
            records.append(dict(obs_id=obs_id, image_id=row.image_id,
                                image_path=row.image_path, mask_path=row.mask_path,
                                prob_path=paths[obs_id], **single.compute()))
    return accumulator.compute(), records


def main(a):
    if a.evaluate_only and a.skip_validation:
        raise ValueError("--evaluate-only needs validation; remove --skip-validation")
    if a.tune_samples < 1 or a.validation_samples < 0:
        raise ValueError("Invalid calibration/audit sample counts")
    if a.use_default_postprocessing and not (a.params and a.skip_tune and a.skip_validation):
        raise ValueError("--use-default-postprocessing is an export option: also supply "
                         "--params --skip-tune --skip-validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = [a.ckpt] if isinstance(a.ckpt, str) else a.ckpt
    if len(set(sha256(p) for p in checkpoints)) != len(checkpoints):
        raise ValueError("Repeated checkpoint weights would bias the equal-weight ensemble")
    models = []
    for checkpoint in checkpoints:
        model = build_model(a.model, a.encoder, None, a.base).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        models.append(model)
        print(f"{checkpoint}: params={count_params(model):,}; size={model_size_mb(model):.2f} MB", flush=True)
    from importlib.metadata import version
    signature = dict(checkpoint_sha256=[sha256(p) for p in checkpoints],
                     model=a.model, encoder=a.encoder, base=a.base, tta=bool(a.tta),
                     tile_size=a.tile_size, overlap=a.overlap, blend=a.blend,
                     torch_version=str(torch.__version__),
                     smp_version=version("segmentation-models-pytorch") if a.model == "smp" else None)
    df = pd.read_csv(a.manifest, keep_default_na=False)
    params, calibration = dict(DEFAULTS), {}
    if a.params:
        calibration = json.loads(Path(a.params).read_text(encoding="utf-8"))
        if calibration.get("inference_signature") != signature:
            raise ValueError("Calibration belongs to different weights/TTA/tiles/blending/library versions. "
                             "Use matching settings or recalibrate.")
        if calibration.get("split_seed") != a.split_seed:
            raise ValueError("Calibration split seed differs from --split-seed")
        params.update(calibration["postprocessing"])
        if not a.skip_tune:
            raise ValueError("Use --skip-tune with --params to reuse a calibration")

    if not a.skip_tune or not a.skip_validation:
        _, val = time_block_split(df[df.split == "train"], seed=a.split_seed)
        if val.empty:
            raise ValueError("Validation subset is empty")
        if not a.skip_tune:
            fit, audit = validation_partition(val, a.tune_samples, a.split_seed, a.validation_samples)
        else:
            fit, audit = val.iloc[:0], val

            if calibration.get("tuning", {}).get("obs_ids"):
                audit = audit[~audit.obs_id.isin(calibration["tuning"]["obs_ids"])]
            if a.validation_samples:
                selected = audit.obs_id.drop_duplicates().head(a.validation_samples)
                audit = audit[audit.obs_id.isin(selected)]
        required = pd.concat([fit, audit if not a.skip_validation else audit.iloc[:0]])
        paths = cache_predictions(models, required, device, a, signature)
        if not a.skip_tune:
            probabilities = MaskSequence([paths[o] for o in fit.obs_id], probabilities=True)
            gts = MaskSequence(fit.mask_path)
            params, fit_pq = tune(probabilities, gts, verbose=True)
            print(f"Calibration fit: {fit.obs_id.nunique()} observations, PQ={fit_pq:.4f}", flush=True)
            calibration = dict(tuning=dict(PQ=fit_pq, observations=int(fit.obs_id.nunique()),
                                           annotation_sets=len(fit), obs_ids=fit.obs_id.drop_duplicates().tolist()),
                               search="two-start coordinate search; not exhaustive")
        if not a.skip_validation:
            if audit.empty:
                raise ValueError("No observations remain for post-processing audit")
            baseline, _ = evaluate_rows(audit, paths, DEFAULTS)
            metrics, records = evaluate_rows(audit, paths, params)
            calibration["validation"] = dict(metrics=metrics, default_postprocessing_metrics=baseline,
                                             observations=int(audit.obs_id.nunique()), annotation_sets=len(audit),
                                             obs_ids=audit.obs_id.drop_duplicates().tolist(),
                                             note="Held out from post-processing tuning. Model checkpoints may "
                                                  "already have been selected on these observations; this is "
                                                  "not an untouched model test set or a verified official score.")


            pooled = PQAccumulator()
            fit_metrics = evaluate_rows(fit, paths, params)[0] if not fit.empty else None
            for result in [metrics, fit_metrics]:
                if result is not None:
                    pooled.tp += result["TP"]
                    pooled.fp += result["FP"]
                    pooled.fn += result["FN"]
                    pooled.iou += result["SQ"] * result["TP"]
                    pooled.split += result["split"]
                    pooled.merge += result["merge"]
            all_rows = pd.concat([fit, audit])
            complete = all_rows.obs_id.nunique() == val.obs_id.nunique()
            calibration["full_validation" if complete else "sampled_validation"] = dict(
                metrics=pooled.compute(), observations=int(all_rows.obs_id.nunique()),
                annotation_sets=len(all_rows),
                note="Includes observations used for tuning; use the separate post-processing audit for selection.")
            csv_path = Path(a.params_out or Path(a.out).with_suffix(".calibration.json")).with_suffix(".per_image.csv")
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(csv_path, index=False)
            print("Audit with default post-processing:", baseline, flush=True)
            print("Audit with selected post-processing:", metrics, flush=True)
            if metrics["PQ"] < baseline["PQ"]:
                print("Tuned post-processing reduced audit PQ. Compare the default recipe before "
                      "exporting; --use-default-postprocessing can reuse these weights/settings "
                      "with default instance conversion.", flush=True)
        calibration.update(postprocessing=params, inference_signature=signature,
                           checkpoints=checkpoints, split_seed=a.split_seed)
        params_path = a.params_out or Path(a.out).with_suffix(".calibration.json")
        write_json(params_path, calibration)
        print("Saved calibration:", params_path, flush=True)
    if a.evaluate_only:
        return calibration

    export_validation = calibration.get("validation")
    if a.use_default_postprocessing:
        params = dict(DEFAULTS)
        if export_validation:
            export_validation = {**export_validation,
                                 "metrics": export_validation["default_postprocessing_metrics"]}

    test = df[df.split == "test"]
    if test.empty or test.obs_id.duplicated().any():
        raise ValueError("Need a nonempty test set with unique observation IDs")
    rows, missing = [], 0
    start = time.perf_counter()
    for index, row in enumerate(test.itertuples(), 1):
        prob = predict_ensemble(models, load_image(row.image_path), device,
                                tta=bool(a.tta), tile_size=a.tile_size,
                                overlap=a.overlap, blend=a.blend)
        inst = prob_to_instances(prob, **params)
        if inst.shape != (2048, 2048):
            raise ValueError(f"Expected 2048 x 2048 prediction for {row.obs_id}, got {inst.shape}")
        ids = np.unique(inst)
        ids = ids[ids > 0]
        missing += int(len(ids) == 0)
        for number, instance_id in enumerate(ids, 1):
            rows.append((f"{row.obs_id}_{number}", rle_encode(inst == instance_id)))
        if index % 30 == 0 or index == len(test):
            print(f"Predicted {index}/{len(test)} test images", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    write_submission(rows, a.out)
    details = dict(checkpoints=checkpoints, inference_signature=signature, csv_sha256=sha256(a.out),
                   model=a.model, encoder=a.encoder, output=a.out, postprocessing=params,
                   test_images=len(test), predicted_filaments=len(rows),
                   images_without_predictions=missing, seconds_per_image=elapsed / len(test),
                   tta=bool(a.tta), split_seed=a.split_seed, calibration_file=a.params,
                   used_tuned_postprocessing=bool((a.params or not a.skip_tune)
                                                 and not a.use_default_postprocessing),
                   validation=export_validation)
    write_json(Path(a.out).with_suffix(".metadata.json"), details)
    print(f"End-to-end seconds/test image: {elapsed / len(test):.3f}; empty images: {missing}", flush=True)
    return details


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt", required=True, nargs="+", help="One checkpoint, or several compatible models for a probability ensemble")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--model", default="smp", choices=["smp", "light"])
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--seed", type=int, default=0, help="Legacy option; use --split-seed for the validation split")
    p.add_argument("--tta", type=int, choices=[0, 1], default=0)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--blend", choices=["uniform", "hann"], default="uniform")
    p.add_argument("--skip-tune", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--evaluate-only", action="store_true", help="Calibrate/evaluate without predicting test images")
    p.add_argument("--tune-samples", type=int, default=24, help="Unique validation observations used to tune post-processing")
    p.add_argument("--validation-samples", type=int, default=0, help="Limit audit observations for a smoke check; 0 audits all remaining observations")
    p.add_argument("--cache-dir", default="runs/prob_cache", help="Disk cache of validation probability maps (about 16 MB per observation/configuration)")
    p.add_argument("--params", help="Reuse calibration JSON, requiring matching inference settings")
    p.add_argument("--params-out", help="Where to save calibration and diagnostics")
    p.add_argument("--use-default-postprocessing", action="store_true",
                   help="Export with DEFAULTS when its audit beats tuning; requires --params --skip-tune --skip-validation")
    p.add_argument("--split-seed", type=int, default=0, help="Must match training")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
