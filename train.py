import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data import load_image, load_instance_mask, random_crop, augment, time_block_split
from losses import seg_loss, distill_loss
from models import build_model, count_params, model_size_mb, predict_prob
from postprocess import prob_to_instances, DEFAULTS
from pq_metric import PQAccumulator


class FilamentDS(Dataset):
    def __init__(self, df, crop=512, steps=None, sampling="annotation", pos_prob=0.7):
        self.df, self.crop = df.reset_index(drop=True), crop
        self.n = len(self.df) if steps is None else steps
        if self.df.empty or self.n <= 0:
            raise ValueError("Training data and samples per epoch must be nonempty.")
        if sampling not in {"annotation", "observation"}:
            raise ValueError("sampling must be annotation or observation")
        if not 0 <= pos_prob <= 1:
            raise ValueError("pos_prob must be between zero and one")
        self.sampling, self.pos_prob = sampling, pos_prob
        self.observations = (list(self.df.groupby("obs_id", sort=False).indices.values())
                             if sampling == "observation" else None)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.observations is None:
            index = i % len(self.df)
        else:

            group = self.observations[np.random.randint(len(self.observations))]
            index = group[np.random.randint(len(group))]
        row = self.df.iloc[index]
        img = load_image(row.image_path)
        mask = (load_instance_mask(row.mask_path) > 0).astype(np.uint8)
        img, mask = random_crop(img, mask, self.crop, pos_prob=self.pos_prob)
        img, mask = augment(img, mask)
        return torch.from_numpy(img)[None], torch.from_numpy(mask.astype(np.float32))[None]


def validate(model, val_df, device, tta=False, params=DEFAULTS,
             tile_size=512, overlap=64, blend="uniform"):

    if val_df.empty:
        raise ValueError("Validation subset is empty; check the manifest and split.")
    acc = PQAccumulator()
    for image_path, annotations in val_df.groupby("image_path", sort=False):
        prob = predict_prob(model, load_image(image_path), device, tta=tta,
                            tile_size=tile_size, overlap=overlap, blend=blend)
        pred = prob_to_instances(prob, **params)
        for mask_path in annotations.mask_path:
            acc.update(load_instance_mask(mask_path), pred)
    return acc.compute(), None, None


def _seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _rng_state(generator):
    state = np.random.get_state()
    return {
        "python": random.getstate(),

        "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def _restore_rng(state, generator):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count for RNG restoration.")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _save_torch(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _save_json(value, path):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1)
    os.replace(temporary, path)


def _check_args(a):
    if a.epochs <= 0 or a.bs <= 0 or a.val_every <= 0 or a.workers < 0:
        raise ValueError("epochs, batch size and val-every must be positive; workers cannot be negative.")
    if a.crop <= 0 or a.crop % 32:
        raise ValueError("crop must be a positive multiple of 32.")
    if a.steps_per_epoch is not None and a.steps_per_epoch < a.bs:
        raise ValueError("samples-per-epoch must be at least the batch size.")
    if not math.isfinite(a.lr) or a.lr <= 0 or not 0 <= a.pos_prob <= 1:
        raise ValueError("lr must be positive and finite; pos-prob must be in [0, 1].")
    if a.val_tile_size <= 0 or not 0 <= a.val_overlap < a.val_tile_size:
        raise ValueError("Require val-tile-size > 0 and 0 <= val-overlap < val-tile-size.")
    if a.resume and a.overwrite:
        raise ValueError("Use --resume or --overwrite, not both.")


def run(a, seed):
    _check_args(a)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / f"seed{seed}"
    resume_path = (out / "last.pt" if a.resume == "auto" else
                   Path(a.resume.format(seed=seed)) if a.resume else None)
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {resume_path}")
    if out.exists() and any(out.iterdir()) and not a.resume and not a.overwrite:
        raise FileExistsError(f"Output directory already contains files: {out}. "
                              "Use a new --out, --resume, or explicitly --overwrite.")
    if (a.resume and out.exists() and any(out.iterdir())
            and resume_path.resolve().parent != out.resolve()):
        raise FileExistsError("Resuming an external checkpoint requires an empty output directory; "
                              "use a new --out to protect the existing run.")
    out.mkdir(parents=True, exist_ok=True)
    if a.overwrite:
        for name in ("best.pt", "last.pt", "final.pt", "summary.json"):
            path = out / name
            if path.is_file():
                path.unlink()

    df = pd.read_csv(a.manifest)
    df = df[df.split == "train"]
    tr_df, va_df = time_block_split(df, seed=a.split_seed)
    if tr_df.empty or va_df.empty:
        raise ValueError("Train or validation subset is empty; check the manifest and split.")
    print(f"[seed {seed}] train rows {len(tr_df)} | val rows {len(va_df)}")
    manifest_hash = hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()


    weights = "imagenet" if a.pretrained and not (a.resume or a.init_checkpoint) else None
    model = build_model(a.model, a.encoder, weights, a.base).to(device)
    if a.init_checkpoint:
        initial = torch.load(a.init_checkpoint.format(seed=seed), map_location="cpu", weights_only=True)
        model.load_state_dict(initial["model"] if "model" in initial else initial, strict=True)
    teacher = None
    if a.teacher:
        teacher = build_model("smp", a.encoder, None).to(device)
        teacher.load_state_dict(torch.load(a.teacher, map_location="cpu", weights_only=True))
        teacher.eval()

    generator = torch.Generator().manual_seed(seed)
    ds = FilamentDS(tr_df, a.crop, steps=a.steps_per_epoch,
                    sampling=a.sampling, pos_prob=a.pos_prob)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=device.type == "cuda",
                    worker_init_fn=_seed_worker, generator=generator)
    if not len(dl):
        raise ValueError("No complete training batch; reduce --bs or increase --samples-per-epoch.")
    print(f"Samples per epoch: {len(ds)}; optimizer steps per epoch: {len(dl)}; sampling: {a.sampling}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min(s - warm, total - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_epoch, log, start_epoch, elapsed = -1.0, None, [], 0, 0.0
    if resume_path is not None:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or saved.get("format_version") != 1:
            raise ValueError("--resume expects last.pt from this trainer; use --init-checkpoint for weights only.")

        keys = ("model", "encoder", "base", "epochs", "bs", "crop", "lr",
                "steps_per_epoch", "cldice", "teacher", "kd", "val_every", "workers", "split_seed",
                "sampling", "pos_prob", "val_tile_size", "val_overlap", "val_blend")
        differences = [key for key in keys if saved["args"].get(key) != getattr(a, key)]
        if saved["seed"] != seed or saved["manifest_sha256"] != manifest_hash or differences:
            raise ValueError(f"Resume configuration/data mismatch (changed options: {differences}). "
                             "Keep the original settings, including total epochs; "
                             "use --init-checkpoint for a new fine-tuning run.")
        if saved["best_val_PQ"] >= 0 and not (out / "best.pt").is_file():
            previous_best = resume_path.parent / "best.pt"
            if not previous_best.is_file():
                raise FileNotFoundError("Resume needs the companion best.pt to preserve the best model.")
            shutil.copy2(previous_best, out / "best.pt")
        model.load_state_dict(saved["model"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        best, best_epoch = saved["best_val_PQ"], saved["best_epoch"]
        log, start_epoch, elapsed = saved["log"], saved["epoch"] + 1, saved["train_time_s"]
        _restore_rng(saved["rng"], generator)
        print(f"Resuming after epoch {start_epoch}; best validation PQ={best:.4f}")

    t0 = time.time()
    def summary():
        return dict(seed=seed, best_val_PQ=best, best_epoch=best_epoch,
                    train_time_s=elapsed + time.time() - t0, params=count_params(model),
                    size_mb=model_size_mb(model), args=vars(a), log=log,
                    manifest_sha256=manifest_hash, samples_per_epoch=len(ds),
                    optimizer_steps_per_epoch=len(dl))

    for ep in range(start_epoch, a.epochs):
        model.train()
        run_loss = 0.0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = seg_loss(logits, y, w_cldice=a.cldice)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = loss + a.kd * distill_loss(logits.float(), t_logits.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()

        msg = dict(epoch=ep, loss=run_loss / len(dl))
        if (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            res, _, _ = validate(model, va_df, device, tile_size=a.val_tile_size,
                                  overlap=a.val_overlap, blend=a.val_blend)
            msg.update({k: float(v) for k, v in res.items()})
            if res["PQ"] > best:
                best, best_epoch = float(res["PQ"]), ep
                _save_torch(model.state_dict(), out / "best.pt")
        print(msg, flush=True)
        log.append(msg)
        report = summary()
        _save_torch(dict(format_version=1, model=model.state_dict(), optimizer=opt.state_dict(),
                         scheduler=sched.state_dict(), scaler=scaler.state_dict(), epoch=ep,
                         seed=seed, best_val_PQ=best, best_epoch=best_epoch, log=log,
                         train_time_s=report["train_time_s"], args=vars(a),
                         manifest_sha256=manifest_hash, rng=_rng_state(generator)), out / "last.pt")
        _save_json(report, out / "summary.json")

    _save_torch(model.state_dict(), out / "final.pt")
    _save_json(summary(), out / "summary.json")
    return best


def make_parser():
    p = argparse.ArgumentParser(description="Train solar filament segmentation with observation-safe validation.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="smp", choices=["smp", "light"])
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--samples-per-epoch", "--steps-per-epoch", "--steps_per_epoch",
                   dest="steps_per_epoch", type=int, default=None,
                   help="Random crop samples per epoch, NOT batches; default one sample per annotation row.")
    p.add_argument("--sampling", choices=["annotation", "observation"], default="annotation")
    p.add_argument("--pos-prob", type=float, default=0.7,
                   help="Probability of anchoring a crop on a foreground pixel.")
    p.add_argument("--cldice", type=float, default=0.0)
    p.add_argument("--teacher", default=None)
    p.add_argument("--kd", type=float, default=1.0)
    p.add_argument("--val-every", "--val_every", dest="val_every", type=int, default=2)
    p.add_argument("--val-tile-size", type=int, default=512)
    p.add_argument("--val-overlap", type=int, default=64)
    p.add_argument("--val-blend", choices=["uniform", "hann"], default="uniform")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--split-seed", type=int, default=0,
                   help="Fixed validation split for ALL training seeds; default 0")
    restarts = p.add_mutually_exclusive_group()
    restarts.add_argument("--init-checkpoint", default=None,
                          help="Warm-start weights with a fresh optimizer/schedule; optional {seed} in path.")
    restarts.add_argument("--resume", nargs="?", const="auto", default=None,
                          help="Resume last.pt with unchanged training settings; omit path to use --out/seedN/last.pt.")
    p.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing run's generated artifacts.")
    return p


if __name__ == "__main__":
    a = make_parser().parse_args()
    scores = [run(a, s) for s in a.seed]
    print(f"val PQ over seeds: mean={np.mean(scores):.4f}  std={np.std(scores):.4f}")
