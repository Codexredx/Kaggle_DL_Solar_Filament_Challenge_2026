


import argparse
import math
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import load_image, load_instance_mask, time_block_split
from losses import seg_loss
from models import build_model, predict_prob
from postprocess import prob_to_instances, DEFAULTS
from pq_metric import PQAccumulator
from train import FilamentDS


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main(args):
    df = pd.read_csv(args.manifest)
    df = df[df.split == "train"]
    tr, va = time_block_split(df, seed=args.split_seed)
    if tr.empty or va.empty:
        raise ValueError("Train or validation subset empty; check manifest and split.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={dev}; train annotation sets={len(tr)}, validation sets={len(va)}")
    if dev.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    net = build_model(args.model, args.encoder,
                      "imagenet" if args.pretrained else None, args.base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    ds = FilamentDS(tr, crop=args.crop, steps=(args.batches + 1) * args.bs)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0,
                    drop_last=True)
    batch_seconds = []
    net.train()
    for i, (x, y) in enumerate(dl):
        if i >= args.batches + 1:
            break
        synchronize(dev)
        t0 = time.perf_counter()
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            logits = net(x)
            loss = seg_loss(logits, y, w_cldice=args.cldice)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        synchronize(dev)
        if i > 0:
            batch_seconds.append(time.perf_counter() - t0)
    avg_batch = np.mean(batch_seconds)
    train_batches = len(tr) // args.bs
    train_epoch = train_batches * avg_batch
    print(f"Measured training batch (excluding disk-to-batch DataLoader latency): {avg_batch:.3f} s")
    print(f"Projected train-only epoch: ~{train_epoch/60:.1f} minutes, "
          "BEFORE data-loading / validation / checkpoint overhead")

    net.eval()
    results = []
    acc = PQAccumulator()
    for _, row in va.head(args.val_images).iterrows():
        synchronize(dev)
        t0 = time.perf_counter()
        img = load_image(row.image_path)
        prob = predict_prob(net, img, dev, tta=bool(args.tta))
        inst = prob_to_instances(prob, **DEFAULTS)
        gt = load_instance_mask(row.mask_path)
        acc.update(gt, inst)
        synchronize(dev)
        dt = time.perf_counter() - t0
        results.append(dt)
        print(f"Validation image {len(results)}: {dt:.1f} s full pipeline")
    val_per_image = np.mean(results)
    validation_pass = val_per_image * len(va)
    n_validation_passes = math.ceil(args.epochs / args.val_every)
    projection = args.epochs * train_epoch + n_validation_passes * validation_pass
    print(f"Projected validation pass: ~{validation_pass/60:.1f} minutes")
    print(f"Projected {args.epochs}-epoch run with validation every {args.val_every}: "
          f"~{projection/3600:.1f} hours minimum estimate")
    print("This estimate is indicative only: excludes data-loader stall, model saving, "
          "post-processing search, compression and Kaggle submission time. "
          "Validation predictions from this probe are from an almost-untrained model "
          "and must NOT be reported as assignment results.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hardware-based COS711 A2 runtime probe")
    p.add_argument("--manifest", default="manifest.csv")
    p.add_argument("--model", choices=["light", "smp"], default="light")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--pretrained", type=int, default=0)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--val-images", type=int, default=2)
    p.add_argument("--tta", type=int, default=0)
    p.add_argument("--cldice", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=0)
    main(p.parse_args())
