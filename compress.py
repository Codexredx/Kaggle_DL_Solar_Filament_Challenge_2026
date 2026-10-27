


import argparse, copy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from data import time_block_split
from losses import seg_loss
from models import build_model, count_params, model_size_mb, time_inference
from train import FilamentDS, validate


def conv_params(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv2d)]


def sparsity(model):
    w = torch.cat([m.weight.flatten() for m, _ in conv_params(model)])
    return float((w == 0).float().mean())


def prune_model(model, amount, tr_df, device, finetune_epochs=0, crop=256, bs=1, workers=0):
    m = copy.deepcopy(model)
    prune.global_unstructured(conv_params(m), pruning_method=prune.L1Unstructured, amount=amount)
    if finetune_epochs:
        dl = DataLoader(FilamentDS(tr_df, crop, steps=200), batch_size=bs, shuffle=True, num_workers=workers)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        for _ in range(finetune_epochs):
            m.train()
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = seg_loss(m(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
    for mod, name in conv_params(m):
        prune.remove(mod, name)
    return m


def evaluate_variant(name, model, va_df, device, half=False):
    m = copy.deepcopy(model).half() if half else model
    t = time_inference(m, device, shape=(512, 512), half=half)

    from models import predict_prob
    from data import load_image, load_instance_mask
    from postprocess import prob_to_instances, DEFAULTS
    from pq_metric import PQAccumulator
    acc = PQAccumulator()
    for _, r in va_df.iterrows():
        prob = predict_prob(m, load_image(r.image_path), device, half=half)
        acc.update(load_instance_mask(r.mask_path), prob_to_instances(prob, **DEFAULTS))
    pq = acc.compute()["PQ"]
    return dict(variant=name, params=count_params(model), sparsity=round(sparsity(model), 3),
                size_mb=round(model_size_mb(m), 2), ms_per_512tile_net=round(t * 1000, 1), val_PQ=round(float(pq), 4))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="smp"); p.add_argument("--encoder", default="resnet34")
    p.add_argument("--base", type=int, default=16); p.add_argument("--finetune", type=int, default=2)
    p.add_argument("--bs", type=int, default=1); p.add_argument("--crop", type=int, default=256)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=24, help="0 evaluates the complete validation set")
    p.add_argument("--out", default="compression_results.csv")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.manifest); tr, va = time_block_split(df[df.split == "train"], seed=a.split_seed)
    if a.val_limit > 0:
        va = va.sample(n=min(a.val_limit, len(va)), random_state=a.split_seed).reset_index(drop=True)
    print(f"Compression evaluation uses {len(va)} validation annotation sets")
    base = build_model(a.model, a.encoder, None, a.base).to(device)
    base.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=True))

    rows = [evaluate_variant("fp32 baseline", base, va, device)]
    if device.type == "cuda":
        rows.append(evaluate_variant("fp16", base, va, device, half=True))
    for amt in [0.3, 0.5, 0.7, 0.9]:
        pm = prune_model(base, amt, tr, device, a.finetune, crop=a.crop, bs=a.bs, workers=a.workers)
        rows.append(evaluate_variant(f"prune {int(amt*100)}% (+{a.finetune} ep finetune)", pm, va, device))
        del pm
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("NOTE: Unstructured zero weights remain in dense tensors; file size and inference speed may not improve.")
    print("Network-only timing is NOT end-to-end timing; infer.py measures full pipeline.")
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(a.out, index=False)
