import argparse, json, os, time
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
    def __init__(self, df, crop=512, steps=None):
        self.df, self.crop = df.reset_index(drop=True), crop
        self.n = steps or len(self.df)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        row = self.df.iloc[i % len(self.df)]
        img = load_image(row.image_path)
        mask = (load_instance_mask(row.mask_path) > 0).astype(np.uint8)
        img, mask = random_crop(img, mask, self.crop)
        img, mask = augment(img, mask)
        return torch.from_numpy(img)[None], torch.from_numpy(mask.astype(np.float32))[None]


def validate(model, val_df, device, tta=False, params=DEFAULTS):
    acc = PQAccumulator()
    for _, r in val_df.iterrows():
        prob = predict_prob(model, load_image(r.image_path), device, tta=tta)
        gt = load_instance_mask(r.mask_path)
        acc.update(gt, prob_to_instances(prob, **params))
    return acc.compute(), None, None


def run(a, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(a.out, f"seed{seed}"); os.makedirs(out, exist_ok=True)

    df = pd.read_csv(a.manifest)
    df = df[df.split == "train"]
    tr_df, va_df = time_block_split(df, seed=a.split_seed)
    print(f"[seed {seed}] train rows {len(tr_df)} | val rows {len(va_df)}")

    model = build_model(a.model, a.encoder, "imagenet" if a.pretrained else None, a.base).to(device)
    teacher = None
    if a.teacher:                                  
        teacher = build_model("smp", a.encoder, None).to(device)
        teacher.load_state_dict(torch.load(a.teacher, map_location=device)); teacher.eval()

    dl = DataLoader(FilamentDS(tr_df, a.crop, steps=a.steps_per_epoch), batch_size=a.bs,
                    shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl); warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") 

    best, log, t0 = -1, [], time.time()
    for ep in range(a.epochs):
        model.train(); run_loss = 0
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
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item()

        msg = dict(epoch=ep, loss=run_loss / len(dl))
        if (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            res, _, _ = validate(model, va_df, device)
            msg.update({k: float(v) for k, v in res.items()})
            if res["PQ"] > best:
                best = res["PQ"]; torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        print(msg); log.append(msg)

    summary = dict(seed=seed, best_val_PQ=best, train_time_s=time.time() - t0,
                   params=count_params(model), size_mb=model_size_mb(model), args=vars(a), log=log)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="smp", choices=["smp", "light"])
    p.add_argument("--encoder", default="resnet34"); p.add_argument("--base", type=int, default=16)
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--bs", type=int, default=8)
    p.add_argument("--crop", type=int, default=512); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps-per-epoch", "--steps_per_epoch", dest="steps_per_epoch", type=int, default=None)
    p.add_argument("--cldice", type=float, default=0.0)
    p.add_argument("--teacher", default=None); p.add_argument("--kd", type=float, default=1.0)
    p.add_argument("--val-every", "--val_every", dest="val_every", type=int, default=2); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--split-seed", type=int, default=0,
                   help="Fixed validation split for ALL training seeds; default 0")
    a = p.parse_args()
    scores = [run(a, s) for s in a.seed]
    print(f"val PQ over seeds: mean={np.mean(scores):.4f}  std={np.std(scores):.4f}")
