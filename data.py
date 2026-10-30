


import cv2
import numpy as np
import pandas as pd


def load_image(path):
    path = str(path)
    if path.endswith(".npy"):
        img = np.load(path).astype(np.float32)
    elif path.endswith((".fits", ".fts", ".fit")):
        from astropy.io import fits
        img = fits.getdata(path).astype(np.float32)
    else:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        img = raw.astype(np.float32)
        if img.ndim == 3:
            img = img.mean(-1)
    img = np.nan_to_num(img)
    lo, hi = np.percentile(img, [1, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def load_instance_mask(path):
    path = str(path)
    m = np.load(path) if path.endswith(".npy") else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not load instance mask: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() <= 1:
        _, m = cv2.connectedComponents((m > 0).astype(np.uint8), connectivity=8)
    return m.astype(np.int32)


def _polys_to_mask(polys, h, w):
    from pycocotools import mask as mu
    if not polys:
        return np.zeros((h, w), dtype=np.uint8)
    rles = mu.frPyObjects(polys, h, w)
    return mu.decode(mu.merge(rles)).astype(np.uint8)


def build_manifest(root, cache_dir="masks", out_csv="manifest.csv"):
    import json, os
    from collections import defaultdict
    from pathlib import Path
    root = Path(root).resolve()
    os.makedirs(cache_dir, exist_ok=True)
    json_files = sorted((root / "train").glob("*.json"))
    if len(json_files) != 1:
        raise ValueError(f"Expected one training COCO JSON under {root / 'train'}, found: {json_files}")
    with open(json_files[0], encoding="utf-8") as handle:
        coco = json.load(handle)
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[str(a["image_id"])].append(a)

    rows = []
    for im in coco["images"]:
        h, w = im["height"], im["width"]
        inst = np.zeros((h, w), np.uint16)
        for k, a in enumerate(anns[str(im["id"])], start=1):
            inst[_polys_to_mask(a["segmentation"], h, w) > 0] = k
        mp = Path(cache_dir) / f"{im['id']}.png"
        if not cv2.imwrite(str(mp), inst):
            raise OSError(f"Unable to write instance mask: {mp}")
        stem = Path(im["file_name"]).stem
        image_path = root / "train" / "train_images" / im["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Training image missing: {image_path}")
        rows.append(dict(image_path=str(image_path),
                         mask_path=str(mp), image_id=str(im["id"]), obs_id=stem,
                         time=stem[:14], instrument=stem[14:], n_filaments=len(anns[str(im["id"])]),
                         split="train"))
    test_dir = root / "test" / "test_images"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test image directory missing: {test_dir}")
    for p in sorted(test_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = p.stem
        rows.append(dict(image_path=str(p), mask_path="", image_id=stem, obs_id=stem,
                         time=stem[:14], instrument=stem[14:], n_filaments=-1, split="test"))
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.groupby("split").size().to_dict(), "| instruments:", df.instrument.value_counts().to_dict())
    return df


def time_block_split(df, block=20, val_every=5, buffer=2, seed=0):
    obs = (df.drop_duplicates("obs_id").sort_values("time")["obs_id"].tolist())
    n_blocks = int(np.ceil(len(obs) / block))
    offset = np.random.default_rng(seed).integers(0, val_every)
    val, drop = set(), set()
    for b in range(n_blocks):
        lo, hi = b * block, min((b + 1) * block, len(obs))
        if (b + offset) % val_every == 0:
            val.update(obs[lo:hi])
            drop.update(obs[max(0, lo - buffer):lo]); drop.update(obs[hi:hi + buffer])
    train = [o for o in obs if o not in val and o not in drop]
    return df[df.obs_id.isin(train)].reset_index(drop=True), \
           df[df.obs_id.isin(val)].reset_index(drop=True)


def pad_to_multiple(img, m=32, value=0):
    h, w = img.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    pad = [(0, 0)] * (img.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(img, pad, constant_values=value), (h, w)


def random_crop(img, mask, size, pos_prob=0.7, rng=np.random):
    h, w = img.shape
    size = min(size, h, w)
    ys, xs = np.nonzero(mask) if (mask.any() and rng.rand() < pos_prob) else (None, None)
    if ys is not None:
        k = rng.randint(len(ys))
        cy, cx = ys[k], xs[k]
        y0 = np.clip(cy - rng.randint(size), 0, h - size)
        x0 = np.clip(cx - rng.randint(size), 0, w - size)
    else:
        y0, x0 = rng.randint(0, h - size + 1), rng.randint(0, w - size + 1)
    return img[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size]


def augment(img, mask, rng=np.random):
    if rng.rand() < 0.5: img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.rand() < 0.5: img, mask = img[::-1], mask[::-1]
    k = rng.randint(4); img, mask = np.rot90(img, k), np.rot90(mask, k)
    img = np.ascontiguousarray(img); mask = np.ascontiguousarray(mask)

    if rng.rand() < 0.8:
        img = np.clip(img * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1), 0, 1)
        img = img ** rng.uniform(0.8, 1.25)
    if rng.rand() < 0.3:
        s = rng.uniform(0.5, 1.5)
        img = cv2.GaussianBlur(img, (0, 0), s)
    if rng.rand() < 0.3:
        img = np.clip(img + rng.normal(0, rng.uniform(0.005, 0.03), img.shape), 0, 1)
    return img.astype(np.float32), mask


def rle_encode(mask):
    from pycocotools import mask as mu
    binary = np.asfortranarray(np.asarray(mask, dtype=np.uint8))
    if binary.ndim != 2:
        raise ValueError(f"RLE mask must be HxW; got shape={binary.shape}")
    return mu.encode(binary)["counts"].decode("ascii")


def rle_counts(encoded):
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("Compressed RLE must be a nonempty string")
    counts = []
    position = 0
    while position < len(encoded):
        value, shift = 0, 0
        while True:
            if position == len(encoded):
                raise ValueError("Truncated compressed RLE count")
            code = ord(encoded[position]) - 48
            position += 1
            if not 0 <= code <= 63:
                raise ValueError("Invalid character in compressed RLE")
            value |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    value -= 1 << shift
                break
            if shift > 60:
                raise ValueError("Compressed RLE count is too large")
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("Compressed RLE contains a negative run length")
        counts.append(value)
    return counts


def rle_decode(s, shape):
    from pycocotools import mask as mu
    if not isinstance(s, str) or not s:
        return np.zeros(shape, dtype=np.uint8)
    counts = rle_counts(s)
    pixels = int(np.prod(shape))
    if sum(counts) != pixels:
        raise ValueError(f"RLE runs cover {sum(counts)} pixels; expected {pixels}")
    return mu.decode({"counts": s.encode("ascii"), "size": list(shape)}).astype(np.uint8)


def _selftest():
    m = np.zeros((50, 60), np.uint8); m[5:20, 10:30] = 1; m[30:35, 40:58] = 1
    assert (rle_decode(rle_encode(m), m.shape) == m).all(); print("RLE round-trip OK")

    df = pd.DataFrame({"obs_id": [f"o{i}" for i in range(200) for _ in range(2)],
                       "time": [i for i in range(200) for _ in range(2)]})
    tr, va = time_block_split(df)
    print(len(tr), len(va), "overlap of obs ids:", set(tr.obs_id) & set(va.obs_id))

    img = np.random.rand(300, 300).astype(np.float32); mk = np.zeros((300, 300), np.int32); mk[100:110, 50:250] = 1
    ci, cm = random_crop(img, mk, 128); ai, am = augment(ci, cm); print("crop/aug shapes", ai.shape, am.shape)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("build")
        ap.add_argument("--root", required=True); ap.add_argument("--cache", default="masks")
        ap.add_argument("--out", default="manifest.csv")
        a = ap.parse_args(); build_manifest(a.root, a.cache, a.out)
    else:
        _selftest()
