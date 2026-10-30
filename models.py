import time
import numpy as np
import torch
import torch.nn as nn

from data import pad_to_multiple


def build_unet(encoder="resnet34", weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=1)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)


class LightUNet(nn.Module):
    def __init__(self, base=16, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([ConvBlock(1 if i == 0 else ch[i - 1], ch[i]) for i in range(depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([ConvBlock(ch[i] * 2, ch[i]) for i in range(depth)])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x); x = self.pool(x)
        for i in reversed(range(len(self.dec))):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.head(x)


def build_model(kind="smp", encoder="resnet34", weights="imagenet", base=16):
    return build_unet(encoder, weights) if kind == "smp" else LightUNet(base=base)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    n = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
    return n / 1024 ** 2


@torch.no_grad()
def predict_prob(model, img, device, tta=False, half=False, tile_size=512, overlap=64,
                 blend="uniform"):

    model.eval()
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    if blend not in {"uniform", "hann"}:
        raise ValueError("blend must be 'uniform' or 'hann'")
    img = np.asarray(img)
    if img.ndim != 2 or not all(img.shape):
        raise ValueError("Expected a nonempty HxW image")
    h, w = img.shape
    stride = tile_size - overlap

    def starts(n):
        if n <= tile_size:
            return [0]
        arr = list(range(0, n - tile_size + 1, stride))
        if arr[-1] != n - tile_size:
            arr.append(n - tile_size)
        return arr

    acc = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    weights = {}
    for y0 in starts(h):
        for x0 in starts(w):
            tile = np.ascontiguousarray(img[y0:y0+tile_size, x0:x0+tile_size])
            padded, (th, tw) = pad_to_multiple(tile, 32)
            inp = torch.from_numpy(np.ascontiguousarray(padded))[None, None].to(device)
            if half:
                inp = inp.half()
            def fwd(x):
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and not half)):
                    return torch.sigmoid(model(x).float())
            outs = [fwd(inp)]
            if tta:
                outs += [fwd(inp.flip((-1,))).flip((-1,)),
                         fwd(inp.flip((-2,))).flip((-2,)),
                         fwd(inp.flip((-1, -2))).flip((-1, -2))]
            patch = torch.stack(outs).mean(0)[0, 0, :th, :tw].cpu().numpy()
            if (th, tw) not in weights:
                if blend == "hann":


                    weights[th, tw] = np.maximum(
                        np.outer(np.hanning(th), np.hanning(tw)), 1e-3
                    ).astype(np.float32)
                else:
                    weights[th, tw] = np.ones((th, tw), dtype=np.float32)
            weight = weights[th, tw]
            acc[y0:y0+th, x0:x0+tw] += patch * weight
            count[y0:y0+th, x0:x0+tw] += weight
    return acc / np.maximum(count, np.finfo(np.float32).tiny)


@torch.no_grad()
def time_inference(model, device, shape=(1024, 1024), n=20, half=False):
    model.eval()
    x = torch.rand(1, 1, *shape, device=device)
    x = x.half() if half else x
    for _ in range(3):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n
