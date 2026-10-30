import itertools
import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as sk_watershed
from scipy import ndimage as ndi

from pq_metric import PQAccumulator

DEFAULTS = dict(low=0.35, high=0.6, close_px=3, min_area=30, watershed=False, ws_min_dist=15)


def prob_to_instances(prob, low=0.35, high=0.6, close_px=3, min_area=30,
                      watershed=False, ws_min_dist=15):

    mask = apply_hysteresis_threshold(prob, low, high).astype(np.uint8)


    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    mask = keep[labels]

    if watershed and mask.any():
        dist = ndi.distance_transform_edt(mask)
        pts = peak_local_max(dist, min_distance=ws_min_dist, labels=mask.astype(int),
                             exclude_border=False)
        markers = np.zeros(mask.shape, np.int32)
        markers[tuple(pts.T)] = np.arange(1, len(pts) + 1)
        return sk_watershed(-dist, markers, mask=mask).astype(np.int32)

    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)


def evaluate(probs, gts, **params):
    if len(probs) != len(gts) or not len(probs):
        raise ValueError("Need equal, nonempty probability and ground-truth lists")
    acc = PQAccumulator()
    for p, g in zip(probs, gts):
        acc.update(g, prob_to_instances(p, **params))
    return acc.compute()


def tune(probs, gts, grid=None, verbose=True):


    if grid is None:
        candidates = dict(low=[0.2, 0.35, 0.45, 0.6],
                          high=[0.5, 0.6, 0.7, 0.85],
                          close_px=[0, 3, 6, 9],
                          min_area=[30, 100, 300, 1000])
        best, best_pq, scores = dict(DEFAULTS), -1.0, {}

        def score(params):
            key = tuple(sorted(params.items()))
            if key not in scores:
                result = evaluate(probs, gts, **params)
                scores[key] = result["PQ"]
                if verbose:
                    print(params, f"PQ={result['PQ']:.4f}", flush=True)
            return scores[key]


        for start in [dict(DEFAULTS), {**DEFAULTS, "low": 0.45,
                                       "high": 0.7, "close_px": 6, "min_area": 100}]:
            current, value = start, score(start)
            for _ in range(2):
                changed = False
                for name, values in candidates.items():
                    for candidate in values:
                        params = {**current, name: candidate}
                        if params["low"] > params["high"]:
                            continue
                        pq = score(params)
                        if pq > value:
                            current, value, changed = params, pq, True
                if not changed:
                    break
            if value > best_pq:
                best, best_pq = current, value
        return best, best_pq
    keys = list(grid)
    best, best_pq = None, -1
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        if params["low"] > params["high"]:
            continue
        r = evaluate(probs, gts, **params)
        if verbose:
            print(params, f"PQ={r['PQ']:.4f} split={r['split']} merge={r['merge']}")
        if r["PQ"] > best_pq:
            best, best_pq = params, r["PQ"]
    return best, best_pq


if __name__ == "__main__":

    rng = np.random.default_rng(0)
    gt = np.zeros((256, 256), np.int32)
    gt[40:50, 20:200] = 1; gt[120:128, 30:220] = 2
    prob = (gt > 0).astype(float) * 0.9 + rng.normal(0, 0.12, gt.shape).clip(-0.3, 0.3)
    prob[40:50, 100:104] = 0.1
    prob = prob.clip(0, 1)
    print("default :", evaluate([prob], [gt], **DEFAULTS))
    print("no close:", evaluate([prob], [gt], **{**DEFAULTS, "close_px": 0}))
    print("tuned   :", tune([prob], [gt], verbose=False))
