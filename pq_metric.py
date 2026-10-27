


import numpy as np


def match_instances(gt, pred, iou_thr=0.5):
    g = gt.astype(np.int64).ravel()
    p = pred.astype(np.int64).ravel()

    g_ids, g_area = np.unique(g[g > 0], return_counts=True)
    p_ids, p_area = np.unique(p[p > 0], return_counts=True)
    g_area = dict(zip(g_ids.tolist(), g_area.tolist()))
    p_area = dict(zip(p_ids.tolist(), p_area.tolist()))

    both = (g > 0) & (p > 0)
    if both.any():

        base = int(p.max()) + 1
        code = g[both] * base + p[both]
        codes, inter = np.unique(code, return_counts=True)
        gi, pi = codes // base, codes % base
    else:
        gi = pi = inter = np.array([], dtype=np.int64)

    tp, iou_sum = 0, 0.0
    for a, b, i in zip(gi, pi, inter):
        union = g_area[int(a)] + p_area[int(b)] - int(i)
        iou = i / union
        if iou > iou_thr:
            tp += 1
            iou_sum += iou

    fp = len(p_area) - tp
    fn = len(g_area) - tp


    n_split = n_merge = 0
    if len(gi):
        from collections import defaultdict
        per_g, per_p = defaultdict(int), defaultdict(int)
        for a, b, i in zip(gi, pi, inter):
            if i / p_area[int(b)] >= 0.1:
                per_g[int(a)] += 1
            if i / g_area[int(a)] >= 0.1:
                per_p[int(b)] += 1
        n_split = sum(v > 1 for v in per_g.values())
        n_merge = sum(v > 1 for v in per_p.values())
    return tp, fp, fn, iou_sum, n_split, n_merge


class PQAccumulator:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.iou = 0.0
        self.split = self.merge = 0

    def update(self, gt, pred):
        tp, fp, fn, iou, s, m = match_instances(gt, pred)
        self.tp += tp; self.fp += fp; self.fn += fn
        self.iou += iou; self.split += s; self.merge += m

    def compute(self):
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        pq = self.iou / denom if denom > 0 else 0.0
        sq = self.iou / self.tp if self.tp > 0 else 0.0
        rq = self.tp / denom if denom > 0 else 0.0
        return dict(PQ=pq, SQ=sq, RQ=rq, TP=self.tp, FP=self.fp, FN=self.fn,
                    split=self.split, merge=self.merge)


if __name__ == "__main__":

    gt = np.zeros((100, 100), int); gt[10:30, 10:60] = 1; gt[60:80, 10:60] = 2
    acc = PQAccumulator(); acc.update(gt, gt.copy())
    print("perfect  :", acc.compute()["PQ"])

    frag = gt.copy(); frag[10:30, 35:60] = 3
    acc = PQAccumulator(); acc.update(gt, frag)
    print("fragment :", acc.compute())

    merged = np.where(gt > 0, 1, 0)
    acc = PQAccumulator(); acc.update(gt, merged)
    print("merged   :", acc.compute())
