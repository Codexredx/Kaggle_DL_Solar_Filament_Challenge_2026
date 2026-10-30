import numpy as np


def match_instances(gt, pred, iou_thr=0.5):

    gt, pred = np.asarray(gt), np.asarray(pred)
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError("Ground truth and prediction must have the same HxW shape")
    if not 0.5 <= iou_thr <= 1:
        raise ValueError("Exclusive-map PQ requires 0.5 <= iou_thr <= 1")
    if gt.dtype.kind not in "biu" or pred.dtype.kind not in "biu":
        raise ValueError("Instance maps must contain integer labels")
    g, p = gt.ravel(), pred.ravel()
    if (g.size and g.min() < 0) or (p.size and p.min() < 0):
        raise ValueError("Instance labels must be nonnegative")
    gm, pm = int(g.max(initial=0)), int(p.max(initial=0))
    both = (g > 0) & (p > 0)


    if (gm + 1) * (pm + 1) <= 1_000_000:
        g = g.astype(np.int64, copy=False)
        p = p.astype(np.int64, copy=False)
        ga = np.bincount(g, minlength=gm + 1)
        pa = np.bincount(p, minlength=pm + 1)
        ng, npred = np.count_nonzero(ga[1:]), np.count_nonzero(pa[1:])
        base = pm + 1
        counts = np.bincount(g[both] * base + p[both])
        codes = np.flatnonzero(counts)
        gi, pi = codes // base, codes % base
        inter = counts[codes]
        g_inter_area, p_inter_area = ga[gi], pa[pi]
    else:


        g_ids, g_counts = np.unique(g[g > 0], return_counts=True)
        p_ids, p_counts = np.unique(p[p > 0], return_counts=True)
        ng, npred = len(g_ids), len(p_ids)

        compact_g = np.searchsorted(g_ids, g[both])
        compact_p = np.searchsorted(p_ids, p[both])
        pairs, inter = np.unique(np.column_stack((compact_g, compact_p)),
                                 axis=0, return_counts=True)
        gi, pi = pairs[:, 0], pairs[:, 1]
        g_inter_area, p_inter_area = g_counts[gi], p_counts[pi]

    ious = inter / (g_inter_area + p_inter_area - inter)
    matched = ious > iou_thr
    tp = int(matched.sum())

    _, split_counts = np.unique(gi[inter / p_inter_area >= 0.1], return_counts=True)
    _, merge_counts = np.unique(pi[inter / g_inter_area >= 0.1], return_counts=True)
    return (tp, int(npred - tp), int(ng - tp), float(ious[matched].sum()),
            int((split_counts > 1).sum()), int((merge_counts > 1).sum()))


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
