from collections import Counter

import numpy as np
import pytest
import torch

from models import predict_prob
from pq_metric import PQAccumulator, match_instances


class IdentityLogits(torch.nn.Module):
    def forward(self, x):
        return x


class EdgeBiasedLogits(torch.nn.Module):
    def forward(self, x):
        prob = torch.full_like(x, 0.9)
        prob[..., :3, :] = 0.1
        prob[..., -3:, :] = 0.1
        prob[..., :, :3] = 0.1
        prob[..., :, -3:] = 0.1
        return torch.logit(prob)


@pytest.mark.parametrize("blend", ["uniform", "hann"])
@pytest.mark.parametrize("shape", [(53, 71), (7, 13), (1, 1), (32, 32)])
@pytest.mark.parametrize("tta", [False, True])
def test_tiling_preserves_pointwise_model_at_every_pixel(blend, shape, tta):
    img = np.random.default_rng(12).normal(size=shape).astype(np.float32)
    actual = predict_prob(IdentityLogits(), img, torch.device("cpu"),
                          tta=tta, tile_size=32, overlap=12, blend=blend)
    expected = torch.sigmoid(torch.from_numpy(img)).numpy()
    assert actual.shape == img.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_hann_downweights_tile_boundary_artifacts_in_overlap():
    img = np.zeros((32, 48), dtype=np.float32)
    kwargs = dict(tile_size=32, overlap=16)
    uniform = predict_prob(EdgeBiasedLogits(), img, torch.device("cpu"), **kwargs)
    hann = predict_prob(EdgeBiasedLogits(), img, torch.device("cpu"),
                        blend="hann", **kwargs)
    assert uniform[16, 18] == pytest.approx(0.5)
    assert hann[16, 18] > 0.8

    assert hann[0, 0] == pytest.approx(0.1)
    assert np.isfinite(hann).all()


@pytest.mark.parametrize("kwargs", [dict(tile_size=0), dict(overlap=-1),
                                   dict(tile_size=32, overlap=32), dict(blend="invalid")])
def test_invalid_tiling_arguments_fail(kwargs):
    with pytest.raises(ValueError):
        predict_prob(IdentityLogits(), np.zeros((10, 10), np.float32),
                     torch.device("cpu"), **kwargs)


def reference_match(gt, pred):

    ga = Counter(int(v) for v in gt.ravel() if v > 0)
    pa = Counter(int(v) for v in pred.ravel() if v > 0)
    inter = Counter((int(g), int(p)) for g, p in zip(gt.ravel(), pred.ravel())
                    if g > 0 and p > 0)
    matched = [n / (ga[g] + pa[p] - n) for (g, p), n in inter.items()
               if n / (ga[g] + pa[p] - n) > 0.5]
    split = Counter(g for (g, p), n in inter.items() if n / pa[p] >= 0.1)
    merge = Counter(p for (g, p), n in inter.items() if n / ga[g] >= 0.1)
    tp = len(matched)
    return (tp, len(pa) - tp, len(ga) - tp, sum(matched),
            sum(n > 1 for n in split.values()), sum(n > 1 for n in merge.values()))


def test_pq_perfect_empty_fragmented_merged_and_unmatched():
    gt = np.zeros((20, 20), np.int32)
    gt[2:6, 2:12] = 1
    gt[12:16, 2:12] = 2
    assert match_instances(gt, gt) == (2, 0, 0, 2.0, 0, 0)
    empty = np.zeros_like(gt)
    assert match_instances(empty, empty) == (0, 0, 0, 0.0, 0, 0)
    assert match_instances(gt, empty) == (0, 0, 2, 0.0, 0, 0)
    assert match_instances(empty, gt) == (0, 2, 0, 0.0, 0, 0)
    fragmented = gt.copy()
    fragmented[2:6, 7:12] = 3
    assert match_instances(gt, fragmented) == (1, 2, 1, 1.0, 1, 0)
    merged = (gt > 0).astype(np.int32)
    assert match_instances(gt, merged) == (0, 1, 2, 0.0, 0, 1)
    acc = PQAccumulator()
    acc.update(gt, gt)
    assert acc.compute()["PQ"] == 1
    assert acc.compute()["RQ"] == 1


@pytest.mark.parametrize("sparse", [False, True])
def test_metric_matches_reference_over_random_maps(sparse):
    rng = np.random.default_rng(42)
    for _ in range(30):
        gt = rng.integers(0, 5, size=(17, 23), dtype=np.int64)
        pred = gt.copy()
        changed = rng.random(gt.shape) < rng.uniform(0.1, 0.9)
        pred[changed] = rng.integers(0, 7, size=changed.sum())
        if sparse:
            gt *= 1_000_000_007
            pred *= 2_000_000_011
        actual, expected = match_instances(gt, pred), reference_match(gt, pred)
        assert actual[:3] == expected[:3]
        assert actual[3] == pytest.approx(expected[3])
        assert actual[4:] == expected[4:]


def test_uint64_labels_do_not_overflow_or_allocate_by_largest_id():
    gt = np.array([[0, 2**63 + 3, 2**63 + 3]], np.uint64)
    pred = np.array([[0, 10, 10]], np.int64)
    assert match_instances(gt, pred) == (1, 0, 0, 1.0, 0, 0)


@pytest.mark.parametrize("gt,pred,threshold", [
    (np.zeros((2, 3), int), np.zeros((3, 2), int), 0.5),
    (np.array([[-1, 0]]), np.array([[0, 0]]), 0.5),
    (np.zeros((2, 3)), np.zeros((2, 3)), 0.5),
    (np.zeros((2, 3), int), np.zeros((2, 3), int), 0.4),
])
def test_invalid_instance_maps_or_ambiguous_threshold_fail(gt, pred, threshold):
    with pytest.raises(ValueError):
        match_instances(gt, pred, threshold)
