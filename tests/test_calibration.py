import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import infer
from models import build_model
from postprocess import DEFAULTS, evaluate, prob_to_instances, tune


def annotation_rows(observations=8):
    return pd.DataFrame([
        dict(obs_id=f"obs{i:02d}", image_id=f"annotator{annotator}-obs{i:02d}",
             image_path=f"image{i}.png", mask_path=f"mask{i}-{annotator}.png", split="train")
        for i in range(observations) for annotator in range(1 + i % 3)
    ])


def test_partition_keeps_observations_and_annotation_sets_together():
    rows = annotation_rows()
    fit, audit = infer.validation_partition(rows, tune_samples=3, seed=9)
    assert fit.obs_id.nunique() == 3
    assert audit.obs_id.nunique() == 5
    assert set(fit.obs_id).isdisjoint(audit.obs_id)
    assert set(fit.image_id) | set(audit.image_id) == set(rows.image_id)
    for subset in [fit, audit]:
        for obs_id, group in subset.groupby("obs_id"):
            assert set(group.image_id) == set(rows[rows.obs_id == obs_id].image_id)
    repeated = infer.validation_partition(rows.sample(frac=1, random_state=2), 3, seed=9)
    assert set(repeated[0].obs_id) == set(fit.obs_id)
    assert set(repeated[1].obs_id) == set(audit.obs_id)


def test_partition_leaves_audit_and_can_limit_its_observations():
    rows = annotation_rows()
    fit, audit = infer.validation_partition(rows, tune_samples=999, seed=2)
    assert fit.obs_id.nunique() == 7
    assert audit.obs_id.nunique() == 1
    _, limited = infer.validation_partition(rows, tune_samples=2, seed=2, validation_samples=2)
    assert limited.obs_id.nunique() == 2
    for obs_id in limited.obs_id.unique():
        assert len(limited[limited.obs_id == obs_id]) == len(rows[rows.obs_id == obs_id])


@pytest.mark.parametrize("rows,samples,limit", [
    (annotation_rows(1), 1, 0), (annotation_rows(4), 0, 0),
    (annotation_rows(4), 1, -1),
])
def test_partition_rejects_invalid_counts(rows, samples, limit):
    with pytest.raises(ValueError):
        infer.validation_partition(rows, samples, validation_samples=limit)


def test_ensemble_averages_probabilities_and_forwards_inference_recipe(monkeypatch):
    models = [object(), object(), object()]
    values = {id(model): value for model, value in zip(models, [0.1, 0.4, 1.0])}
    calls = []

    def predict(model, img, device, **kwargs):
        calls.append((model, device, kwargs))
        return np.full(img.shape, values[id(model)], dtype=np.float32)

    monkeypatch.setattr(infer, "predict_prob", predict)
    img = np.zeros((8, 13), np.float32)
    recipe = dict(tta=True, tile_size=256, overlap=96, blend="hann")
    result = infer.predict_ensemble(models, img, "cpu", **recipe)
    np.testing.assert_allclose(result, 0.5, atol=1e-7)
    assert result.dtype == np.float32
    assert [call[0] for call in calls] == models
    assert all(call[2] == recipe for call in calls)


def test_masksequence_loads_probabilities_and_integer_instance_maps(tmp_path):
    probability = np.arange(15, dtype=np.float32).reshape(3, 5) / 15
    prob_path = tmp_path / "prob.npy"
    np.save(prob_path, probability)
    seq = infer.MaskSequence([prob_path], probabilities=True)
    assert len(seq) == 1
    assert isinstance(seq[0], np.memmap)
    np.testing.assert_array_equal(list(seq)[0], probability)
    gt = np.zeros((10, 12), np.uint16)
    gt[1:3, 1:4] = 2
    gt[6:9, 7:11] = 7
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(mask_path), gt)
    masks = infer.MaskSequence([mask_path])
    np.testing.assert_array_equal(masks[0], gt)
    assert masks[0].dtype == np.int32
    with pytest.raises(IndexError):
        _ = masks[1]


@pytest.fixture
def synthetic_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint = tmp_path / "light.pt"
    torch.save(build_model("light", base=2).state_dict(), checkpoint)
    gt = np.zeros((48, 48), np.uint16)
    gt[8:16, 8:20] = 1
    gt[30:38, 27:39] = 2
    mask_path, prob_path = tmp_path / "mask.png", tmp_path / "prob.npy"
    assert cv2.imwrite(str(mask_path), gt)
    np.save(prob_path, (gt > 0).astype(np.float32) * 0.95)
    rows = annotation_rows(6)
    rows["mask_path"] = str(mask_path)
    manifest = tmp_path / "manifest.csv"
    rows.to_csv(manifest, index=False)
    monkeypatch.setattr(infer, "time_block_split", lambda df, seed: (df.iloc[:0], df))
    seen = []

    def cache(models, required, device, args, signature):
        seen.append(required.copy())
        return {obs_id: str(prob_path) for obs_id in required.obs_id.unique()}

    def search(probabilities, gts, verbose):
        assert len(probabilities) == len(gts)
        assert len(probabilities) > 0

        assert probabilities[0].shape == gts[0].shape == gt.shape
        return dict(DEFAULTS, close_px=0), 1.0

    monkeypatch.setattr(infer, "cache_predictions", cache)
    monkeypatch.setattr(infer, "tune", search)
    calibration = tmp_path / "calibration.json"
    csv = tmp_path / "submission.csv"
    common = ["--manifest", str(manifest), "--ckpt", str(checkpoint),
              "--model", "light", "--base", "2", "--tune-samples", "2",
              "--validation-samples", "2", "--params-out", str(calibration), "--out", str(csv)]
    return dict(common=common, calibration=calibration, csv=csv,
                rows=rows, seen=seen, tmp_path=tmp_path)


def test_evaluate_only_calibrates_then_reuses_params_and_excludes_fit_observations(synthetic_calibration):
    case = synthetic_calibration
    result = infer.main(infer.parser().parse_args(case["common"] + ["--evaluate-only"]))
    assert not case["csv"].exists()
    assert case["calibration"].is_file()
    saved = json.loads(case["calibration"].read_text())
    assert saved["postprocessing"]["close_px"] == 0
    assert saved["validation"]["metrics"]["PQ"] == pytest.approx(1.0)
    assert saved["tuning"]["observations"] == 2
    assert saved["validation"]["observations"] == 2
    assert set(saved["tuning"]["obs_ids"]).isdisjoint(saved["validation"]["obs_ids"])
    assert case["calibration"].with_suffix(".per_image.csv").is_file()
    reused = infer.main(infer.parser().parse_args(
        case["common"] + ["--evaluate-only", "--skip-tune", "--params", str(case["calibration"]) ]))
    assert reused["postprocessing"] == result["postprocessing"]
    assert set(case["seen"][-1].obs_id).isdisjoint(saved["tuning"]["obs_ids"])
    for obs_id in case["seen"][-1].obs_id.unique():
        assert len(case["seen"][-1][case["seen"][-1].obs_id == obs_id]) == len(
            case["rows"][case["rows"].obs_id == obs_id])


@pytest.mark.parametrize("changed", [
    ["--blend", "hann"], ["--tta", "1"], ["--overlap", "128"], ["--tile-size", "768"],
])
def test_signature_mismatch_stops_submission_before_inference(synthetic_calibration, changed):
    case = synthetic_calibration
    infer.main(infer.parser().parse_args(case["common"] + ["--evaluate-only"]))
    prior_calls = len(case["seen"])
    args = infer.parser().parse_args(case["common"] + [
        "--skip-tune", "--skip-validation", "--params", str(case["calibration"]), *changed])
    with pytest.raises(ValueError, match="different weights/TTA/tiles/blending"):
        infer.main(args)
    assert len(case["seen"]) == prior_calls
    assert not case["csv"].exists()


def test_changed_checkpoint_stops_params_reuse(synthetic_calibration):
    case = synthetic_calibration
    infer.main(infer.parser().parse_args(case["common"] + ["--evaluate-only"]))
    other_checkpoint = case["tmp_path"] / "other.pt"
    model = build_model("light", base=2)
    with torch.no_grad():
        next(model.parameters()).fill_(1.0)
    torch.save(model.state_dict(), other_checkpoint)
    args = infer.parser().parse_args(case["common"] + [
        "--skip-tune", "--skip-validation", "--params", str(case["calibration"]),
        "--ckpt", str(other_checkpoint)])
    with pytest.raises(ValueError, match="different weights/TTA/tiles/blending"):
        infer.main(args)
    assert not case["csv"].exists()


@pytest.mark.parametrize("other_flags", [
    [], ["--skip-tune"], ["--skip-validation"],
    ["--skip-tune", "--skip-validation"],
    ["--params", "placeholder.json", "--skip-tune"],
    ["--params", "placeholder.json", "--skip-validation"],
])
def test_default_export_requires_a_calibration_and_both_skip_flags(synthetic_calibration, other_flags):
    case = synthetic_calibration
    args = infer.parser().parse_args(case["common"] + ["--use-default-postprocessing", *other_flags])
    with pytest.raises(ValueError, match="export option"):
        infer.main(args)
    assert not case["csv"].exists()


def test_default_export_preserves_calibration_and_reports_matching_metrics(synthetic_calibration, monkeypatch):
    case = synthetic_calibration
    infer.main(infer.parser().parse_args(case["common"] + ["--evaluate-only"]))
    saved = json.loads(case["calibration"].read_text())
    saved["validation"]["metrics"]["PQ"] = 0.25
    saved["validation"]["default_postprocessing_metrics"]["PQ"] = 0.75
    case["calibration"].write_text(json.dumps(saved), encoding="utf-8")
    original_calibration = case["calibration"].read_bytes()
    manifest = Path(case["common"][case["common"].index("--manifest") + 1])
    test_row = pd.DataFrame([dict(obs_id="test_obs", image_id="test_obs", image_path="mocktest.png",
                                  mask_path="", split="test")])
    pd.concat([case["rows"], test_row]).to_csv(manifest, index=False)
    monkeypatch.setattr(infer, "load_image", lambda path: np.zeros((4, 4), np.float32))
    monkeypatch.setattr(infer, "predict_ensemble", lambda *args, **kwargs: np.zeros((4, 4), np.float32))
    used_params = []

    def instances(prob, **params):
        used_params.append(params)
        labels = np.zeros((2048, 2048), np.int32)
        labels[10:20, 10:20] = 1
        return labels

    monkeypatch.setattr(infer, "prob_to_instances", instances)
    args = infer.parser().parse_args(case["common"] + [
        "--params", str(case["calibration"]), "--skip-tune", "--skip-validation",
        "--use-default-postprocessing"])
    result = infer.main(args)
    metadata = json.loads(case["csv"].with_suffix(".metadata.json").read_text())
    assert used_params == [DEFAULTS]
    assert result["postprocessing"] == metadata["postprocessing"] == DEFAULTS
    assert not metadata["used_tuned_postprocessing"]
    assert metadata["validation"]["metrics"]["PQ"] == 0.75
    assert metadata["calibration_file"] == str(case["calibration"])
    assert case["calibration"].read_bytes() == original_calibration
    assert pd.read_csv(case["csv"]).filament_id.tolist() == ["test_obs_1"]


@pytest.mark.parametrize("audit_limit,expected_key", [(0, "full_validation"), (2, "sampled_validation")])
def test_validation_summary_pools_counts_and_iou_instead_of_averaging_pq(
        synthetic_calibration, monkeypatch, audit_limit, expected_key):
    case = synthetic_calibration
    fit, audit = infer.validation_partition(case["rows"], 2, seed=0,
                                           validation_samples=audit_limit)
    fit_ids = set(fit.obs_id)

    def metrics(tp, fp, fn, sq, split, merge):
        denominator = tp + 0.5 * fp + 0.5 * fn
        return dict(PQ=tp * sq / denominator, SQ=sq, RQ=tp / denominator,
                    TP=tp, FP=fp, FN=fn, split=split, merge=merge)

    fit_metrics = metrics(9, 0, 0, 0.6, 2, 1)
    audit_metrics = metrics(1, 9, 0, 0.8, 3, 4)

    def evaluate_subset(rows, paths, params):
        return (fit_metrics if set(rows.obs_id) == fit_ids else audit_metrics), []

    monkeypatch.setattr(infer, "evaluate_rows", evaluate_subset)
    args = infer.parser().parse_args(case["common"] + [
        "--evaluate-only", "--validation-samples", str(audit_limit)])
    result = infer.main(args)
    pooled = result[expected_key]
    other_key = "sampled_validation" if expected_key == "full_validation" else "full_validation"
    assert other_key not in result
    assert pooled["observations"] == fit.obs_id.nunique() + audit.obs_id.nunique()
    assert pooled["annotation_sets"] == len(fit) + len(audit)
    assert pooled["metrics"]["TP"] == 10
    assert pooled["metrics"]["FP"] == 9
    assert pooled["metrics"]["FN"] == 0
    assert pooled["metrics"]["split"] == 5
    assert pooled["metrics"]["merge"] == 5
    assert pooled["metrics"]["SQ"] == pytest.approx(0.62)
    assert pooled["metrics"]["PQ"] == pytest.approx(6.2 / 14.5)
    assert pooled["metrics"]["PQ"] != pytest.approx((fit_metrics["PQ"] + audit_metrics["PQ"]) / 2)


def test_watershed_option_calls_function_instead_of_shadowed_boolean():
    prob = np.zeros((64, 64), np.float32)
    cv2.circle(prob, (24, 32), 10, 0.95, -1)
    cv2.circle(prob, (40, 32), 10, 0.95, -1)
    instances = prob_to_instances(prob, low=0.3, high=0.6, close_px=0,
                                  min_area=1, watershed=True, ws_min_dist=8)
    assert instances.dtype == np.int32
    assert instances.max() == 2
    assert np.array_equal(instances > 0, prob > 0)


def test_tune_custom_grid_uses_pq_and_rejects_spurious_area_filter():
    gt = np.zeros((32, 32), np.int32)
    gt[8:16, 8:20] = 1
    prob = (gt > 0).astype(np.float32) * 0.9
    grid = dict(low=[0.2, 0.5], high=[0.6, 0.8], close_px=[0],
                min_area=[1, 1000], watershed=[False])
    params, score = tune([prob], [gt], grid=grid, verbose=False)
    assert score == pytest.approx(1.0)
    assert params["min_area"] == 1
    assert evaluate([prob], [gt], **params)["PQ"] == pytest.approx(score)


def test_default_search_is_deterministic_and_does_not_degrade_its_start():
    gt = np.zeros((64, 64), np.int32)
    gt[16:32, 12:44] = 1
    prob = (gt > 0).astype(np.float32) * 0.9
    prob[16:32, 26:29] = 0.1
    first = tune([prob], [gt], verbose=False)
    second = tune([prob], [gt], verbose=False)
    assert first == second
    assert first[1] >= evaluate([prob], [gt], **DEFAULTS)["PQ"]
    assert first[0]["low"] <= first[0]["high"]
