import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

import train


class TinySeg(torch.nn.Module):


    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.dropout = torch.nn.Dropout2d(0.2)

    def forward(self, x):
        return self.dropout(self.conv(x))


@pytest.fixture
def cpu_training(tmp_path, monkeypatch):

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(12)
    rows = []
    for index in range(4):
        image = tmp_path / f"image{min(index, 2)}.npy"
        mask = tmp_path / f"mask{index}.npy"
        if not image.exists():
            np.save(image, rng.random((32, 32), dtype=np.float32))
        value = np.zeros((32, 32), np.int32)
        value[8:24, 8:24] = 1
        np.save(mask, value)
        rows.append(dict(image_path=str(image), mask_path=str(mask), image_id=str(index),
                         obs_id=str(min(index, 2)), time=index, split="train"))
    frame = pd.DataFrame(rows)
    manifest = tmp_path / "manifest.csv"
    frame.to_csv(manifest, index=False)

    def predict(model, image, device, **kwargs):
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(image)[None, None].to(device)
            return torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]

    monkeypatch.setattr(train, "time_block_split", lambda df, **kw: (df.iloc[:2], df.iloc[2:]))
    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs: TinySeg())
    monkeypatch.setattr(train, "predict_prob", predict)

    def make_args(name, **overrides):
        args = train.make_parser().parse_args([
            "--manifest", str(manifest), "--out", str(tmp_path / name),
            "--model", "light", "--epochs", "3", "--bs", "1", "--crop", "32",
            "--samples-per-epoch", "4", "--workers", "0", "--pretrained", "1",
            "--val-every", "1",
        ])
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    yield tmp_path, frame, make_args
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("sampling", ["annotation", "observation"])
def test_interrupted_resume_is_bit_identical_and_accepts_pretrained_change(cpu_training, monkeypatch, sampling):
    root, _, make_args = cpu_training
    continuous = make_args("continuous", sampling=sampling)
    train.run(continuous, 7)
    resumed = make_args("resumed", sampling=sampling)
    real_save = train._save_json

    def interrupt_after_first_epoch(value, path):
        real_save(value, path)
        if len(value["log"]) == 1:
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(train, "_save_json", interrupt_after_first_epoch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train.run(resumed, 7)


    resumed.resume, resumed.pretrained = "auto", 0
    train.run(resumed, 7)
    left = torch.load(root / "continuous/seed7/last.pt", weights_only=True)
    right = torch.load(root / "resumed/seed7/last.pt", weights_only=True)
    assert left["log"] == right["log"]
    assert left["scheduler"] == right["scheduler"]
    for name in left["model"]:
        assert torch.equal(left["model"][name], right["model"][name])
    for state_id, state in left["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, right["optimizer"]["state"][state_id][name])
    assert torch.equal(left["rng"]["loader"], right["rng"]["loader"])
    assert torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    assert left["rng"]["numpy"] == right["rng"]["numpy"]
    assert left["rng"]["python"] == right["rng"]["python"]
    assert json.loads((root / "resumed/seed7/summary.json").read_text())["log"] == left["log"]


    resumed.epochs += 1
    with pytest.raises(ValueError, match="mismatch"):
        train.run(resumed, 7)


def test_warm_start_loads_weights_and_starts_a_fresh_optimizer(cpu_training, monkeypatch):
    root, _, make_args = cpu_training
    source = TinySeg()
    initial = {name: tensor.detach().clone() for name, tensor in source.state_dict().items()}
    checkpoint = root / "initial.pt"
    torch.save(initial, checkpoint)
    first_forward = []

    def build(*args, **kwargs):
        assert args[2] is None, "Warm start must not download encoder weights."
        model = TinySeg()

        def capture(module, inputs):
            if not first_forward:
                first_forward.append({k: v.detach().clone() for k, v in module.state_dict().items()})

        model.register_forward_pre_hook(capture)
        return model

    monkeypatch.setattr(train, "build_model", build)
    args = make_args("warm", epochs=1, init_checkpoint=str(checkpoint))
    train.run(args, 7)
    assert first_forward
    assert all(torch.equal(initial[k], first_forward[0][k]) for k in initial)
    last = torch.load(root / "warm/seed7/last.pt", weights_only=True)
    assert last["epoch"] == 0 and len(last["log"]) == 1
    assert all(state["step"].item() == 4 for state in last["optimizer"]["state"].values())

    assert set(torch.load(root / "warm/seed7/best.pt", weights_only=True)) == set(initial)
    assert set(torch.load(root / "warm/seed7/final.pt", weights_only=True)) == set(initial)


def test_grouped_validation_predicts_once_and_scores_each_annotation(cpu_training, monkeypatch):
    _, frame, _ = cpu_training
    mask = train.load_instance_mask(frame.iloc[2].mask_path)
    predict = Mock(return_value=(mask > 0).astype(np.float32))
    monkeypatch.setattr(train, "predict_prob", predict)
    params = dict(low=0.5, high=0.8, close_px=0, min_area=1, watershed=False, ws_min_dist=15)
    result, _, _ = train.validate(TinySeg(), frame.iloc[2:], torch.device("cpu"),
                                  params=params, tile_size=32, overlap=8, blend="hann")
    assert predict.call_count == 1
    assert result["TP"] == 2 and result["PQ"] == 1.0
    assert predict.call_args.kwargs["blend"] == "hann"


@pytest.mark.parametrize("flag", ["--samples-per-epoch", "--steps-per-epoch", "--steps_per_epoch"])
def test_sample_budget_aliases_mean_samples(flag):
    args = train.make_parser().parse_args(["--manifest", "manifest.csv", "--out", "unused", flag, "32"])
    assert args.steps_per_epoch == 32
    assert args.bs == 8


def test_existing_run_is_protected_unless_overwrite_is_explicit(cpu_training):
    root, _, make_args = cpu_training
    args = make_args("protected", epochs=1)
    output = root / "protected/seed7"
    output.mkdir(parents=True)
    artifact = output / "best.pt"
    artifact.write_bytes(b"existing model")
    note = output / "notes.txt"
    note.write_text("Keep user-authored files.", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already contains"):
        train.run(args, 7)
    assert artifact.read_bytes() == b"existing model"
    args.overwrite = True
    train.run(args, 7)
    assert torch.load(artifact, weights_only=True)
    assert note.read_text(encoding="utf-8") == "Keep user-authored files."


def test_external_resume_cannot_overwrite_an_unrelated_run(cpu_training):
    root, _, make_args = cpu_training
    source = root / "external.pt"
    source.write_bytes(b"checkpoint content is irrelevant before the protection check")
    output = root / "protected/seed7"
    output.mkdir(parents=True)
    note = output / "notes.txt"
    note.write_text("existing run", encoding="utf-8")
    args = make_args("protected", resume=str(source))
    with pytest.raises(FileExistsError, match="empty output directory"):
        train.run(args, 7)
    assert note.read_text(encoding="utf-8") == "existing run"
