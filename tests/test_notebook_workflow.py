import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from infer import parser as inference_parser
from train import make_parser as training_parser


ROOT = Path(__file__).resolve().parents[1]


def notebook_cells(filename="COS711_A2.ipynb"):
    notebook = json.loads((ROOT / filename).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def execute(cells, marker, state):
    selected = [source for source in cells if marker in source]
    assert len(selected) == 1, f"Cell marker is not unique: {marker}"
    exec(compile(selected[0], f"notebook: {marker}", "exec"), state)


def run_stage(cells, stage, state):
    markers = {
        "train": 'run_script("train.py", args)',
        "calibrate": 'run_script("infer.py", inference_arguments() + [',
        "infer": "export_args = inference_arguments()",
    }
    execute(cells, markers[stage], state)


def synthetic_official_data(root):
    train = root / "train" / "train_images"
    test = root / "test" / "test_images"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    images, annotations = [], []
    for number in range(40):
        name = f"20110101{number:06d}Bh.jpeg"

        (train / name).write_bytes(b"synthetic image placeholder")
        images.append(dict(id=f"annotator-{number}", file_name=name, width=8, height=8))
        annotations.append(dict(id=number, image_id=f"annotator-{number}",
                                segmentation=[[1, 1, 5, 1, 5, 5, 1, 5]]))
    (root / "train" / "official_annotations.json").write_text(
        json.dumps(dict(images=images, annotations=annotations)), encoding="utf-8")
    (test / "20120101000000Bh.jpeg").write_bytes(b"synthetic image placeholder")


@pytest.fixture
def prepared_notebook(tmp_path, monkeypatch, request):
    cells = notebook_cells(getattr(request, "param", "COS711_A2.ipynb"))
    state = {"__name__": "__main__"}
    execute(cells, "EXPERIMENT = ", state)
    state.update(EXPECTED_TEST_IMAGES=1, RUN_MODE="calibrate", CHECKPOINT_SEEDS=[1])
    local_data = tmp_path / "MAGFiLO_1.0_Kaggle_2026"
    synthetic_official_data(local_data)
    original_inputs = {str(path.relative_to(local_data)): path.read_bytes()
                       for path in local_data.rglob("*") if path.is_file()}
    baseline = tmp_path / "runs" / "base" / "seed1" / "best.pt"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"a synthetic checkpoint; it is never loaded")
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Mock GPU")
    monkeypatch.setattr(sys, "path", list(sys.path))



    module_names = ["checkpoint_io", "data", "models", "losses", "pq_metric", "postprocess", "train", "infer",
                    "submission_check", "compress", "timing_probe"]
    for name in module_names:
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    stale_data = SimpleNamespace(__file__="old/notebook/data.py")
    sys.modules["data"] = stale_data

    original_run = subprocess.run
    calls = []

    def intercept(command, **kwargs):
        calls.append((list(command), kwargs))
        if Path(command[2]).name == "data.py":
            return original_run(command, **kwargs)
        if command[1:3] == ["-m", "pip"]:
            raise AssertionError("Tests must not install dependencies")
        if Path(command[2]).name == "train.py":
            arguments = training_parser().parse_args(command[3:])
            for seed in arguments.seed:
                checkpoint = Path(arguments.out) / f"seed{seed}" / "best.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"synthetic trained checkpoint")
        if Path(command[2]).name == "infer.py" and "--evaluate-only" in command:
            destination = Path(command[command.index("--params-out") + 1])
            destination.write_text(json.dumps(dict(
                postprocessing={"low": .4}, tuning={"PQ": .3},
                validation={"metrics": {"PQ": .28}, "default_postprocessing_metrics": {"PQ": .30}},
                full_validation={"metrics": {"PQ": .29}}
            )), encoding="utf-8")
        if Path(command[2]).name == "infer.py" and "--evaluate-only" not in command:
            arguments = inference_parser().parse_args(command[3:])
            destination = Path(arguments.out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("filament_id,segmentation_rle\n20120101000000Bh_1,synthetic_counts\n", encoding="utf-8")
            destination.with_suffix(".metadata.json").write_text(json.dumps(dict(
                output=str(destination), checkpoints=arguments.ckpt,
                used_tuned_postprocessing=not arguments.use_default_postprocessing,
                validation={"metrics": {"PQ": .30 if arguments.use_default_postprocessing else .28}}
            )), encoding="utf-8")
        if Path(command[2]).name == "compress.py":
            destination = Path(command[command.index("--out") + 1])
            destination.write_text("variant,val_PQ\nsynthetic,0.3\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", intercept)
    execute(cells, "import importlib.util", state)
    execute(cells, "EMBEDDED_SOURCES =", state)
    assert "data" not in sys.modules
    execute(cells, "if DATA_ROOT:", state)
    assert Path(sys.modules["data"].__file__).resolve().parent == state["CODE_DIR"]
    assert state["COMPETITION_ROOT"] == local_data
    assert len(state["train_rows"]) and len(state["val_rows"])
    assert set(state["train_rows"].obs_id).isdisjoint(state["val_rows"].obs_id)
    assert all(Path(path).is_relative_to(state["WORK"]) for path in state["frame"].mask_path if path)
    assert original_inputs == {str(path.relative_to(local_data)): path.read_bytes()
                               for path in local_data.rglob("*") if path.is_file()}
    return cells, state, calls, baseline


def inference_call(calls):
    return next(command for command, _ in reversed(calls) if Path(command[2]).name == "infer.py")


def test_calibration_discovers_baseline_and_builds_valid_cli(prepared_notebook, capsys):
    cells, state, calls, baseline = prepared_notebook
    execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    assert state["ACTIVE_CHECKPOINTS"] == [baseline]
    run_stage(cells, "calibrate", state)
    command = inference_call(calls)
    args = inference_parser().parse_args(command[3:])
    assert args.evaluate_only and args.tta == 1 and args.blend == "hann"
    assert args.tile_size == 512 and args.overlap == 128
    assert args.tune_samples == 24 and args.ckpt == [str(baseline)]
    assert Path(args.params_out) == state["EXP_DIR"] / "calibration.json"
    assert state["calibration_report"]["full_validation"]["metrics"]["PQ"] == .29
    assert "Set USE_TUNED_POSTPROCESSING=False" in capsys.readouterr().out


def test_training_builds_valid_cli_without_starting_training(prepared_notebook, capsys):
    cells, state, calls, _ = prepared_notebook
    state["RUN_MODE"] = "train"
    execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    assert state["ACTIVE_CHECKPOINTS"] == []
    run_stage(cells, "train", state)
    command = next(command for command, _ in calls if Path(command[2]).name == "train.py")
    args = training_parser().parse_args(command[3:])
    assert args.epochs == 60 and args.steps_per_epoch == 3584
    assert args.bs == 4 and args.cldice == 0 and args.sampling == "annotation"
    assert args.seed == [0] and args.split_seed == 0
    checkpoint = Path(args.out) / "seed0" / "best.pt"
    assert checkpoint.is_file()
    assert repr(str(checkpoint)) in capsys.readouterr().out


def test_default_recipe_trains_seed_zero_without_existing_weights(prepared_notebook):
    cells, state, calls, baseline = prepared_notebook
    execute(cells, "EXPERIMENT = ", state)
    assert state["RUN_MODE"] == "all"
    assert state["EXPERIMENT"] == "r34_bcedice_v2"
    assert state["TRAIN_SEEDS"] == [0] and state["CHECKPOINT_SEEDS"] == [0]
    assert state["CHECKPOINTS"] == []
    baseline.unlink()
    execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    assert state["ACTIVE_CHECKPOINTS"] == []
    run_stage(cells, "train", state)
    command = next(command for command, _ in calls if Path(command[2]).name == "train.py")
    args = training_parser().parse_args(command[3:])
    assert args.seed == [0] and args.epochs == 60 and args.steps_per_epoch == 3584


def test_missing_calibration_weights_explains_training_or_explicit_paths(prepared_notebook):
    cells, state, _, baseline = prepared_notebook
    baseline.unlink()
    with pytest.raises(ValueError) as failure:
        execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    message = str(failure.value)
    assert "RUN_MODE='train'" in message or 'RUN_MODE="train"' in message
    assert "CHECKPOINTS" in message and "Set CHECKPOINTS explicitly" in message


def test_missing_explicit_checkpoint_lists_available_weights_without_substitution(prepared_notebook):
    cells, state, calls, _ = prepared_notebook
    missing = state["WORK"] / "previous_session" / "seed0" / "best.pt"
    available = state["WORK"] / "saved_training" / "seed0" / "best.pt"
    available.parent.mkdir(parents=True)
    available.write_bytes(b"synthetic relocated weights")
    state["CHECKPOINTS"] = [str(missing)]
    with pytest.raises(FileNotFoundError) as failure:
        execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    message = str(failure.value)
    assert "Available best.pt" in message and str(available) in message
    assert "CHECKPOINTS" in message and "saved training outputs" in message
    assert state["ACTIVE_CHECKPOINTS"] == [missing]
    assert not any(Path(command[2]).name == "train.py" for command, _ in calls)


def test_training_success_without_expected_weights_reports_incomplete_output(prepared_notebook, monkeypatch):
    cells, state, _, _ = prepared_notebook
    state["RUN_MODE"] = "train"
    execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    with pytest.raises(RuntimeError) as failure:
        run_stage(cells, "train", state)
    message = str(failure.value)
    assert "best.pt" in message and "training" in message.lower()


def test_inference_reuses_attached_calibration_and_checks_submission(prepared_notebook, tmp_path):
    cells, state, calls, baseline = prepared_notebook
    calibration = tmp_path / "attached_saved_outputs" / "calibration.json"
    calibration.parent.mkdir()
    calibration.write_text(json.dumps({"postprocessing": {}}), encoding="utf-8")
    state.update(RUN_MODE="infer", CALIBRATION_PATH=str(calibration), CHECKPOINTS=[str(baseline)])
    execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    run_stage(cells, "infer", state)
    args = inference_parser().parse_args(inference_call(calls)[3:])
    assert args.params == str(calibration)
    assert args.skip_tune and args.skip_validation and not args.evaluate_only
    assert Path(args.out) == state["EXP_DIR"] / "submission.csv"
    assert Path(calls[-1][0][2]).name == "submission_check.py"


def test_inference_can_export_audit_defaults(prepared_notebook):
    cells, state, calls, _ = prepared_notebook
    state.update(RUN_MODE="infer", USE_TUNED_POSTPROCESSING=False)
    execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    state["CALIBRATION"].write_text("{}", encoding="utf-8")
    run_stage(cells, "infer", state)
    assert "--use-default-postprocessing" in inference_call(calls)


def test_ambiguous_baseline_checkpoints_require_explicit_selection(prepared_notebook, tmp_path):
    cells, state, _, _ = prepared_notebook
    duplicate = tmp_path / "runs" / "copied" / "base" / "seed1" / "best.pt"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"different synthetic weights")
    with pytest.raises(ValueError, match="Set CHECKPOINTS explicitly"):
        execute(cells, "ACTIVE_CHECKPOINTS = []", state)


def test_kaggle_discovers_nested_official_layout_and_attached_baseline(prepared_notebook, tmp_path):
    cells, state, _, _ = prepared_notebook
    attached_inputs = tmp_path / "kaggle_input"
    official_root = attached_inputs / "competitions" / "filament-segmentation-2026" / "MAGFiLO"
    shutil.copytree(state["COMPETITION_ROOT"], official_root)
    checkpoint = attached_inputs / "own_weights" / "base" / "seed1" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic attached baseline")

    real_path = Path
    state["Path"] = lambda value: attached_inputs if str(value) == "/kaggle/input" else real_path(value)
    state["ON_KAGGLE"] = True
    execute(cells, "if DATA_ROOT:", state)
    assert state["COMPETITION_ROOT"] == official_root
    execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    assert state["ACTIVE_CHECKPOINTS"] == [checkpoint]


def test_saved_model_notebook_preset_needs_no_calibration_attachment():
    state = {}
    execute(notebook_cells("COS711_A2_FROM_BEST.ipynb"), "EXPERIMENT = ", state)
    assert state["RUN_MODE"] == "submit"
    assert state["EXPERIMENT"] == "r34_bcedice_v2_seed0_hann_tta"
    assert state["CHECKPOINTS"] == [] and state["CALIBRATION_PATH"] == ""
    assert state["RUN_COMPRESSION"] is False


def test_seed42_preset_trains_a_new_seed_on_the_original_validation_split():
    state = {}
    execute(notebook_cells("COS711_A2_SEED42.ipynb"), "EXPERIMENT = ", state)
    assert state["RUN_MODE"] == "all"
    assert state["EXPERIMENT"] == "r34_bcedice_v2_seed42"
    assert state["TRAIN_SEEDS"] == [42] and state["SPLIT_SEED"] == 0
    assert state["CHECKPOINTS"] == [] and state["CALIBRATION_PATH"] == ""
    assert state["INIT_CHECKPOINT"] == "" and state["RESUME_CHECKPOINT"] == ""


def automatic_run(cells, state):
    execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    if state["RUN_MODE"] == "all":
        run_stage(cells, "train", state)
    run_stage(cells, "calibrate", state)
    run_stage(cells, "infer", state)


def script_names(calls):
    return [Path(command[2]).name for command, _ in calls]


def test_all_mode_trains_calibrates_exports_and_records_root_artifacts(prepared_notebook):
    cells, state, calls, _ = prepared_notebook
    state.update(RUN_MODE="all", CHECKPOINTS=[])
    automatic_run(cells, state)
    names = script_names(calls)
    assert names[-4:] == ["train.py", "infer.py", "infer.py", "submission_check.py"]
    trained = state["TRAIN_OUT"] / "seed0" / "best.pt"
    assert state["ACTIVE_CHECKPOINTS"] == [trained]
    exports = [command for command, _ in calls if Path(command[2]).name == "infer.py"]
    calibration = inference_parser().parse_args(exports[0][3:])
    prediction = inference_parser().parse_args(exports[1][3:])
    assert calibration.ckpt == prediction.ckpt == [str(trained)]
    assert calibration.evaluate_only and not prediction.evaluate_only
    assert prediction.params == calibration.params_out
    assert prediction.skip_tune and prediction.skip_validation
    assert prediction.use_default_postprocessing
    assert state["USE_TUNED_POSTPROCESSING"] is False
    assert (state["WORK"] / "submission.csv").read_bytes() == state["OUTPUT_CSV"].read_bytes()
    metadata = json.loads((state["WORK"] / "submission.metadata.json").read_text())
    assert metadata["output"] == str(state["WORK"] / "submission.csv")
    assert metadata["used_tuned_postprocessing"] is False
    original = json.loads(state["OUTPUT_CSV"].with_suffix(".metadata.json").read_text())
    assert original["output"] == str(state["OUTPUT_CSV"])
    assert (state["EXP_DIR"] / "run_summary.json").is_file()


@pytest.mark.parametrize("prepared_notebook", ["COS711_A2_SEED42.ipynb"], indirect=True)
def test_seed42_preset_runs_training_through_final_csv_without_mode_changes(prepared_notebook):
    cells, state, calls, _ = prepared_notebook
    state.update(RUN_MODE="all", CHECKPOINTS=[])
    automatic_run(cells, state)
    train_calls = [command for command, _ in calls if Path(command[2]).name == "train.py"]
    assert len(train_calls) == 1
    training = training_parser().parse_args(train_calls[0][3:])
    assert training.seed == [42] and training.split_seed == 0
    expected = state["WORK"] / "trained_models" / "r34_bcedice_v2_seed42" / "seed42" / "best.pt"
    assert expected.is_file()
    assert state["ACTIVE_CHECKPOINTS"] == [expected]
    assert script_names(calls)[-4:] == ["train.py", "infer.py", "infer.py", "submission_check.py"]
    for command, _ in calls:
        if Path(command[2]).name == "infer.py":
            args = inference_parser().parse_args(command[3:])
            assert args.ckpt == [str(expected)] and args.split_seed == 0
    assert (state["WORK"] / "submission.csv").read_bytes() == state["OUTPUT_CSV"].read_bytes()


def test_submit_mode_reuses_unique_flat_checkpoint_and_never_trains(prepared_notebook):
    cells, state, calls, baseline = prepared_notebook
    uploaded = baseline.parents[2] / "uploaded_weights" / "best.pt"
    uploaded.parent.mkdir()
    baseline.rename(uploaded)
    state.update(RUN_MODE="submit", CHECKPOINTS=[], CALIBRATION_PATH="")
    automatic_run(cells, state)
    assert state["ACTIVE_CHECKPOINTS"] == [uploaded]
    assert "train.py" not in script_names(calls)
    assert script_names(calls)[-3:] == ["infer.py", "infer.py", "submission_check.py"]
    assert (state["WORK"] / "submission.csv").is_file()
    assert state["CALIBRATION"].is_file()


def test_all_mode_uses_trained_seed_order_for_probability_ensemble(prepared_notebook):
    cells, state, calls, _ = prepared_notebook
    state.update(RUN_MODE="all", CHECKPOINTS=[], TRAIN_SEEDS=[0, 42])
    automatic_run(cells, state)
    expected = [str(state["TRAIN_OUT"] / f"seed{seed}" / "best.pt") for seed in [0, 42]]
    for command, _ in calls:
        if Path(command[2]).name == "infer.py":
            assert inference_parser().parse_args(command[3:]).ckpt == expected


def test_submit_mode_will_not_guess_between_several_models(prepared_notebook):
    cells, state, calls, baseline = prepared_notebook
    duplicate = baseline.parents[2] / "other" / "best.pt"
    duplicate.parent.mkdir()
    duplicate.write_bytes(b"another checkpoint")
    state.update(RUN_MODE="submit", CHECKPOINTS=[])
    with pytest.raises(ValueError, match="CHECKPOINTS"):
        execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    assert "train.py" not in script_names(calls)


def test_explicit_model_archive_requires_extraction(prepared_notebook):
    cells, state, _, baseline = prepared_notebook
    archive = baseline.with_suffix(".pt.zip")
    archive.write_bytes(b"PK outer archive")
    state.update(RUN_MODE="submit", CHECKPOINTS=[str(archive)])
    with pytest.raises(ValueError, match="best.pt"):
        execute(cells, "ACTIVE_CHECKPOINTS = []", state)


def test_pytorch_checkpoint_zip_container_is_not_rejected_as_download_archive(prepared_notebook):
    cells, state, _, baseline = prepared_notebook
    baseline.write_bytes(b"PK\x03\x04 tensor serialization")
    state.update(RUN_MODE="submit", CHECKPOINTS=[str(baseline)])
    execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    assert state["ACTIVE_CHECKPOINTS"] == [baseline]


def test_all_mode_rejects_attached_checkpoint_configuration_instead_of_retraining(prepared_notebook):
    cells, state, calls, baseline = prepared_notebook
    state.update(RUN_MODE="all", CHECKPOINTS=[str(baseline)])
    with pytest.raises(ValueError, match="submit"):
        execute(cells, "import importlib.util", state)
        execute(cells, "ACTIVE_CHECKPOINTS = []", state)
    assert "train.py" not in script_names(calls)


@pytest.mark.parametrize("mode", ["all", "submit"])
def test_automatic_modes_do_not_reuse_an_external_calibration(prepared_notebook, mode):
    cells, state, _, baseline = prepared_notebook
    state.update(RUN_MODE=mode, CALIBRATION_PATH="previous_calibration.json",
                 CHECKPOINTS=[str(baseline)] if mode == "submit" else [])
    with pytest.raises(ValueError, match="CALIBRATION_PATH"):
        execute(cells, "import importlib.util", state)
        execute(cells, "ACTIVE_CHECKPOINTS = []", state)


def test_optional_compression_follows_verified_csv_export(prepared_notebook, monkeypatch):
    cells, state, calls, baseline = prepared_notebook
    state.update(RUN_MODE="submit", CHECKPOINTS=[str(baseline)], RUN_COMPRESSION=True)
    original_run = subprocess.run

    def check_export_first(command, **kwargs):
        if Path(command[2]).name == "compress.py":
            assert (state["WORK"] / "submission.csv").is_file()
            assert (state["WORK"] / "submission.metadata.json").is_file()
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", check_export_first)
    automatic_run(cells, state)
    if "compress.py" not in script_names(calls):
        execute(cells, 'run_script("compress.py"', state)
    assert script_names(calls)[-1] == "compress.py"


def test_automatic_submission_uses_tuning_when_its_audit_is_better(prepared_notebook, monkeypatch):
    cells, state, calls, baseline = prepared_notebook
    state.update(RUN_MODE="submit", CHECKPOINTS=[str(baseline)], USE_TUNED_POSTPROCESSING=False)
    original_run = subprocess.run

    def improve_tuned_audit(command, **kwargs):
        result = original_run(command, **kwargs)
        if Path(command[2]).name == "infer.py" and "--evaluate-only" in command:
            report_path = Path(command[command.index("--params-out") + 1])
            report = json.loads(report_path.read_text())
            report["validation"]["metrics"]["PQ"] = .42
            report_path.write_text(json.dumps(report), encoding="utf-8")
        return result

    monkeypatch.setattr(subprocess, "run", improve_tuned_audit)
    automatic_run(cells, state)
    assert state["USE_TUNED_POSTPROCESSING"] is True
    assert not inference_parser().parse_args(inference_call(calls)[3:]).use_default_postprocessing
    metadata = json.loads((state["WORK"] / "submission.metadata.json").read_text())
    assert metadata["used_tuned_postprocessing"] is True


def test_failed_submission_check_does_not_publish_root_csv(prepared_notebook, monkeypatch):
    cells, state, _, baseline = prepared_notebook
    state.update(RUN_MODE="submit", CHECKPOINTS=[str(baseline)])
    original_run = subprocess.run

    def fail_submission_check(command, **kwargs):
        if Path(command[2]).name == "submission_check.py":
            raise subprocess.CalledProcessError(1, command)
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_submission_check)
    with pytest.raises(subprocess.CalledProcessError):
        automatic_run(cells, state)
    assert not (state["WORK"] / "submission.csv").exists()


@pytest.mark.parametrize("failure,exception", [("interrupted", OSError), ("truncated", RuntimeError)])
def test_failed_root_copy_preserves_previous_submission_and_cleans_stage(
        prepared_notebook, monkeypatch, failure, exception):
    cells, state, _, baseline = prepared_notebook
    state.update(RUN_MODE="submit", CHECKPOINTS=[str(baseline)])
    root_csv = state["WORK"] / "submission.csv"
    previous = b"previous complete submission\n"
    root_csv.write_bytes(previous)
    original_copy = shutil.copyfile
    staged = []

    def incomplete_copy(source, destination, *args, **kwargs):
        target = Path(destination)
        if target.parent == state["WORK"] and target.name.startswith(".submission."):
            staged.append(target)
            target.write_bytes(Path(source).read_bytes()[:12])
            if failure == "interrupted":
                raise OSError("Simulated interrupted root submission copy")
            return destination
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", incomplete_copy)
    with pytest.raises(exception):
        automatic_run(cells, state)
    assert len(staged) == 1
    assert root_csv.read_bytes() == previous
    assert not staged[0].exists()
    assert not list(state["WORK"].glob(".submission.*"))


@pytest.mark.parametrize("explicit", [False, True])
def test_submit_restores_unpacked_weights_before_calibration(prepared_notebook, tmp_path, explicit):
    import zipfile

    cells, state, calls, baseline = prepared_notebook
    checkpoint = tmp_path / "original_weights.pt"
    weights = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    torch.save(weights, checkpoint)
    extraction = tmp_path / "extracted_checkpoint"
    with zipfile.ZipFile(checkpoint) as archive:
        archive.extractall(extraction)
    baseline.unlink()
    shutil.copytree(next(extraction.iterdir()), baseline)
    state.update(RUN_MODE="submit", CHECKPOINTS=[str(baseline)] if explicit else [])
    automatic_run(cells, state)
    restored = state["ACTIVE_CHECKPOINTS"][0]
    assert restored.is_file() and restored.is_relative_to(state["WORK"] / "checkpoint_cache")
    assert baseline.is_dir() and (baseline / "data.pkl").is_file()
    assert torch.equal(torch.load(restored, map_location="cpu", weights_only=True)["weight"], weights["weight"])
    assert "train.py" not in script_names(calls)
    assert inference_parser().parse_args(inference_call(calls)[3:]).ckpt == [str(restored)]
    assert (state["WORK"] / "submission.csv").is_file()
