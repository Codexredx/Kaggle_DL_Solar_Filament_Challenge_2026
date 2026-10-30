import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "checkpoint_io.py", "data.py", "models.py", "losses.py", "pq_metric.py", "postprocess.py",
    "train.py", "infer.py", "submission_check.py", "compress.py", "timing_probe.py",
]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": dedent(text).strip().splitlines(True),
    }


def build():
    sources = {name: (ROOT / name).read_text(encoding="utf-8-sig") for name in MODULES}
    cells = [
        markdown('# Solar filament segmentation\n\nAttach the official competition data and enable GPU and Internet. Mode `all` trains, calibrates, and exports in one run. Mode `submit` uses an attached best.pt, recalibrates, and exports. See `KAGGLE_UPLOAD_GUIDE.md` for upload instructions.'),
        code("""
        RUN_MODE = "all"
        EXPERIMENT = "r34_bcedice_v2"
        DATA_ROOT = ""
        CALIBRATION_PATH = ""
        CHECKPOINTS = []
        CHECKPOINT_SEEDS = [0]

        MODEL = "smp"
        ENCODER = "resnet34"
        BASE = 16
        SPLIT_SEED = 0
        TTA = True
        TILE_SIZE = 512
        TILE_OVERLAP = 128
        BLEND = "hann"
        TUNE_OBSERVATIONS = 24
        USE_TUNED_POSTPROCESSING = True
        RUN_COMPRESSION = False
        EXPECTED_TEST_IMAGES = 180


        TRAIN_SEEDS = [0]
        EPOCHS = 60
        SAMPLES_PER_EPOCH = 3584
        BATCH_SIZE = 4
        CROP_SIZE = 512
        LEARNING_RATE = 3e-4
        CLDICE_WEIGHT = 0.0
        SAMPLING = "annotation"
        WORKERS = 2
        INIT_CHECKPOINT = ""
        RESUME_CHECKPOINT = ""
        """),
        code("""
        import importlib.util
        import subprocess
        import sys
        from pathlib import Path

        if RUN_MODE not in {"all", "submit", "train", "calibrate", "infer"}:
            raise ValueError("RUN_MODE must be all, submit, train, calibrate, or infer")
        if RUN_MODE == "all" and CHECKPOINTS:
            raise ValueError("Use RUN_MODE='submit' for existing CHECKPOINTS; clear CHECKPOINTS for fresh all-mode training.")
        if RUN_MODE in {"all", "submit"} and CALIBRATION_PATH:
            raise ValueError("Leave CALIBRATION_PATH blank in automatic modes; calibration is generated in this run.")
        if not EXPERIMENT or Path(EXPERIMENT).name != EXPERIMENT:
            raise ValueError("EXPERIMENT must be one folder name")
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("Select the project's .venv Jupyter kernel locally, or a Kaggle GPU runtime with PyTorch.")
        dependencies = {
            "numpy": "numpy>=1.26,<3", "pandas": "pandas>=2,<3",
            "cv2": "opencv-python-headless>=4.8,<5", "scipy": "scipy>=1.11,<2",
            "segmentation_models_pytorch": "segmentation-models-pytorch>=0.4,<1",
            "pycocotools": "pycocotools>=2.0.7,<3", "skimage": "scikit-image>=0.22,<1",
            "matplotlib": "matplotlib>=3.7,<4",
        }
        missing = [requirement for name, requirement in dependencies.items()
                   if importlib.util.find_spec(name) is None]
        if missing:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

        import json
        import shutil
        import numpy as np
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt

        if not torch.cuda.is_available():
            raise RuntimeError("Enable a GPU in Kaggle settings, or select your CUDA-enabled local environment.")
        print("GPU:", torch.cuda.get_device_name(0), "| PyTorch:", torch.__version__)
        ORIGINAL_CWD = Path.cwd().resolve()
        ON_KAGGLE = Path("/kaggle/input").is_dir()
        WORK = Path("/kaggle/working") if ON_KAGGLE else ORIGINAL_CWD / "runs" / "notebook_working"
        CODE_DIR = WORK / "filament_code"
        EXP_DIR = WORK / EXPERIMENT
        CODE_DIR.mkdir(parents=True, exist_ok=True)
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        print("Experiment outputs:", EXP_DIR)
        """),
        code(
            "EMBEDDED_SOURCES = " + repr(sources) + "\n"
            + "for filename, source in EMBEDDED_SOURCES.items():\n"
            + "    compile(source, filename, 'exec')\n"
            + "    (CODE_DIR / filename).write_text(source, encoding='utf-8')\n"
            + "sys.path.insert(0, str(CODE_DIR))\n"
            + "for filename in EMBEDDED_SOURCES:\n"
            + "    sys.modules.pop(Path(filename).stem, None)\n"
            + "print('Embedded modules:', ', '.join(EMBEDDED_SOURCES))\n"
        ),
        code("""
        if DATA_ROOT:
            candidates = [Path(DATA_ROOT).expanduser().resolve()]
        elif ON_KAGGLE:
            candidates = sorted({p.parent.parent for p in Path("/kaggle/input").rglob("train_images")
                                 if p.parent.name == "train"})
        else:
            candidates = [ORIGINAL_CWD / "MAGFiLO_1.0_Kaggle_2026"]
        candidates = [p for p in candidates if (p / "train" / "train_images").is_dir()
                      and (p / "test" / "test_images").is_dir()
                      and len(list((p / "train").glob("*.json"))) == 1]
        if len(candidates) != 1:
            raise ValueError(f"Set DATA_ROOT to the official competition data root. Candidates: {candidates}")
        COMPETITION_ROOT = candidates[0]
        MANIFEST = WORK / "official_manifest.csv"
        MASK_CACHE = WORK / "official_masks"
        subprocess.run([sys.executable, "-u", str(CODE_DIR / "data.py"), "build",
                        "--root", str(COMPETITION_ROOT), "--cache", str(MASK_CACHE),
                        "--out", str(MANIFEST)], check=True, cwd=str(CODE_DIR))
        frame = pd.read_csv(MANIFEST, keep_default_na=False)
        n_test = int((frame.split == "test").sum())
        if n_test != EXPECTED_TEST_IMAGES:
            raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} official test images; found {n_test}")
        from data import time_block_split
        train_rows, val_rows = time_block_split(frame[frame.split == "train"], seed=SPLIT_SEED)
        assert set(train_rows.obs_id).isdisjoint(val_rows.obs_id)
        print("Training:", len(train_rows), "annotation sets /", train_rows.obs_id.nunique(), "observations")
        print("Validation:", len(val_rows), "annotation sets /", val_rows.obs_id.nunique(), "observations")
        print("The same observation's annotator versions remain together in the split.")
        """),
        code("""
        from checkpoint_io import find_checkpoints, prepare_checkpoint

        ACTIVE_CHECKPOINTS = []
        if RUN_MODE not in {"train", "all"}:
            if CHECKPOINTS:
                ACTIVE_CHECKPOINTS = [Path(p).expanduser().resolve() for p in CHECKPOINTS]
            elif RUN_MODE == "submit":
                roots = [Path("/kaggle/input")] if ON_KAGGLE else [ORIGINAL_CWD / "runs"]
                found = find_checkpoints(roots)
                if len(found) != 1:
                    raise ValueError(
                        f"Found {len(found)} best.pt files: {found}. Attach the extracted weights "
                        "as a Dataset input, or set CHECKPOINTS explicitly to your intended model."
                    )
                ACTIVE_CHECKPOINTS = found
            else:
                search_roots = [Path("/kaggle/input")] if ON_KAGGLE else [ORIGINAL_CWD / "runs"]
                for seed in CHECKPOINT_SEEDS:
                    found = sorted({p.resolve() for p in find_checkpoints(search_roots)
                                    if p.parent.name == f"seed{seed}" and p.parent.parent.name == "base"})
                    if len(found) != 1:
                        raise ValueError(
                            f"RUN_MODE={RUN_MODE!r} needs trained weights. Found {len(found)} baseline "
                            f"seed{seed} checkpoints: {found}. Set CHECKPOINTS explicitly to an existing best.pt. "
                            "For a fresh model, set RUN_MODE='train' and choose TRAIN_SEEDS, "
                            "then rerun the notebook from the top."
                        )
                    ACTIVE_CHECKPOINTS.append(found[0])
            if not ACTIVE_CHECKPOINTS or len(set(ACTIVE_CHECKPOINTS)) != len(ACTIVE_CHECKPOINTS):
                raise ValueError("Supply at least one unique checkpoint")
            for index, checkpoint in enumerate(ACTIVE_CHECKPOINTS):
                if checkpoint.suffix.lower() == ".zip":
                    raise ValueError("Extract the downloaded ZIP and select best.pt inside it, not the outer archive.")
                if checkpoint.is_dir():
                    restored = prepare_checkpoint(checkpoint, WORK / "checkpoint_cache")
                    print("Restored unpacked checkpoint:", checkpoint, "->", restored)
                    ACTIVE_CHECKPOINTS[index] = checkpoint = restored
                if not checkpoint.is_file():
                    roots = [WORK, Path("/kaggle/input")] if ON_KAGGLE else [WORK, ORIGINAL_CWD / "runs"]
                    available = sorted({p.resolve() for root in roots if root.is_dir()
                                        for p in root.rglob("best.pt") if p.is_file()})
                    listing = "\\n".join(map(str, available)) or "None found in the current working files or attached inputs."
                    raise FileNotFoundError(
                        f"Checkpoint not found: {checkpoint}\\nAvailable best.pt files:\\n{listing}\\n"
                        "Set CHECKPOINTS to the exact path of your intended trained model. "
                        "In a new Kaggle session, attach the saved training outputs and use their input path. "
                        "If no weights were saved, finish or resume training first; calibration does not create them."
                    )
                with checkpoint.open("rb") as handle:
                    if handle.read(64).startswith(b"version https://git-lfs.github.com/spec"):
                        raise ValueError(f"Git LFS pointer found instead of model weights: {checkpoint}")
            print("Selected checkpoints:", *ACTIVE_CHECKPOINTS, sep="\\n  ")

        if CALIBRATION_PATH and RUN_MODE in {"calibrate", "all", "submit"}:
            raise ValueError("CALIBRATION_PATH reuses an existing report in infer mode; leave blank to calibrate.")
        CALIBRATION = (Path(CALIBRATION_PATH).expanduser().resolve()
                       if CALIBRATION_PATH and RUN_MODE == "infer" else EXP_DIR / "calibration.json")
        OUTPUT_CSV = EXP_DIR / "submission.csv"
        TRAIN_OUT = WORK / "trained_models" / EXPERIMENT

        def run_script(name, arguments):
            command = [sys.executable, "-u", str(CODE_DIR / name), *map(str, arguments)]
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True, cwd=str(CODE_DIR))

        def inference_arguments():
            return ["--manifest", MANIFEST, "--ckpt", *ACTIVE_CHECKPOINTS,
                    "--model", MODEL, "--encoder", ENCODER, "--base", BASE,
                    "--split-seed", SPLIT_SEED, "--tta", int(TTA), "--tile-size", TILE_SIZE,
                    "--overlap", TILE_OVERLAP, "--blend", BLEND,
                    "--cache-dir", WORK / "prob_cache"]
        """),
        markdown('## Training\n\nModes `all` and `train` train the selected seeds. In `all`, their verified best.pt files are passed directly to calibration and prediction. Use `submit` to reuse existing trained weights.'),
        code("""
        if RUN_MODE in {"train", "all"}:
            if INIT_CHECKPOINT and RESUME_CHECKPOINT:
                raise ValueError("Choose INIT_CHECKPOINT or RESUME_CHECKPOINT")
            if RESUME_CHECKPOINT and len(TRAIN_SEEDS) != 1:
                raise ValueError("Resume one seed at a time")
            args = ["--manifest", MANIFEST, "--out", TRAIN_OUT, "--model", MODEL,
                    "--encoder", ENCODER, "--base", BASE, "--epochs", EPOCHS,
                    "--samples-per-epoch", SAMPLES_PER_EPOCH, "--bs", BATCH_SIZE,
                    "--crop", CROP_SIZE, "--lr", LEARNING_RATE, "--cldice", CLDICE_WEIGHT,
                    "--val-every", 5, "--workers", WORKERS, "--sampling", SAMPLING,
                    "--split-seed", SPLIT_SEED, "--seed", *TRAIN_SEEDS]
            if INIT_CHECKPOINT:
                args += ["--init-checkpoint", Path(INIT_CHECKPOINT).expanduser().resolve(), "--pretrained", 0]
            if RESUME_CHECKPOINT:
                args += ["--resume", Path(RESUME_CHECKPOINT).expanduser().resolve()]
            run_script("train.py", args)
            trained_checkpoints = [TRAIN_OUT / f"seed{seed}" / "best.pt" for seed in TRAIN_SEEDS]
            missing_checkpoints = [str(path) for path in trained_checkpoints if not path.is_file()]
            if missing_checkpoints:
                raise RuntimeError(
                    f"Training returned without the expected best.pt files: {missing_checkpoints}. "
                    "Check the training output before switching to calibration."
                )
            if RUN_MODE == "all":
                ACTIVE_CHECKPOINTS = trained_checkpoints
                print("Training complete. Continuing with calibration:", *ACTIVE_CHECKPOINTS, sep="\\n  ")
            else:
                print("Next: set RUN_MODE='calibrate' and CHECKPOINTS to:")
                print([str(path) for path in trained_checkpoints])
            print("Save the complete seed folders before ending this session or starting a new execution.")
        """),
        markdown('## Calibration\n\nModes `all` and `submit` calibrate automatically and select tuned or default post-processing using audit PQ. Mode `calibrate` runs this stage separately.'),
        code("""
        if RUN_MODE in {"calibrate", "all", "submit"}:
            run_script("infer.py", inference_arguments() + [
                "--evaluate-only", "--tune-samples", TUNE_OBSERVATIONS, "--params-out", CALIBRATION])
            if RUN_MODE == "calibrate":
                print("Calibration complete. Next set RUN_MODE='infer' with the same checkpoint/settings "
                      "to generate test predictions at:", OUTPUT_CSV)
        if RUN_MODE in {"calibrate", "infer", "all", "submit"} and CALIBRATION.is_file():
            calibration_report = json.loads(CALIBRATION.read_text(encoding="utf-8"))
            for field in ("postprocessing", "tuning", "validation", "full_validation", "sampled_validation"):
                if field in calibration_report:
                    print(field + ":", json.dumps(calibration_report[field], indent=2))
            audit = calibration_report.get("validation", {})
            tuned_pq = audit.get("metrics", {}).get("PQ")
            default_pq = audit.get("default_postprocessing_metrics", {}).get("PQ")
            if RUN_MODE in {"all", "submit"}:
                if tuned_pq is None or default_pq is None:
                    raise RuntimeError("Automatic export requires the calibration audit comparison.")
                USE_TUNED_POSTPROCESSING = tuned_pq >= default_pq
                print("Selected post-processing:", "tuned" if USE_TUNED_POSTPROCESSING else "default",
                      "| tuned audit PQ:", tuned_pq, "| default audit PQ:", default_pq)
            elif tuned_pq is not None and default_pq is not None and tuned_pq < default_pq:
                print("Default post-processing scored higher on the audit observations. "
                      "Set USE_TUNED_POSTPROCESSING=False when exporting this recipe, "
                      "or improve calibration before choosing a submission.")
        """),
        code("""
        per_image_csv = CALIBRATION.with_suffix(".per_image.csv")
        if RUN_MODE in {"calibrate", "all", "submit"} and per_image_csv.is_file():
            from data import load_image, load_instance_mask
            from postprocess import prob_to_instances
            from pq_metric import PQAccumulator
            scores = pd.read_csv(per_image_csv)
            display(scores.sort_values("PQ").head(8))
            chosen = pd.concat([scores.sort_values("PQ").drop_duplicates("obs_id").head(2),
                                scores.sort_values("PQ", ascending=False).drop_duplicates("obs_id").head(2)])
            chosen = chosen.drop_duplicates(["obs_id", "mask_path"])
            from postprocess import DEFAULTS
            params = calibration_report["postprocessing"] if USE_TUNED_POSTPROCESSING else dict(DEFAULTS)
            fig, axes = plt.subplots(len(chosen), 3, figsize=(14, 4 * len(chosen)), squeeze=False)
            for axis, row in zip(axes, chosen.itertuples(index=False)):
                image = load_image(row.image_path)
                gt = load_instance_mask(row.mask_path)
                probability = np.load(row.prob_path)
                pred = prob_to_instances(probability, **params)
                displayed_score = PQAccumulator()
                displayed_score.update(gt, pred)
                axis[0].imshow(image, cmap="gray"); axis[0].set_title(row.obs_id)
                axis[1].imshow(image, cmap="gray")
                axis[1].imshow(np.ma.masked_where(gt == 0, gt), cmap="tab20", alpha=.65)
                axis[1].set_title("One annotator's instances")
                axis[2].imshow(image, cmap="gray")
                axis[2].imshow(np.ma.masked_where(pred == 0, pred), cmap="tab20", alpha=.65)
                axis[2].set_title(f"Prediction; local PQ={displayed_score.compute()['PQ']:.3f}")
                for item in axis:
                    item.axis("off")
            fig.tight_layout()
            fig.savefig(EXP_DIR / "validation_examples.png", dpi=120)
            plt.show()
        """),
        markdown('## Submission\n\nModes `all` and `submit` export automatically. Upload the final submission.csv from the working root to the competition. Mode `infer` exports separately using an existing calibration report.'),
        code("""
        if RUN_MODE in {"infer", "all", "submit"}:
            if not CALIBRATION.is_file():
                raise FileNotFoundError(f"Run calibration first with these settings: {CALIBRATION}")
            export_args = inference_arguments() + ["--params", CALIBRATION,
                          "--skip-tune", "--skip-validation", "--out", OUTPUT_CSV]
            if not USE_TUNED_POSTPROCESSING:
                export_args += ["--use-default-postprocessing"]
            run_script("infer.py", export_args)
            run_script("submission_check.py", ["--csv", OUTPUT_CSV, "--manifest", MANIFEST])
            print("Upload this CSV:", OUTPUT_CSV)
            if RUN_MODE in {"all", "submit"}:
                source_metadata = OUTPUT_CSV.with_suffix(".metadata.json")
                if not OUTPUT_CSV.is_file() or not source_metadata.is_file():
                    raise RuntimeError("Inference returned without the expected CSV and metadata files.")
                root_csv = WORK / "submission.csv"
                root_metadata = WORK / "submission.metadata.json"
                import tempfile
                with tempfile.NamedTemporaryFile(dir=WORK, prefix=".submission.", suffix=".csv", delete=False) as handle:
                    staged_csv = Path(handle.name)
                try:
                    shutil.copyfile(OUTPUT_CSV, staged_csv)
                    if staged_csv.read_bytes() != OUTPUT_CSV.read_bytes():
                        raise RuntimeError("Final CSV copy differs from the checked predictions")
                    staged_csv.replace(root_csv)
                finally:
                    staged_csv.unlink(missing_ok=True)
                export_metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
                export_metadata.update(output=str(root_csv), experiment_output=str(OUTPUT_CSV))
                root_metadata.write_text(json.dumps(export_metadata, indent=2), encoding="utf-8")
                summary = dict(mode=RUN_MODE, checkpoints=[str(p) for p in ACTIVE_CHECKPOINTS],
                               calibration=str(CALIBRATION), output=str(root_csv), metadata=str(root_metadata),
                               experiment_output=str(OUTPUT_CSV),
                               used_tuned_postprocessing=USE_TUNED_POSTPROCESSING,
                               validation=export_metadata.get("validation"))
                (EXP_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print("Pipeline complete. Upload this final CSV to the competition:", root_csv)
                if RUN_COMPRESSION:
                    run_script("compress.py", ["--manifest", MANIFEST, "--ckpt", ACTIVE_CHECKPOINTS[0],
                               "--model", MODEL, "--encoder", ENCODER, "--base", BASE,
                               "--split-seed", SPLIT_SEED, "--finetune", 0, "--val-limit", 24,
                               "--out", EXP_DIR / "compression_results.csv"])
        """),
        code("""
        for summary_path in sorted(TRAIN_OUT.glob("seed*/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            history = pd.DataFrame(summary["log"])
            fig, axes = plt.subplots(1, 2, figsize=(11, 3))
            axes[0].plot(history.epoch + 1, history.loss)
            axes[0].set(xlabel="Epoch", ylabel="Training loss", title=summary_path.parent.name)
            if "PQ" in history:
                evaluated = history.dropna(subset=["PQ"])
                axes[1].plot(evaluated.epoch + 1, evaluated.PQ, "o-")
            axes[1].set(xlabel="Epoch", ylabel="Default-postprocessing validation PQ")
            fig.tight_layout(); plt.show()
            print("Best validation PQ:", summary["best_val_PQ"], "| Training seconds:", summary["train_time_s"])
        """),
    ]
    plot_index = next(i for i, cell in enumerate(cells)
                      if cell["cell_type"] == "code" and "".join(cell["source"]).startswith("per_image_csv ="))
    cells.append(cells.pop(plot_index))
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"filament-{index:02d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
            if "".join(cell["source"]).startswith("EMBEDDED_SOURCES ="):
                cell["metadata"] = {"jupyter": {"source_hidden": True}, "collapsed": True}
    destination = ROOT / "COS711_A2.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {destination.name}: {len(cells)} cells, {len(sources)} embedded modules")
    saved_model_notebook = json.loads(json.dumps(notebook))
    for cell in saved_model_notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("RUN_MODE ="):
            source = source.replace('RUN_MODE = "all"', 'RUN_MODE = "submit"', 1)
            source = source.replace('EXPERIMENT = "r34_bcedice_v2"',
                                    'EXPERIMENT = "r34_bcedice_v2_seed0_hann_tta"', 1)
            cell["source"] = source.splitlines(True)
    destination = ROOT / "COS711_A2_FROM_BEST.ipynb"
    destination.write_text(json.dumps(saved_model_notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {destination.name}: existing best.pt to final submission in one run")
    seed42_notebook = json.loads(json.dumps(notebook))
    for cell in seed42_notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("RUN_MODE ="):
            source = source.replace('EXPERIMENT = "r34_bcedice_v2"',
                                    'EXPERIMENT = "r34_bcedice_v2_seed42"', 1)
            source = source.replace("TRAIN_SEEDS = [0]", "TRAIN_SEEDS = [42]", 1)
            source = source.replace("CHECKPOINT_SEEDS = [0]", "CHECKPOINT_SEEDS = [42]", 1)
            cell["source"] = source.splitlines(True)
    seed42_notebook["cells"][0] = markdown(
        '# Solar filament segmentation: seed 42\n\n'
        'Attach the official competition data and enable GPU and Internet. Run All trains a new '
        'seed-42 model, saves best.pt, calibrates, predicts, validates every mask, and exports '
        '/kaggle/working/submission.csv. The validation split remains fixed at seed 0. '
        'See KAGGLE_UPLOAD_GUIDE.md for details.')
    seed42_notebook["cells"][0]["id"] = "filament-00"
    destination = ROOT / "COS711_A2_SEED42.ipynb"
    destination.write_text(json.dumps(seed42_notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {destination.name}: train seed 42 through final submission in one run")


if __name__ == "__main__":
    build()
