"""Train/evaluate an explicit experiment matrix, or evaluate it on test."""
import argparse
import subprocess
import sys

from src.artifacts import (SEEDS, dataset_metadata, evaluation_complete, evaluation_directory,
                           run_directory, training_config, validate_training)
from src.config import EXPERIMENT, ROOT

FUSION = [("multimodal", 0.0, 0.0), ("multimodal", 0.0, 0.25)]
EXPERIMENTS = [("text", 0.0, 0.0), ("image", 0.0, 0.0),
               *FUSION, ("multimodal", 0.3, 0.0)]


def step(module, args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", choices=["v2", "v3"], default=EXPERIMENT)
    p.add_argument("--matrix", choices=["full", "fusion"], default="full")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--tag", default="main")
    p.add_argument("--workers", type=int)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume incomplete training runs")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="training overrides, e.g. --extra --batch-size 16")
    a = p.parse_args(argv)
    reserved = {"--model", "--modality-dropout", "--swap-aug", "--seed", "--recipe", "--tag", "--quick"}
    if any(arg.split("=")[0] in reserved for arg in a.extra):
        p.error("Use the matrix and launcher options to set model, recipe, seed, tag or quick mode")
    if a.split == "test" and (a.quick or a.resume or "--quick" in a.extra or "--resume" in a.extra):
        p.error("Final test requires completed full runs")
    seeds = a.seeds or (SEEDS[:1] if a.quick else SEEDS)
    experiments = FUSION if a.matrix == "fusion" else EXPERIMENTS
    extra = [part for part in a.extra if part != "--resume"]
    workers = ["--workers", str(a.workers)] if a.workers is not None else []
    jobs = []
    for model, md, sw in experiments:
        for seed in seeds:
            args = [*extra, "--model", model, "--modality-dropout", str(md), "--swap-aug", str(sw),
                    "--seed", str(seed), "--recipe", a.recipe, "--tag", a.tag,
                    *workers, *(["--quick"] if a.quick else [])]
            config, _ = training_config(args)
            jobs.append((config, run_directory(config), args))

    try:
        manifests = {}
        if a.split == "test":
            # Preflight every run before any model evaluation. This branch never trains.
            for config, directory, _ in jobs:
                manifests[directory] = validate_training(directory, config, check_last=False)
            dataset_metadata("test")
        else:
            try:
                dataset = {s: dataset_metadata(s) for s in ("train", "val")}
            except FileNotFoundError:
                step("src.prepare_data", ["--splits", "train", "val"])
                dataset = {s: dataset_metadata(s) for s in ("train", "val")}
            for config, directory, args in jobs:
                if (directory / "config.json").exists():
                    manifest = validate_training(directory, config)
                    if manifest["dataset"] != dataset:
                        raise ValueError(f"Training data changed for {directory}. Use a new --tag.")
                    print(f"[skip] verified training artifacts in {directory}")
                    manifests[directory] = manifest
                    continue
                if (a.resume or "--resume" in a.extra) and directory.exists():
                    args = [*args, "--resume"]
                step("src.train", args)
                manifests[directory] = validate_training(directory, config)

        dataset = dataset_metadata(a.split)
        for config, directory, _ in jobs:
            manifest = manifests[directory]
            if not all(evaluation_complete(evaluation_directory(directory, objective), a.split,
                                           config, checkpoint, dataset)
                       for objective, checkpoint in manifest["checkpoints"].items()
                       if objective != "last" and checkpoint is not None):
                step("src.evaluate", ["--run-dir", directory, "--split", a.split, "--workers", config["workers"]])
        step("src.report", ["--split", a.split, *[arg for _, directory, _ in jobs
                                                  for arg in ("--run-dir", directory)]])
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"Pipeline blocked: {exc}\n")


if __name__ == "__main__":
    main()
