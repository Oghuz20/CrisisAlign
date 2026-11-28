"""Run identity and completion checks shared by the pipeline commands."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from src.config import (CLASSES, ENCODER_DROPOUT, EXPERIMENT, FUSION_DROPOUT, PROCESSED_DIR, ROOT,
                        TEXT_MODEL)

CONDITIONS = ("clean", "text_missing", "image_missing", "text_swapped", "image_swapped")
SEEDS = (42, 43, 44)
RUN_ROOT = ROOT / "runs" / "disagreement"
RECIPES = {
    "v2": {"encoder_dropout": None, "fusion_dropout": 0.0, "weight_decay": 0.01},
    "v3": {"encoder_dropout": ENCODER_DROPOUT, "fusion_dropout": FUSION_DROPOUT, "weight_decay": 0.05},
}


def training_config(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["text", "image", "multimodal"])
    p.add_argument("--recipe", choices=RECIPES, default=EXPERIMENT)
    p.add_argument("--tag", default="main", help="new tag for a separate run of the same configuration")
    p.add_argument("--modality-dropout", type=float, default=0.0)
    p.add_argument("--swap-aug", type=float, default=0.0)
    p.add_argument("--image-class-weighting", choices=["capped-sqrt"],
                   help="opt in to mild train-derived weighting for image classification")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-text", type=float, default=2e-5)
    p.add_argument("--lr-image", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--text-model", default=TEXT_MODEL)
    p.add_argument("--no-pretrained", action="store_true", help="random image weights (debug only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        p.error("--tag must contain only letters, numbers, underscores or hyphens")
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.workers < 0:
        p.error("epochs, patience and batch size must be positive; workers must be nonnegative")
    for name in ("modality_dropout", "swap_aug", "label_smoothing"):
        if not 0 <= getattr(args, name) <= 1:
            p.error(f"{name} must be between 0 and 1")
    run = vars(args).copy()
    run.pop("resume")
    if args.image_class_weighting is None:
        run.pop("image_class_weighting")  # preserve baseline fingerprints and manifests
    elif args.model == "text":
        p.error("--image-class-weighting requires an image classification head")
    recipe = RECIPES[args.recipe]
    run.update(encoder_dropout=recipe["encoder_dropout"], fusion_dropout=recipe["fusion_dropout"],
               weight_decay=recipe["weight_decay"] if args.weight_decay is None else args.weight_decay,
               epochs=min(args.epochs, 2) if args.quick else args.epochs)
    try:
        identity(run)
    except ValueError:
        p.error("numeric settings must be finite")
    return run, args.resume


def identity(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def image_weighting(labels):
    """Capped inverse-square-root frequency relative to mean class support."""
    observed = Counter(labels)
    counts = {name: observed[name] for name in CLASSES}
    if set(observed) != set(CLASSES):
        raise ValueError("Image weighting requires positive train support for every configured class")
    reference = sum(counts.values()) / len(counts)
    weights = {name: min(1.5, max(0.75, math.sqrt(reference / n))) for name, n in counts.items()}
    return {"method": "capped-sqrt", "counts": counts, "weights": weights}


def model_name(config):
    name = config["model"]
    if name == "multimodal":
        for key, label in (("modality_dropout", "md"), ("swap_aug", "swap")):
            if config[key]:
                name += f"_{label}{config[key]:g}"
    return name


def run_directory(config):
    name = f"{model_name(config)}_seed{config['seed']}"
    return RUN_ROOT / config["recipe"] / f"{name}-{identity(config)[:16]}"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_metadata(split):
    path = PROCESSED_DIR / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        fields = {"split", "event_name", "tweet_id", "image_id", "image_path", "text",
                  "text_label", "image_label", "agree"}
        if not fields <= set(rows.fieldnames or ()) or next(rows, {}).get("split") != split:
            raise ValueError(f"Invalid or empty processed {split} split: {path}")
        for row in rows:
            if row["split"] != split:
                raise ValueError(f"Unexpected split marker at CSV line {rows.line_num} in {path}")
    return {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def objectives(config):
    return ("classification", "warning") if config["model"] == "multimodal" else ("classification",)


def checkpoint_record(path, epoch, metric, score):
    return {"file": path.name, "epoch": epoch, "metric": metric, "score": score, "sha256": sha256(path)}


def validate_checkpoint(run_dir, record):
    file = record["file"]
    if Path(file).name != file or not file.endswith(".pt"):
        raise ValueError(f"Invalid checkpoint filename in {run_dir}")
    path = run_dir / file
    if not path.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint {path}")
    try:
        with ZipFile(path) as archive:
            if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                raise ValueError(f"Invalid PyTorch checkpoint structure: {path}")
    except BadZipFile as exc:
        raise ValueError(f"Corrupt PyTorch checkpoint: {path}") from exc
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    return path


def validate_training(run_dir, expected=None, *, check_last=True):
    manifest_path = run_dir / "config.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed training manifest {manifest_path}")
    manifest = read_json(manifest_path)
    if "training_config" not in manifest:
        raise ValueError(f"No current training manifest in {run_dir}. Use --archive for historical reports.")
    config = manifest["training_config"]
    if (manifest["config_sha256"] != identity(config) or
            any(manifest.get(k) != v for k, v in config.items()) or
            (expected is not None and config != expected)):
        raise ValueError(f"Stale or mismatched configuration: {run_dir}")
    history_path = run_dir / "history.csv"
    if not history_path.is_file() or sha256(history_path) != manifest["history_sha256"]:
        raise ValueError(f"Missing or mismatched training history: {run_dir}")
    with history_path.open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    selection = manifest["checkpoints"]
    if [int(row["epoch"]) for row in history] != list(range(1, selection["last"]["epoch"] + 1)):
        raise ValueError(f"History does not match the final epoch in {run_dir}")
    if "warning" in objectives(config) and selection["warning"] is None:
        if (manifest["warning_status"] != "undefined_on_clean_validation" or
                any(row["val_conflict_ap"] for row in history)):
            raise ValueError(f"Missing warning checkpoint metadata: {run_dir}")
    records = {"classification": "macro_f1"}
    if check_last:
        records["last"] = None
    if "warning" in objectives(config) and selection["warning"] is not None:
        records["warning"] = "conflict_ap"
    for objective, metric in records.items():
        record = selection[objective]
        validate_checkpoint(run_dir, record)
        if (type(record["epoch"]) is not int or
                not 1 <= record["epoch"] <= selection["last"]["epoch"] <= config["epochs"] or
                record["metric"] != metric):
            raise ValueError(f"Invalid checkpoint metadata in {run_dir}")
        if metric is not None and (not 0 <= record["score"] <= 1 or
                                   float(history[record["epoch"] - 1][f"val_{metric}"]) != record["score"]):
            raise ValueError(f"Invalid selection score in {run_dir}")
    if selection.get("warning") and selection["warning"]["file"] == selection["classification"]["file"]:
        raise ValueError(f"Expected separate classification and warning files in {run_dir}")
    if manifest["best_epoch"] != selection["classification"]["epoch"]:
        raise ValueError(f"Classification epoch differs from checkpoint metadata in {run_dir}")
    return manifest


def evaluation_directory(run_dir, objective):
    return run_dir if objective == "classification" else run_dir / "warning"


def evaluation_complete(directory, split, config, checkpoint, dataset):
    """Require the evaluation manifest, metrics and each matching prediction archive."""
    try:
        import numpy as np

        meta = read_json(directory / f"evaluation_{split}.json")
        if (meta["checkpoint"] != checkpoint or meta["dataset"] != dataset or
                meta["config_sha256"] != identity(config)):
            return False
        metrics_path = directory / f"eval_{split}.json"
        metrics = read_json(metrics_path)
        if set(metrics) != set(CONDITIONS) or sha256(metrics_path) != meta["metrics_sha256"]:
            return False
        heads = {"text", "image", "agree"} if config["model"] == "multimodal" else {config["model"]}
        arrays = heads | {"y_agree", "y_text_label", "y_image_label", "tweet_id", "image_id"}
        required = {h: {"macro_f1", "accuracy", "critical_recall", "ece", "per_class_f1"}
                    for h in heads - {"agree"}}
        if "agree" in heads:
            required["warning"] = {"conflict_f1", "conflict_precision", "conflict_recall", "conflict_ap"}
        for condition in CONDITIONS:
            outputs = metrics[condition]["all"]
            if set(outputs) != set(required) or any(not keys <= set(outputs[h]) for h, keys in required.items()):
                return False
            path = directory / f"pred_{split}_{condition}.npz"
            if sha256(path) != meta["predictions"][condition]:
                return False
            with np.load(path, allow_pickle=False) as archive:
                if not arrays <= set(archive.files):
                    return False
                count = len(archive["y_agree"])
                if not count or any(archive[key].shape !=
                                    ((count, len(CLASSES)) if key in {"text", "image"} else (count,))
                                    for key in arrays):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return False


def update_selection(previous, epoch, score, min_delta=0.0):
    """Return the winning epoch and score. Exact ties retain the earlier epoch."""
    if score is None or not math.isfinite(score):
        return previous
    if previous is None or score > previous["score"] + min_delta:
        return {"epoch": epoch, "score": score}
    return previous


def should_stop(epoch, selected, patience):
    """Each active objective gets its own patience window."""
    return all(best is not None and epoch - best["epoch"] >= patience for best in selected.values())
