"""Evaluate frozen classification and warning checkpoints."""
import argparse
from pathlib import Path

from src.artifacts import (CONDITIONS, dataset_metadata, evaluation_complete, evaluation_directory,
                           identity, objectives, read_json, sha256, validate_checkpoint, validate_training,
                           write_json)
from src.precision import autocast_dtype


def evaluate_run(run_dir, split, objective, config, checkpoint, dataset, workers, revision=None):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from src.data import apply_condition, load_split
    from src.metrics import evaluate_outputs
    from src.models import build_model
    from src.train import make_loader, predict

    directory = evaluation_directory(run_dir, objective)
    directory.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = autocast_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(config["text_model"], revision=revision)
    model = build_model(config["model"], config["text_model"], False, config["modality_dropout"],
                        config["encoder_dropout"], config["fusion_dropout"], text_pretrained=False,
                        text_revision=revision).to(device)
    path = validate_checkpoint(run_dir, checkpoint)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    base = load_split(split, config["quick"], config["seed"])
    predictions = {}
    results = {}
    for condition in CONDITIONS:
        df = apply_condition(base, condition, seed=config["seed"])
        loader = make_loader(df, config["model"], tokenizer, 64, False, workers)
        outs, tgts, _ = predict(model, loader, device, dtype)
        results[condition] = evaluate_outputs(outs, tgts)
        path = directory / f"pred_{split}_{condition}.npz"
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **outs, **{f"y_{k}": v for k, v in tgts.items()},
                            tweet_id=df["tweet_id"].to_numpy(dtype=str),
                            image_id=df["image_id"].to_numpy(dtype=str))
        temp.replace(path)
        predictions[condition] = sha256(path)
    metrics_path = directory / f"eval_{split}.json"
    write_json(metrics_path, results)
    write_json(directory / f"evaluation_{split}.json", {
        "checkpoint": checkpoint, "dataset": dataset, "config_sha256": identity(config),
        "metrics_sha256": sha256(metrics_path), "predictions": predictions,
        "device": str(device), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "autocast_dtype": str(dtype) if dtype is not None else None,
    })
    print(f"Saved {objective} evaluation from epoch {checkpoint['epoch']} in {directory}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--checkpoint", choices=["all", "classification", "warning"], default="all")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)
    try:
        manifest = validate_training(args.run_dir, check_last=False)
        config = manifest["training_config"]
        if args.split == "test" and config["quick"]:
            raise ValueError("Quick runs are for validation checks only")
        choices = objectives(config) if args.checkpoint == "all" else (args.checkpoint,)
        for objective in choices:
            if manifest["checkpoints"].get(objective) is None and args.checkpoint != "all":
                raise ValueError(f"No {objective} checkpoint was selected in {args.run_dir}")
        dataset = dataset_metadata(args.split)
        if args.split == "val" and dataset != manifest["dataset"]["val"]:
            raise ValueError("Processed validation data differ from checkpoint selection data")
        for objective in choices:
            checkpoint = manifest["checkpoints"].get(objective)
            if checkpoint is None:
                print(f"[skip] {objective} checkpoint unavailable for {args.run_dir}")
                continue
            directory = evaluation_directory(args.run_dir, objective)
            marker = directory / f"evaluation_{args.split}.json"
            if marker.exists():
                try:
                    previous = read_json(marker)
                except ValueError:
                    previous = {}
                if any(key in previous and previous[key] != value for key, value in
                       (("checkpoint", checkpoint), ("dataset", dataset), ("config_sha256", identity(config)))):
                    raise ValueError(f"Evaluation identity changed in {directory}. Preserve the existing run.")
            if evaluation_complete(directory, args.split, config, checkpoint, dataset):
                print(f"[skip] {objective} evaluation already complete in {directory}")
                continue
            evaluate_run(args.run_dir, args.split, objective, config, checkpoint, dataset, args.workers,
                         manifest.get("pretrained", {}).get("text_revision"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        p.exit(1, f"Evaluation blocked: {exc}\n")


if __name__ == "__main__":
    main()
