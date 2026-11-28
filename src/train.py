"""Train and select checkpoints on the official validation split."""
import platform
import random
import subprocess
import time
from importlib.metadata import version
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.artifacts import (checkpoint_record, dataset_metadata, identity, image_weighting, objectives,
                           read_json, run_directory, sha256, should_stop, training_config,
                           update_selection, validate_checkpoint, write_json)
from src.config import CLASSES, MAX_TEXT_LEN, ROOT
from src.data import Collator, CrisisDataset, load_split
from src.metrics import evaluate_outputs
from src.models import build_model, param_groups
from src.precision import autocast_dtype, optimizer_step

AGREE_WEIGHT = 0.5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}


def compute_loss(out, b, smoothing=0.0, image_weights=None):
    loss = 0.0
    if "text" in out:
        loss = loss + F.cross_entropy(out["text"].float(), b["text_label"], label_smoothing=smoothing)
    if "image" in out:
        loss = loss + F.cross_entropy(out["image"].float(), b["image_label"], weight=image_weights,
                                      label_smoothing=smoothing)
    if "agree" in out:
        loss = loss + AGREE_WEIGHT * F.binary_cross_entropy_with_logits(out["agree"].float(), b["agree"])
    return loss


@torch.no_grad()
def predict(model, loader, device, dtype):
    model.eval()
    outs, tgts, total, n = {}, {}, 0.0, 0
    for b in loader:
        b = to_device(b, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            out = model(b)
        total += compute_loss(out, b).item() * len(b["agree"])
        n += len(b["agree"])
        for k, v in out.items():
            outs.setdefault(k, []).append(v.float().cpu().numpy())
        for k in ("text_label", "image_label", "agree"):
            tgts.setdefault(k, []).append(b[k].cpu().numpy())
    outs = {k: np.concatenate(v) for k, v in outs.items()}
    tgts = {k: np.concatenate(v) for k, v in tgts.items()}
    return outs, tgts, total / n


def swap_captions(b, p, generator):
    """Swap augmentation: give a fraction p of the batch another post's caption."""
    n = len(b["agree"])
    chosen = torch.nonzero(torch.rand(n, generator=generator) < p).flatten()
    if len(chosen) < 2:
        return b
    donor = chosen[torch.randperm(len(chosen), generator=generator)]
    b = dict(b)
    for k in ("input_ids", "attention_mask", "text_label", "has_text"):
        b[k] = b[k].clone()
        b[k][chosen] = b[k][donor]
    b["agree"] = (b["text_label"] == b["image_label"]).float()
    return b


def selection_score(res):
    heads = [res["all"][h]["macro_f1"] for h in ("text", "image") if h in res["all"]]
    return float(np.mean(heads))


def make_loader(df, model_kind, tokenizer, batch_size, train, workers, generator=None):
    ds = CrisisDataset(df, train=train, use_text=model_kind != "image",
                       use_image=model_kind != "text")
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers,
                      collate_fn=Collator(tokenizer, MAX_TEXT_LEN), generator=generator,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def plot_curve(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(hist["epoch"], hist["train_loss"], marker="o", label="train loss")
    ax1.plot(hist["epoch"], hist["val_loss"], marker="o", label="validation loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["val_macro_f1"], color="tab:green", marker="s", ls="--",
             label="validation macro-F1")
    ax2.set_ylabel("macro-F1")
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8, loc="center right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_checkpoint(value, path):
    temp = path.with_suffix(".pt.tmp")
    torch.save(value, temp)
    temp.replace(path)


def provenance(device, dtype):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        commit = git("rev-parse", "HEAD")
        changed = git("status", "--porcelain", "--", "src")
    except FileNotFoundError:
        commit, changed = None, None
    return {
        "git_commit": commit, "source_dirty": bool(changed) if changed is not None else None,
        "source_sha256": {p.name: sha256(p) for p in sorted((ROOT / "src").glob("*.py"))},
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in
                     ("torch", "torchvision", "transformers", "numpy", "pandas", "scikit-learn")},
        "device": str(device), "cuda": torch.version.cuda,
        "autocast_dtype": str(dtype) if dtype is not None else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def main():
    config, resume = training_config()
    args = SimpleNamespace(**config)
    run_dir = run_directory(config)
    if (run_dir / "config.json").exists():
        raise SystemExit(f"Completed or damaged run at {run_dir}. Preserve it and use a new --tag.")
    if not resume and run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}. Use --resume or a new --tag.")

    dataset = {split: dataset_metadata(split) for split in ("train", "val")}
    set_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    origin = provenance(device, dtype)
    if not resume:
        run_dir.mkdir(parents=True)
        write_json(run_dir / "started.json", {"training_config": config, "dataset": dataset,
                                             "provenance": origin})
    else:
        started = read_json(run_dir / "started.json")
        if (started["training_config"] != config or started["dataset"] != dataset or
                started["provenance"] != origin):
            raise SystemExit(f"Cannot resume with changed configuration, data or environment: {run_dir}")
        if not (run_dir / "last.pt").is_file():
            raise SystemExit(f"No resumable checkpoint in {run_dir}. Use a new --tag.")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    train_df = load_split("train", args.quick, args.seed)
    val_df = load_split("val", args.quick, args.seed)
    weighting = image_weighting(train_df["image_label"]) if config.get("image_class_weighting") else None
    image_weights = (torch.tensor([weighting["weights"][name] for name in CLASSES],
                                  device=device, dtype=torch.float32) if weighting else None)
    if weighting:
        print(f"Image classification weighting: {weighting}")
    gen = torch.Generator().manual_seed(args.seed)
    train_dl = make_loader(train_df, args.model, tokenizer, args.batch_size, True, args.workers, gen)
    val_dl = make_loader(val_df, args.model, tokenizer, args.batch_size * 2, False, args.workers)

    model = build_model(args.model, args.text_model, not args.no_pretrained,
                        args.modality_dropout, args.encoder_dropout, args.fusion_dropout).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(param_groups(model, args.lr_text, args.lr_image, args.lr_head,
                                         args.weight_decay))
    steps = args.epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    print(f"{run_dir.name}: {n_params / 1e6:.1f} M params | device {device} | precision {dtype or 'FP32'} | "
          f"train {len(train_df)} | val {len(val_df)}")

    hist, start = [], 1
    selected = dict.fromkeys(objectives(config))
    if resume:
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if ck["config_sha256"] != identity(config):
            raise SystemExit(f"Resume checkpoint configuration mismatch: {run_dir}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if scaler is not None:
            scaler.load_state_dict(ck["scaler"])
        hist, selected, start = ck["hist"], ck["selected"], ck["epoch"] + 1
        for record in selected.values():
            if record is not None:
                validate_checkpoint(run_dir, record)
        torch.set_rng_state(ck["rng"].cpu())
        gen.set_state(ck["generator"].cpu())
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in ck["cuda_rng"]])
        print(f"Resumed after epoch {ck['epoch']}")

    for epoch in range(start, args.epochs + 1):
        if should_stop(epoch - 1, selected, args.patience):
            break
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for b in train_dl:
            if args.model == "multimodal" and args.swap_aug > 0:
                b = swap_captions(b, args.swap_aug, gen)
            b = to_device(b, device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                out = model(b)
            loss = compute_loss(out, b, args.label_smoothing, image_weights=image_weights)
            optimizer_step(loss, opt, sched, model.parameters(), scaler)
            total += loss.item() * len(b["agree"])
            n += len(b["agree"])
        outs, tgts, val_loss = predict(model, val_dl, device, dtype)
        res = evaluate_outputs(outs, tgts)
        score = selection_score(res)
        warning = res["all"].get("warning", {})
        candidates = {"classification": ("macro_f1", score, 1e-4),
                      "warning": ("conflict_ap", warning.get("conflict_ap"), 0.0)}
        row = {"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
               "val_macro_f1": score, "seconds": round(time.time() - t0)}
        for head in ("text", "image"):
            row[f"val_{head}_macro_f1"] = res["all"].get(head, {}).get("macro_f1")
        for metric in ("f1", "precision", "recall", "ap", "auroc"):
            row[f"val_conflict_{metric}"] = warning.get(f"conflict_{metric}")

        for objective, previous in selected.items():
            metric, value, delta = candidates[objective]
            winner = update_selection(previous, epoch, value, delta)
            improved = winner is not previous
            if improved:
                path = run_dir / f"best_{objective}_epoch{epoch}.pt"
                save_checkpoint(model.state_dict(), path)
                selected[objective] = checkpoint_record(path, epoch, metric, value)
            record = selected[objective]
            row[f"{objective}_improved"] = improved
            row[f"{objective}_checkpoint_epoch"] = record["epoch"] if record else None
            row[f"{objective}_checkpoint_file"] = record["file"] if record else None
        hist.append(row)
        print(f"epoch {epoch} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val macro-F1 {score:.4f} | warning AP {warning.get('conflict_ap')}")
        save_checkpoint({
            "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
            "hist": hist, "selected": selected, "epoch": epoch, "config_sha256": identity(config),
            "rng": torch.get_rng_state(), "generator": gen.get_state(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "scaler": scaler.state_dict() if scaler is not None else None,
        }, run_dir / "last.pt")

    history = pd.DataFrame(hist)
    history_path = run_dir / "history.csv"
    history.to_csv(history_path, index=False)
    plot_curve(history, run_dir / "learning_curve.png")
    if selected["classification"] is None:
        raise SystemExit("No finite validation classification score. No completed run was written.")
    manifest = {
        **config, "training_config": config, "config_sha256": identity(config),
        "n_params": n_params, "dataset": dataset, "provenance": origin,
        "history_sha256": sha256(history_path),
        "pretrained": {
            "text_revision": getattr(model.text_enc.bert.config, "_commit_hash", None)
            if hasattr(model, "text_enc") else None,
            "image_weights": "MobileNet_V3_Large_Weights.IMAGENET1K_V2"
            if hasattr(model, "image_enc") and not args.no_pretrained else None,
        },
        "best_epoch": selected["classification"]["epoch"],
        "best_val_macro_f1": selected["classification"]["score"],
        "warning_status": ("selected" if selected.get("warning") else
                           "undefined_on_clean_validation" if "warning" in selected else "not_applicable"),
        "checkpoints": {**selected, "last": checkpoint_record(
            run_dir / "last.pt", hist[-1]["epoch"], None, None)},
    }
    if weighting:
        manifest["image_class_weighting_details"] = weighting
    write_json(run_dir / "config.json", manifest)
    print(f"Saved {run_dir}")


if __name__ == "__main__":
    main()
