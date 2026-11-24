"""Step 2: train one model.

Run from the project root:
    python -m src.train --model text
    python -m src.train --model image
    python -m src.train --model multimodal
    python -m src.train --model multimodal --modality-dropout 0.3     # robustness remedy
    python -m src.train --model multimodal --swap-aug 0.25            # conflict-warning remedy
    python -m src.train --model multimodal --quick                    # pipeline check (CPU, minutes)
    python -m src.train --model multimodal --resume                   # continue an interrupted run

Validation = official dev split. Model selection: mean macro-F1 of the label
heads on validation; early stopping after --patience epochs without improvement.

Against overfitting: 5 epochs max, patience 2, label smoothing 0.1, head dropout 0.3
(v2); encoder dropout 0.2, fusion dropout 0.2, weight decay 0.05 (v3). Swap augmentation (multimodal only): in a fraction of each
training batch the caption is exchanged with another post's caption, and the
targets follow (text label of the new caption; conflict if labels differ). The
model can then only spot these conflicts by comparing caption and image.
Writes runs/<run>/: history.csv, learning_curve.png, best.pt, last.pt, config.json
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.config import MAX_TEXT_LEN, RUNS_DIR, TEXT_MODEL
from src.data import Collator, CrisisDataset, load_split
from src.metrics import evaluate_outputs
from src.models import build_model, param_groups

AGREE_WEIGHT = 0.5
LABEL_SMOOTHING = 0.1


def run_name(model, modality_dropout, seed, quick, swap_aug=0.0):
    md = f"_md{modality_dropout:g}" if model == "multimodal" and modality_dropout > 0 else ""
    sw = f"_swap{swap_aug:g}" if model == "multimodal" and swap_aug > 0 else ""
    return f"{model}{md}{sw}_seed{seed}{'_quick' if quick else ''}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(b, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}


def compute_loss(out, b, smoothing=0.0):
    loss = 0.0
    if "text" in out:
        loss = loss + F.cross_entropy(out["text"].float(), b["text_label"], label_smoothing=smoothing)
    if "image" in out:
        loss = loss + F.cross_entropy(out["image"].float(), b["image_label"], label_smoothing=smoothing)
    if "agree" in out:
        loss = loss + AGREE_WEIGHT * F.binary_cross_entropy_with_logits(out["agree"].float(), b["agree"])
    return loss


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    outs, tgts, total, n = {}, {}, 0.0, 0
    for b in loader:
        b = to_device(b, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["text", "image", "multimodal"])
    p.add_argument("--modality-dropout", type=float, default=0.0)
    p.add_argument("--swap-aug", type=float, default=0.0,
                   help="fraction of each training batch whose caption is swapped (multimodal)")
    p.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-text", type=float, default=2e-5)
    p.add_argument("--lr-image", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--text-model", default=TEXT_MODEL)
    p.add_argument("--no-pretrained", action="store_true", help="random image weights (debug only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true", help="small subset, 2 epochs: pipeline check")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 2)

    set_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    amp = device.type == "cuda"
    run_dir = RUNS_DIR / run_name(args.model, args.modality_dropout, args.seed, args.quick,
                                  args.swap_aug)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    train_df = load_split("train", args.quick, args.seed)
    val_df = load_split("val", args.quick, args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    train_dl = make_loader(train_df, args.model, tokenizer, args.batch_size, True, args.workers, gen)
    val_dl = make_loader(val_df, args.model, tokenizer, args.batch_size * 2, False, args.workers)

    model = build_model(args.model, args.text_model, not args.no_pretrained,
                        args.modality_dropout).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(param_groups(model, args.lr_text, args.lr_image, args.lr_head,
                                         args.weight_decay))
    steps = args.epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    print(f"{run_dir.name}: {n_params / 1e6:.1f} M params | device {device} | "
          f"train {len(train_df)} | val {len(val_df)}")

    hist, start, best, bad = [], 1, -1.0, 0
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        hist, best, bad, start = ck["hist"], ck["best"], ck["bad"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng"])
        print(f"Resumed after epoch {ck['epoch']}")

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for b in train_dl:
            if args.model == "multimodal" and args.swap_aug > 0:
                b = swap_captions(b, args.swap_aug, gen)
            b = to_device(b, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(b)
            loss = compute_loss(out, b, args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b["agree"])
            n += len(b["agree"])
        outs, tgts, val_loss = predict(model, val_dl, device, amp)
        res = evaluate_outputs(outs, tgts)
        score = selection_score(res)
        improved = score > best + 1e-4
        if improved:
            best, bad = score, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
        row = {"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "val_macro_f1": score,
               "seconds": round(time.time() - t0)}
        if "warning" in res["all"]:
            row["val_conflict_f1"] = res["all"]["warning"]["conflict_f1"]
        hist.append(row)
        print(f"epoch {epoch} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val macro-F1 {score:.4f}" + (f" | conflict-F1 {row['val_conflict_f1']:.4f}"
                                             if "val_conflict_f1" in row else "")
              + f" | {row['seconds']} s" + ("  *best" if improved else ""))
        pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "hist": hist, "best": best, "bad": bad, "epoch": epoch,
                    "rng": torch.get_rng_state()}, run_dir / "last.pt")
        if bad >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    plot_curve(pd.DataFrame(hist), run_dir / "learning_curve.png")
    cfg = {**vars(args), "n_params": n_params, "best_val_macro_f1": best,
           "best_epoch": int(pd.DataFrame(hist).loc[pd.DataFrame(hist)["val_macro_f1"].idxmax(), "epoch"])}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Best validation macro-F1 {best:.4f} | saved {run_dir.relative_to(RUNS_DIR.parent)}")


if __name__ == "__main__":
    main()
