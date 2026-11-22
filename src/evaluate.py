"""Step 3: score a trained run under every stress-test condition.

Run from the project root:
    python -m src.evaluate --run multimodal_seed42                 # validation split
    python -m src.evaluate --run multimodal_seed42 --split test    # once, at the end

Conditions: clean, text_missing, image_missing, text_swapped, image_swapped (see data.py).
Writes runs/<run>/eval_<split>.json and pred_<split>_<condition>.npz.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoTokenizer

from src.config import RUNS_DIR
from src.data import CONDITIONS, apply_condition, load_split
from src.metrics import evaluate_outputs
from src.models import build_model
from src.train import make_loader, predict


def evaluate_run(run, split, conditions=CONDITIONS, workers=2):
    run_dir = RUNS_DIR / run
    cfg = json.loads((run_dir / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_model"])
    model = build_model(cfg["model"], cfg["text_model"], not cfg["no_pretrained"],
                        cfg["modality_dropout"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    base = load_split(split, cfg["quick"], cfg["seed"])

    results = {}
    for cond in conditions:
        df = apply_condition(base, cond, seed=cfg["seed"])
        loader = make_loader(df, cfg["model"], tokenizer, 64, False, workers)
        outs, tgts, _ = predict(model, loader, device, amp)
        results[cond] = evaluate_outputs(outs, tgts)
        np.savez_compressed(run_dir / f"pred_{split}_{cond}.npz", **outs,
                            **{f"y_{k}": v for k, v in tgts.items()})
        r = results[cond]["all"]
        msg = " | ".join(f"{h} macro-F1 {r[h]['macro_f1']:.4f}" for h in ("text", "image") if h in r)
        if "warning" in r:
            msg += f" | conflict-F1 {r['warning']['conflict_f1']:.4f}"
        print(f"  {cond:14s} {msg}")
    (run_dir / f"eval_{split}.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    print(f"Evaluating {args.run} on {args.split}")
    evaluate_run(args.run, args.split, workers=args.workers)
    print(f"Saved runs/{args.run}/eval_{args.split}.json")


if __name__ == "__main__":
    main()
