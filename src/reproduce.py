"""One command that reproduces every headline number.

Run from the project root, after downloading the data (see README):
    python -m src.reproduce --quick          # short check that every step runs
    python -m src.reproduce                  # train all runs, evaluate on validation
    python -m src.reproduce --split test     # same runs (already trained) evaluated on test: once, at the end

Steps (finished steps are skipped; --force reruns them):
    1. prepare_data
    2. train text / image / multimodal / + modality dropout 0.3 / + swap augmentation 0.25, seeds 42-44
    3. evaluate every run under all stress-test conditions
    4. report
"""
import argparse
import subprocess
import sys
import time

from src.config import PROCESSED_DIR, ROOT, RUNS_DIR
from src.report import SEEDS

# (model, modality dropout, swap augmentation)
EXPERIMENTS = [("text", 0.0, 0.0), ("image", 0.0, 0.0), ("multimodal", 0.0, 0.0),
               ("multimodal", 0.3, 0.0), ("multimodal", 0.0, 0.25)]


def step(title, module, args=(), done=None, force=False):
    if done is not None and done.exists() and not force:
        print(f"[skip] {title}")
        return
    print(f"\n=== {title} ===", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True)
    print(f"--- {title}: {time.time() - t0:.0f} s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="extra arguments passed to src.train (e.g. --extra --workers 0)")
    a = p.parse_args()
    from src.train import run_name
    seeds = SEEDS[:1] if a.quick else SEEDS
    q = ["--quick"] if a.quick else []

    step("1. Prepare data", "src.prepare_data", done=PROCESSED_DIR / "test.csv", force=a.force)
    for model, md, sw in EXPERIMENTS:
        for seed in seeds:
            run = run_name(model, md, seed, a.quick, sw)
            step(f"2. Train {run}", "src.train",
                 ["--model", model, "--modality-dropout", str(md), "--swap-aug", str(sw),
                  "--seed", str(seed), *q, *a.extra],
                 done=RUNS_DIR / run / "config.json", force=a.force)
            step(f"3. Evaluate {run} on {a.split}", "src.evaluate", ["--run", run, "--split", a.split],
                 done=RUNS_DIR / run / f"eval_{a.split}.json", force=a.force)
    step("4. Report", "src.report", ["--split", a.split, *q], force=True)
    print("\nDone. Tables: runs/v2/summary_*.md | figures: report/figures/v2/")


if __name__ == "__main__":
    main()
