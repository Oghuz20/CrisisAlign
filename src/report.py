"""Step 4: collect all runs into the final tables and figures.

Run from the project root:
    python -m src.report                 # validation
    python -m src.report --split test    # once, at the end
    python -m src.report --quick         # summarise the *_quick pipeline-check runs

Writes
    runs/v3/summary_<split>.md / .csv  clean results + stress tests (mean ± std over seeds)
    report/figures/v3/stress_test_<split>.png
    report/figures/v3/learning_curves.png
The "pair of unimodal models" warning baseline flags a conflict whenever the
text-only and image-only models predict different classes.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.config import EXPERIMENT, FIG_DIR, RUNS_DIR
from src.data import CONDITIONS
from src.metrics import agreement_metrics, softmax

SEEDS = (42, 43, 44)
MODELS = ("text", "image", "multimodal", "multimodal_md0.3", "multimodal_swap0.25")


def run_id(model, seed, quick):
    return f"{model}_seed{seed}{'_quick' if quick else ''}"


def collect(split, quick):
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            path = RUNS_DIR / run_id(model, seed, quick) / f"eval_{split}.json"
            if not path.exists():
                continue
            res = json.loads(path.read_text())
            for cond in CONDITIONS:
                if cond not in res:
                    continue
                for subset, r in res[cond].items():
                    for h in ("text", "image"):
                        if h in r:
                            rows.append({"model": model, "seed": seed, "condition": cond,
                                         "subset": subset, "output": h,
                                         "macro_f1": r[h]["macro_f1"],
                                         "critical_recall": r[h]["critical_recall"],
                                         "ece": r[h]["ece"]})
                    if "warning" in r:
                        rows.append({"model": model, "seed": seed, "condition": cond,
                                     "subset": subset, "output": "warning",
                                     "macro_f1": r["warning"]["conflict_f1"],
                                     "critical_recall": r["warning"]["conflict_recall"],
                                     "ece": np.nan,
                                     "auroc": r["warning"].get("conflict_auroc", np.nan)})
    return pd.DataFrame(rows)


def warning_baselines(split, quick):
    """Two reference points for the conflict warning (clean input):
    - "always conflict": flags every post; its F1 is high only because conflicts are common;
    - "text+image pair": flags a conflict when separate text-only and image-only models
      predict different classes (soft score = 1 - sum_c p_text(c) * p_image(c)).
    """
    rows = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            t = RUNS_DIR / run_id("text", seed, quick) / f"pred_{split}_{cond}.npz"
            v = RUNS_DIR / run_id("image", seed, quick) / f"pred_{split}_{cond}.npz"
            if not (t.exists() and v.exists()):
                continue
            a, b = np.load(t), np.load(v)
            y = a["y_agree"]
            pt, pv = softmax(a["text"]), softmax(b["image"])
            pair = agreement_metrics(1 - (pt * pv).sum(1), y)
            pair_hard = agreement_metrics((pt.argmax(1) != pv.argmax(1)).astype(float), y)
            entries = [("text+image pair", pair_hard, pair.get("conflict_auroc", np.nan))]
            if cond == "clean":
                entries.append(("always conflict", agreement_metrics(np.ones(len(y)), y), 0.5))
            for name, m, auroc in entries:
                rows.append({"model": name, "seed": seed, "condition": cond, "subset": "all",
                             "output": "warning", "macro_f1": m["conflict_f1"],
                             "critical_recall": m["conflict_recall"], "ece": np.nan,
                             "auroc": auroc})
    return pd.DataFrame(rows)


def fmt(g):
    m, s, n = g.mean(), g.std(), g.count()
    return f"{m:.4f} ± {s:.4f}" if n > 1 else f"{m:.4f}"


def plot_stress(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[(df["subset"] == "all") & (df["output"].isin(["text", "image"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, out in zip(axes, ("text", "image")):
        g = d[d["output"] == out].groupby(["condition", "model"])["macro_f1"].mean().unstack()
        g = g.reindex([c for c in CONDITIONS if c in g.index])
        g.plot.bar(ax=ax, rot=20)
        ax.set_title(f"{out} label")
        ax.set_ylabel("macro-F1")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(quick, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 3), sharey=True)
    for ax, model in zip(axes, MODELS):
        h = RUNS_DIR / run_id(model, SEEDS[0], quick) / "history.csv"
        ax.set_title(model, fontsize=9)
        if h.exists():
            h = pd.read_csv(h)
            ax.plot(h["epoch"], h["train_loss"], marker="o", ms=3, label="train")
            ax.plot(h["epoch"], h["val_loss"], marker="o", ms=3, label="validation")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    df = pd.concat([collect(args.split, args.quick), warning_baselines(args.split, args.quick)],
                   ignore_index=True)
    if df.empty:
        raise SystemExit("No evaluation results found: run training and evaluation first.")
    sfx = f"{args.split}{'_quick' if args.quick else ''}"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUNS_DIR / f"summary_{sfx}.csv", index=False)

    clean = df[df["condition"] == "clean"]
    lab, warn = clean[clean["output"] != "warning"], clean[clean["output"] == "warning"]
    label_tab = (lab.groupby(["output", "model", "subset"])["macro_f1"].apply(fmt)
                 .unstack("subset").reindex(columns=["all", "agree", "conflict"]))
    label_tab = label_tab.join(lab[lab["subset"] == "all"].groupby(["output", "model"])
                               ["critical_recall"].apply(fmt).rename("critical_recall"))
    label_tab = label_tab.join(lab[lab["subset"] == "all"].groupby(["output", "model"])
                               ["ece"].apply(fmt).rename("ECE"))
    order = ["always conflict", "text+image pair", "multimodal", "multimodal_md0.3",
             "multimodal_swap0.25"]
    warn_tab = (warn.groupby("model")[["macro_f1", "critical_recall", "auroc"]]
                .agg(lambda g: fmt(g.dropna()) if g.notna().any() else "-")
                .rename(columns={"macro_f1": "conflict_F1", "critical_recall": "conflict_recall",
                                 "auroc": "AUROC"})
                .reindex([m for m in order if m in set(warn["model"])]))
    all_rows = df[df["subset"] == "all"]
    stress = (all_rows[all_rows["output"] != "warning"]
              .groupby(["output", "model", "condition"])["macro_f1"].mean()
              .unstack("condition").reindex(columns=list(CONDITIONS))).round(4)
    w = all_rows[(all_rows["output"] == "warning") & all_rows["auroc"].notna()]
    stress_warn = (w[w["model"] != "always conflict"]
                   .groupby(["model", "condition"])["auroc"].mean()
                   .unstack("condition").reindex(columns=list(CONDITIONS))).round(4)

    text = [f"# Results on {args.split} (experiment {EXPERIMENT})", "",
            "## Clean input: humanitarian labels (macro-F1 overall and on agree / conflict posts)", "",
            label_tab.to_string(), "",
            "## Clean input: conflict warning", "",
            warn_tab.to_string(), "",
            "## Stress tests: label macro-F1 by condition (mean over seeds)", "",
            stress.to_string(), "",
            "## Stress tests: conflict-warning AUROC by condition", "",
            stress_warn.to_string()]
    (RUNS_DIR / f"summary_{sfx}.md").write_text("\n".join(text))
    plot_stress(df, FIG_DIR / f"stress_test_{sfx}.png")
    plot_curves(args.quick, FIG_DIR / "learning_curves.png")
    print("\n".join(text[2:]))
    print(f"\nSaved runs/{EXPERIMENT}/summary_{sfx}.md/.csv and report/figures/{EXPERIMENT}/")


if __name__ == "__main__":
    main()
