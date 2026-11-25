"""Build tables for explicit run directories or committed historical results."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from src.artifacts import (CONDITIONS, evaluation_complete, evaluation_directory, identity,
                           model_name, read_json)
from src.config import ROOT

FIELDS = ("model", "seed", "objective", "checkpoint_epoch", "checkpoint_sha256",
          "condition", "subset", "output", "macro_f1", "accuracy", "critical_recall", "ece",
          "conflict_f1", "conflict_precision", "conflict_recall", "conflict_ap", "conflict_auroc")


def evaluation_rows(run_dir, split, archive):
    manifest = read_json(run_dir / "config.json")
    config = manifest if archive else manifest["training_config"]
    if archive:
        checkpoints = {"classification": {"epoch": config["best_epoch"]}}
    else:
        checkpoints = {key: record for key, record in manifest["checkpoints"].items()
                       if key != "last" and record is not None}
    for objective, checkpoint in checkpoints.items():
        directory = evaluation_directory(run_dir, objective)
        path = directory / f"eval_{split}.json"
        if archive and not path.is_file():
            continue
        if not archive:
            meta = read_json(directory / f"evaluation_{split}.json")
            if not evaluation_complete(directory, split, config, checkpoint, meta["dataset"]):
                raise ValueError(f"Incomplete evaluation artifacts: {directory}")
        common = {"model": model_name({"modality_dropout": 0, "swap_aug": 0, **config}),
                  "seed": config["seed"], "objective": objective,
                  "checkpoint_epoch": checkpoint["epoch"], "checkpoint_sha256": checkpoint.get("sha256")}
        for condition, subsets in read_json(path).items():
            for subset, outputs in subsets.items():
                for output, metrics in outputs.items():
                    yield {**common, "condition": condition, "subset": subset, "output": output,
                           **{key: value for key, value in metrics.items() if key in FIELDS}}


def archived_baselines(directory, split, quick):
    path = directory / f"summary_{split}{'_quick' if quick else ''}.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["model"] not in ("text+image pair", "always conflict"):
                continue
            yield {"model": row["model"], "seed": int(row["seed"]), "objective": "classification",
                   "condition": row["condition"], "subset": row["subset"], "output": "warning",
                   "conflict_f1": float(row["macro_f1"]), "conflict_recall": float(row["critical_recall"]),
                   "conflict_auroc": float(row["auroc"]) if row["auroc"] else None}


def pair_baselines(runs, split):
    # Pair only runs in the same experiment and with equivalent training settings.
    pairs = defaultdict(dict)
    for directory in runs:
        config = read_json(directory / "config.json")["training_config"]
        if config["model"] in ("text", "image"):
            key = identity({k: v for k, v in config.items() if k != "model"})
            if config["model"] in pairs[key]:
                raise ValueError(f"Duplicate unimodal run for the same configuration: {directory}")
            pairs[key][config["model"]] = (directory, config["seed"])
    for pair in pairs.values():
        if set(pair) != {"text", "image"}:
            continue
        import numpy as np

        from src.metrics import agreement_metrics, softmax

        text, seed = pair["text"]
        image, _ = pair["image"]
        for condition in CONDITIONS:
            suffix = f"pred_{split}_{condition}.npz"
            with np.load(text / suffix) as t, np.load(image / suffix) as v:
                for key in ("tweet_id", "image_id", "y_agree", "y_text_label", "y_image_label"):
                    if not np.array_equal(t[key], v[key]):
                        raise ValueError(f"Unaligned text/image predictions for seed {seed}, {condition}: {key}")
                y = t["y_agree"]
                pt, pv = softmax(t["text"]), softmax(v["image"])
                hard = agreement_metrics((pt.argmax(1) != pv.argmax(1)).astype(float), y)
                soft = agreement_metrics(1 - (pt * pv).sum(1), y)
                hard["conflict_ap"] = soft["conflict_ap"]
                hard["conflict_auroc"] = soft.get("conflict_auroc")
                common = {"seed": seed, "objective": "classification", "condition": condition,
                          "subset": "all", "output": "warning"}
                yield {**common, "model": "text+image pair", **hard}
                if condition == "clean":
                    yield {**common, "model": "always conflict", **agreement_metrics(np.ones(len(y)), y)}


def table(rows, group_keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    columns = (*group_keys, "n", *metrics)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for key, group in sorted(groups.items()):
        cells = [*key, str(len(group))]
        for metric in metrics:
            values = [row[metric] for row in group if row.get(metric) is not None]
            if len(values) > 1:
                cells.append(f"{mean(values):.4f} ± {stdev(values):.4f} ({len(values)})")
            else:
                cells.append(f"{values[0]:.4f} (1)" if values else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) if groups else "No results available."


def write_report(rows, split, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"summary_{split}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    label = [r for r in rows if r["output"] != "warning"]
    warning = [r for r in rows if r["output"] == "warning"]
    warning_metrics = ("conflict_f1", "conflict_precision", "conflict_recall", "conflict_ap", "conflict_auroc")
    text = [
        f"# Humanitarian-label disagreement results on {split}",
        "Results are mean ± sample SD when multiple seeds contribute. Parentheses give metric support in seeds.",
        "Fusion F1, precision and recall use disagreement score > 0.5. The independent pair uses hard "
        "class disagreement for those metrics and the soft score for AP and AUROC.",
        "A missing-input warning predicts the original pair’s label disagreement from the visible input. "
        "The corresponding hidden-label classification result measures label inference.",
        "## Clean classification",
        table([r for r in label if r["condition"] == "clean"], ("objective", "model", "output", "subset"),
              ("macro_f1", "accuracy", "critical_recall", "ece")),
        "## Disagreement warning",
        table(warning, ("objective", "model", "condition"), warning_metrics),
        "## Stress-test classification",
        table([r for r in label if r["subset"] == "all" and r["condition"] != "clean"],
              ("objective", "model", "output", "condition"), ("macro_f1",)),
    ]
    path = output / f"summary_{split}.md"
    path.write_text("\n\n".join(text) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive", choices=["v1", "v2"])
    mode.add_argument("--run-dir", action="append", type=Path)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--quick", action="store_true", help="select the historical v1 quick validation runs")
    args = p.parse_args(argv)
    if args.quick and (args.archive, args.split) != ("v1", "val"):
        p.error("--quick requires --archive v1 --split val")
    try:
        if args.archive:
            archive = ROOT / "runs" / ("v2" if args.archive == "v2" else "")
            runs = sorted(path.parent for path in archive.glob("*/config.json")
                          if path.parent.name.endswith("_quick") == args.quick)
            # Full v1 has validation histories but no full validation evaluation JSONs.
            rows = [row for run in runs for row in evaluation_rows(run, args.split, archive=True)]
            rows.extend(archived_baselines(archive, args.split, args.quick))
            output = ROOT / "report" / "generated" / f"{args.archive}{'_quick' if args.quick else ''}"
        else:
            runs = sorted(set(path.resolve() for path in args.run_dir))
            configs = [read_json(run / "config.json")["training_config"] for run in runs]
            compare = {identity({k: v for k, v in config.items()
                                 if k not in ("model", "seed", "modality_dropout", "swap_aug")})
                       for config in configs}
            if len(compare) != 1:
                raise ValueError("Report runs must share training settings outside model, seed and interventions")
            ids = [(model_name(config), config["seed"]) for config in configs]
            if len(ids) != len(set(ids)):
                raise ValueError("Report contains duplicate model/seed combinations")
            rows = [row for run in runs for row in evaluation_rows(run, args.split, archive=False)]
            rows.extend(pair_baselines(runs, args.split))
            output = ROOT / "report" / "generated" / identity([str(path.relative_to(ROOT))
                       if path.is_relative_to(ROOT) else str(path) for path in runs])[:16]
        if not rows:
            raise ValueError(f"No evaluation results found for {args.split}")
        write_report(rows, args.split, output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(1, f"Report blocked: {exc}\n")


if __name__ == "__main__":
    main()
