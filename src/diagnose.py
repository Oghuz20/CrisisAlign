"""Analyze saved validation predictions by row ID, without model inference."""
import argparse
import csv
import itertools
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support

from src.artifacts import (CONDITIONS, dataset_metadata, evaluation_complete, evaluation_directory,
                           read_json, sha256, should_stop, update_selection, write_json)
from src.config import CLASSES, PROCESSED_DIR, RAW_DIR, ROOT
from src.metrics import agreement_metrics

TARGETS = ("y_text_label", "y_image_label", "y_agree")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def ids(a):
    return list(zip(a["tweet_id"].tolist(), a["image_id"].tolist()))


def aligned(left, right):
    keys = ids(left)
    index = {key: i for i, key in enumerate(ids(right))}
    if len(keys) != len(set(keys)) or len(index) != len(ids(right)) or set(keys) != set(index):
        raise ValueError("Prediction IDs are duplicated or differ across archives")
    order = [index[key] for key in keys]
    if any(not np.array_equal(left[k], right[k][order]) for k in TARGETS):
        raise ValueError("Targets differ after alignment by row ID")
    return order


def probabilities(a):
    with np.errstate(over="ignore"):
        return 1 - 1 / (1 + np.exp(-a["agree"]))


def classification(a, head, mask):
    y, p = a[f"y_{head}_label"][mask], a[head][mask].argmax(1)
    labels = range(len(CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {"macro_f1": float(f1.mean()), "confusion": confusion_matrix(y, p, labels=labels).tolist(),
            "classes": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "support": int(support[i])}
                        for i, name in enumerate(CLASSES)}}


def summary(a, mask):
    y = a["y_agree"][mask] == 0
    s = probabilities(a)[mask]
    p = s > .5
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[False, True]).ravel().tolist()
    pairs = list(zip(a["y_text_label"][mask], a["y_image_label"][mask]))
    types = {}
    for label, take in (("support", np.ones_like(y)), ("fp", ~y & p), ("fn", y & ~p)):
        types[label] = {f"{CLASSES[t]} / {CLASSES[i]}": n
                        for (t, i), n in Counter(k for k, selected in zip(pairs, take) if selected).items()}
    return {"rows": len(y), "prevalence": float(y.mean()), "alert_rate": float(p.mean()),
            "warning": {**agreement_metrics(s, a["y_agree"][mask]), "tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "score_quantiles_10_50_90": {label: np.quantile(s[take], [.1, .5, .9]).tolist()
                                        for label, take in (("agree", ~y), ("disagree", y)) if take.any()},
            "label_pairs": types,
            "classification": {label: {h: classification(a, h, take) for h in ("text", "image")}
                               for label, take in (("all", mask), ("agree", mask & (a["y_agree"] == 1)),
                                                   ("disagree", mask & (a["y_agree"] == 0))) if take.any()}}


def changes(left, right, overlap):
    order = aligned(left, right)
    truth = left["y_agree"] == 0
    before, after = probabilities(left) > .5, probabilities(right)[order] > .5
    x, z = left["image"].argmax(1), right["image"][order].argmax(1)
    critical = CLASSES.index("affected_individuals")
    img_flip = (left["y_image_label"] == critical) & ((x == critical) != (z == critical))
    keys = ids(left)
    for i in np.flatnonzero((before != after) | img_flip):
        yield {"tweet_id": keys[i][0], "image_id": keys[i][1],
               "image_overlap": keys[i] in overlap if overlap is not None else "",
               "text_label": CLASSES[left["y_text_label"][i]], "image_label": CLASSES[left["y_image_label"][i]],
               "warning_change": ("corrected" if after[i] == truth[i] else "introduced") if before[i] != after[i] else "",
               "image_prediction_before": CLASSES[x[i]], "image_prediction_after": CLASSES[z[i]],
               "critical_recall_change": ("gained" if z[i] == critical else "lost") if img_flip[i] else ""}


def audit_data(train, val, with_images):
    tweets = defaultdict(list)
    for row in train:
        tweets[row["tweet_id"]].append(row)
    unstable = [{k: row[k] for k in ("tweet_id", "image_id", "text_label", "image_label", "agree")}
                for group in tweets.values()
                if len({r["text"] for r in group}) == 1 and len({r["text_label"] for r in group}) > 1
                for row in group]
    if not with_images:
        return None, unstable

    def digest(row):
        return sha256(RAW_DIR / row["image_path"])

    overlap = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        reference = defaultdict(list)
        for row, checksum in zip(train, pool.map(digest, train)):
            reference[checksum].append(row["image_id"])
        for row, checksum in zip(val, pool.map(digest, val)):
            if checksum in reference:
                overlap[row["tweet_id"], row["image_id"]] = checksum, reference[checksum]
    return overlap, unstable


def read_run(run, data, validation):
    manifest = read_json(run / "config.json")
    cfg = manifest["training_config"]
    if cfg["quick"] or cfg["model"] != "multimodal" or manifest["dataset"] != data:
        raise ValueError(f"Expected a full fusion run on matching train/dev data: {run}")
    path = run / "history.csv"
    if sha256(path) != manifest["history_sha256"]:
        raise ValueError(f"History checksum mismatch: {run}")
    history = csv_rows(path)
    if not history:
        raise ValueError(f"Empty history: {run}")
    selected = dict.fromkeys(("classification", "warning"))
    stop = False
    for epoch, row in enumerate(history, 1):
        if stop or int(row["epoch"]) != epoch:
            raise ValueError(f"Invalid stopping history: {run}")
        selected["classification"] = update_selection(selected["classification"], epoch, float(row["val_macro_f1"]), 1e-4)
        ap = row["val_conflict_ap"]
        selected["warning"] = update_selection(selected["warning"], epoch, float(ap) if ap else None)
        stop = should_stop(epoch, selected, cfg["patience"])
    if epoch > cfg["epochs"] or (epoch < cfg["epochs"] and not stop):
        raise ValueError(f"Incomplete or invalid epoch horizon: {run}")
    expected_ids = {k: np.array([y[i] for y in validation]) for i, k in enumerate(("tweet_id", "image_id"))}
    expected_ids.update({k: np.array([y[k] for y in validation.values()]) for k in TARGETS})
    result = {}
    for objective, expected in selected.items():
        record = manifest["checkpoints"][objective]
        if expected != ({k: record[k] for k in ("epoch", "score")} if record else None):
            raise ValueError(f"Checkpoint selection does not match history: {run}, {objective}")
        if record is None:
            continue
        directory = evaluation_directory(run, objective)
        if not evaluation_complete(directory, "val", cfg, record, data["val"]):
            raise ValueError(f"Incomplete evaluation artifacts: {directory}")
        result[objective] = conditions = {}
        for condition in CONDITIONS:
            with np.load(directory / f"pred_val_{condition}.npz", allow_pickle=False) as archive:
                conditions[condition] = a = {key: archive[key] for key in archive.files}
            if (set(ids(a)) != set(validation) or len(ids(a)) != len(validation) or
                    not all(np.isfinite(a[k]).all() for k in ("text", "image", "agree")) or
                    not all(np.isin(a[k], range(len(CLASSES))).all() for k in TARGETS[:2]) or
                    not np.array_equal(a["y_agree"], a["y_text_label"] == a["y_image_label"])):
                raise ValueError(f"Invalid IDs, predictions, or labels: {directory}, {condition}")
        aligned(expected_ids, conditions["clean"])
    return manifest, history, result


def plot(path, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, a in series.items():
        y = a["y_agree"] == 0
        if y.any() and not y.all():
            precision, recall, _ = precision_recall_curve(y, probabilities(a))
            ax.step(recall, precision, where="post", label=name)
    if ax.lines:
        ax.legend(fontsize=7)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--tag", required=True, help="fresh directory name under report/diagnostics")
    parser.add_argument("--audit-images", action="store_true", help="hash train/dev images for overlap sensitivity")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag):
        parser.error("Invalid diagnostics tag")
    output = ROOT / "report" / "diagnostics" / args.tag
    try:
        if output.exists():
            raise ValueError(f"Output exists: {output}. Choose a fresh tag.")
        data = {s: dataset_metadata(s) for s in ("train", "val")}
        train, val = (csv_rows(PROCESSED_DIR / f"{s}.csv") for s in ("train", "val"))
        validation = {(r["tweet_id"], r["image_id"]): {f"y_{h}_label": CLASSES.index(r[f"{h}_label"])
                                                     for h in ("text", "image")} | {"y_agree": int(r["agree"])}
                      for r in val}
        if len(validation) != len(val):
            raise ValueError("Duplicate processed validation IDs")
        overlap, annotation_rows = audit_data(train, val, args.audit_images)
        metrics, series, manifests = {}, {}, {}
        for run in args.run_dir:
            if run.name in manifests:
                raise ValueError(f"Duplicate run name: {run.name}")
            manifest, history, results = read_run(run, data, validation)
            manifests[run.name] = {"manifest": manifest, "history": history}
            for objective, predictions in results.items():
                name = f"{run.name}:{objective}"
                series[name] = predictions["clean"]
                metrics[name] = {k: summary(a, np.ones(len(ids(a)), bool)) for k, a in predictions.items()}
                if overlap is not None:
                    mask = np.array([k in overlap for k in ids(predictions["clean"])])
                    for subset, take in (("clean_exact_images_only", mask), ("clean_without_exact_images", ~mask)):
                        if take.any():
                            metrics[name][subset] = summary(predictions["clean"], take)
        transitions = []
        for (a, left), (b, right) in itertools.combinations(series.items(), 2):
            if manifests[a.split(":")[0]]["manifest"]["seed"] == manifests[b.split(":")[0]]["manifest"]["seed"]:
                transitions.extend({"before": a, "after": b, **row} for row in changes(left, right, overlap))
        output.mkdir(parents=True)
        write_json(output / "analysis.json", {
            "runs": manifests, "dataset": data, "metrics": metrics, "class_order": CLASSES,
            "exact_image_overlap": [{"tweet_id": k[0], "image_id": k[1], "sha256": v[0], "train_image_ids": v[1]}
                                    for k, v in overlap.items()] if overlap is not None else None,
            "inconsistent_training_rows": annotation_rows})
        if transitions:
            with (output / "paired_changes.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=transitions[0])
                writer.writeheader()
                writer.writerows(transitions)
        plot(output / "clean_precision_recall.png", series)
        print(f"Saved validation diagnostics to {output}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"Diagnostics blocked: {exc}\n")


if __name__ == "__main__":
    main()
