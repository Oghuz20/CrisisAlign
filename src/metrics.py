"""Metrics for the three outputs."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.config import CLASSES, CRITICAL_CLASS

CRIT = CLASSES.index(CRITICAL_CLASS)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, y, bins=15):
    """Expected calibration error of the top-class confidence."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)


def class_metrics(logits, y):
    probs = softmax(logits)
    pred = probs.argmax(1)
    labels = list(range(len(CLASSES)))
    return {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "critical_recall": float(recall_score(y == CRIT, pred == CRIT, zero_division=0)),
        "ece": ece(probs, y),
        "per_class_f1": dict(zip(CLASSES, f1_score(y, pred, labels=labels, average=None,
                                                   zero_division=0).round(4).tolist())),
    }


def agreement_metrics(conflict_score, agree):
    """Higher scores indicate unequal humanitarian labels."""
    conflict = 1 - agree.astype(int)
    pred = (conflict_score > 0.5).astype(int)
    out = {"conflict_f1": float(f1_score(conflict, pred, zero_division=0)),
           "conflict_precision": float(precision_score(conflict, pred, zero_division=0)),
           "conflict_recall": float(recall_score(conflict, pred, zero_division=0)),
           "conflict_ap": None}
    if 0 < conflict.sum() < len(conflict):
        out["conflict_auroc"] = float(roc_auc_score(conflict, conflict_score))
        out["conflict_ap"] = float(average_precision_score(conflict, conflict_score))
    return out


def evaluate_outputs(out, t):
    """out: dict of logits arrays; t: dict of target arrays. Overall + agree/conflict subsets."""
    res = {}
    for subset, m in (("all", np.ones(len(t["agree"]), bool)),
                      ("agree", t["agree"] == 1), ("conflict", t["agree"] == 0)):
        if not m.any():
            continue
        r = {}
        if "text" in out:
            r["text"] = class_metrics(out["text"][m], t["text_label"][m])
        if "image" in out:
            r["image"] = class_metrics(out["image"][m], t["image_label"][m])
        res[subset] = r
    if "agree" in out:
        p_agree = 1 / (1 + np.exp(-out["agree"]))
        res["all"]["warning"] = agreement_metrics(1 - p_agree, t["agree"])
    return res
