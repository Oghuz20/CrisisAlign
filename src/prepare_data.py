"""Step 1: build clean train / val / test tables from CrisisMMD v2.0.

Run from the project root:
    python -m src.prepare_data

For every tweet-image pair in the OFFICIAL humanitarian split it
- takes the separate TEXT and IMAGE humanitarian labels from the per-event
  annotation files (the official JSON has only one label per row),
- merges the 8 categories into 5 (as in the CrisisMMD multimodal baseline),
- marks whether text and image agree (the "conflict warning" target),
- repairs broken text encoding (mojibake such as "Weâ€™re" -> "We’re") and normalises
  links, user mentions and the retweet prefix,
- checks that every image file opens.

Writes data/processed/{train,val,test}.csv and data/processed/stats.json.
"""
import json
import re

import ftfy
import pandas as pd
from PIL import Image

from src.config import (CLASSES, EVENTS_DIR, MERGE, PROCESSED_DIR, RAW_DIR,
                        SPLIT_DIR, SPLIT_FILES)

URL = re.compile(r"https?://\S+")
USER = re.compile(r"@\w+")
RT = re.compile(r"^\s*RT\s+@user:?\s*", flags=re.IGNORECASE)


def clean_text(text):
    text = ftfy.fix_text(str(text))
    text = URL.sub("URL", text)
    text = USER.sub("@user", text)
    text = RT.sub("", text)
    text = text.replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def load_annotations():
    frames = [pd.read_csv(f, sep="\t", dtype={"tweet_id": str, "image_id": str})
              for f in sorted(EVENTS_DIR.glob("*.tsv"))]
    if not frames:
        raise SystemExit(f"No annotation files in {EVENTS_DIR}. Download the data first (README).")
    ann = pd.concat(frames, ignore_index=True)
    return ann.drop_duplicates("image_id").set_index("image_id")


def image_ok(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_split(name, ann):
    rows = pd.DataFrame(json.loads((SPLIT_DIR / SPLIT_FILES[name]).read_text(encoding="utf-8")))
    rows["image_id"] = rows["image_id"].astype(str)
    df = rows[["event_name", "tweet_id", "image_id"]].join(
        ann[["text_human", "image_human", "tweet_text", "image_path"]], on="image_id")
    missing = df["text_human"].isna() | df["image_human"].isna()
    if missing.any():
        print(f"  {name}: {missing.sum()} rows without both labels dropped")
        df = df[~missing]
    df["text_label"] = df["text_human"].replace(MERGE)
    df["image_label"] = df["image_human"].replace(MERGE)
    assert set(df["text_label"]) <= set(CLASSES) and set(df["image_label"]) <= set(CLASSES)
    df["agree"] = (df["text_label"] == df["image_label"]).astype(int)
    df["text"] = df["tweet_text"].map(clean_text)
    ok = df["image_path"].map(lambda p: image_ok(RAW_DIR / p))
    if (~ok).any():
        print(f"  {name}: {(~ok).sum()} unreadable images dropped")
        df = df[ok]
    df.insert(0, "split", name)
    return df[["split", "event_name", "tweet_id", "image_id", "image_path",
               "text", "text_label", "image_label", "agree"]].reset_index(drop=True)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ann = load_annotations()
    print(f"Annotations: {len(ann)} tweet-image pairs from {ann['tweet_id'].nunique()} tweets")
    stats = {}
    for name in SPLIT_FILES:
        df = build_split(name, ann)
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        stats[name] = {
            "pairs": len(df),
            "tweets": int(df["tweet_id"].nunique()),
            "conflict_rate": round(1 - df["agree"].mean(), 4),
            "text_labels": df["text_label"].value_counts().reindex(CLASSES, fill_value=0).to_dict(),
            "image_labels": df["image_label"].value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        }
        print(f"  {name:5s}: {len(df):6d} pairs | text-image conflict {stats[name]['conflict_rate']:.1%}")
    (PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print("Saved data/processed/{train,val,test}.csv and stats.json")


if __name__ == "__main__":
    main()
