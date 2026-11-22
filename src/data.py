"""Dataset, image transforms and the stress-test conditions.

Conditions (applied at evaluation time; "clean" is the normal input):
    clean          text and image as posted
    text_missing   no caption            (has_text = 0)
    image_missing  no image              (has_image = 0)
    text_swapped   caption taken from another post with a DIFFERENT text label
                   (a misleading caption; targets follow the new caption)
    image_swapped  image taken from another post with a DIFFERENT image label
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import CLASSES, IMAGE_SIZE, PROCESSED_DIR, RAW_DIR

CONDITIONS = ("clean", "text_missing", "image_missing", "text_swapped", "image_swapped")
LABEL_ID = {c: i for i, c in enumerate(CLASSES)}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_split(name, quick=False, seed=0):
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv", dtype={"tweet_id": str, "image_id": str})
    df["text"] = df["text"].fillna("")
    if quick:                                     # small stratified subset for a pipeline check
        n = 600 if name == "train" else 200
        per_class = max(1, n // len(CLASSES))
        idx = [i for _, g in df.groupby("image_label")
               for i in g.sample(min(len(g), per_class), random_state=seed).index]
        df = df.loc[sorted(idx)].reset_index(drop=True)
    return df


def image_transform(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)), transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def apply_condition(df, condition, seed=0):
    """Return a copy of df with the stress-test condition applied (and targets updated)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    df = df.copy()
    df["has_text"], df["has_image"] = 1, 1
    rng = np.random.default_rng(seed)
    if condition == "text_missing":
        df["text"], df["has_text"] = "", 0
    elif condition == "image_missing":
        df["has_image"] = 0
    elif condition in ("text_swapped", "image_swapped"):
        col_label = "text_label" if condition == "text_swapped" else "image_label"
        cols = ["text", "text_label"] if condition == "text_swapped" else ["image_path", "image_label"]
        donor = np.empty(len(df), dtype=int)
        labels = df[col_label].to_numpy()
        for i, lab in enumerate(labels):                  # donor with a different label
            pool = np.flatnonzero(labels != lab)
            donor[i] = rng.choice(pool)
        df[cols] = df[cols].to_numpy()[donor]
    df["agree"] = (df["text_label"] == df["image_label"]).astype(int)
    return df.reset_index(drop=True)


class CrisisDataset(Dataset):
    def __init__(self, df, train=False, use_text=True, use_image=True):
        self.df = df.reset_index(drop=True)
        if "has_text" not in self.df:
            self.df["has_text"], self.df["has_image"] = 1, 1
        self.tf = image_transform(train)
        self.use_text, self.use_image = use_text, use_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        if self.use_image and r["has_image"]:
            with Image.open(RAW_DIR / r["image_path"]) as im:
                image = self.tf(im.convert("RGB"))
        else:
            image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)   # = the mean image after normalisation
        return {
            "text": r["text"] if self.use_text else "",
            "image": image,
            "has_text": float(r["has_text"]),
            "has_image": float(r["has_image"]),
            "text_label": LABEL_ID[r["text_label"]],
            "image_label": LABEL_ID[r["image_label"]],
            "agree": float(r["agree"]),
        }


class Collator:
    """Tokenises a batch of captions in one call (fast tokenizer)."""

    def __init__(self, tokenizer, max_len):
        self.tok, self.max_len = tokenizer, max_len

    def __call__(self, items):
        enc = self.tok([it["text"] for it in items], padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                 "image": torch.stack([it["image"] for it in items])}
        for k, dtype in (("has_text", torch.float32), ("has_image", torch.float32),
                         ("text_label", torch.long), ("image_label", torch.long),
                         ("agree", torch.float32)):
            batch[k] = torch.tensor([it[k] for it in items], dtype=dtype)
        return batch
