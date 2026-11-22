# When Images and Text Disagree: Disaster Tweets (DLE-AI-202 Final Project, Track 2)

Multimodal classification of disaster tweets (CrisisMMD v2.0). One late-fusion
model (DistilBERT + MobileNetV3) predicts three things for every tweet-image pair:

1. the humanitarian category of the **text** (5 classes),
2. the humanitarian category of the **image** (5 classes),
3. a **conflict warning**: caption and image tell different stories.

Research question: what happens when captions are missing, misleading or
inconsistent with the images, compared with text-only and image-only models,
and does modality dropout during training make the model more robust?

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data
CrisisMMD v2.0 (CC BY-NC-SA 4.0), about 1.9 GB:
```bash
hf download QCRI/CrisisMMD --repo-type dataset --local-dir CrisisMMD
```
Official humanitarian split (tweet-level): train 13,608 / validation 2,237 / test 2,237 pairs.
Text and image labels come from `files_individual_events/*.tsv`; 8 categories are merged into 5.

## Reproduce
```bash
python -m src.reproduce --quick        # ~10-minute check that every step runs (CPU)
python -m src.reproduce                # all runs (4 models x 3 seeds) on validation, GPU recommended
python -m src.reproduce --split test   # final test numbers, once
```
Tables: `runs/summary_<split>.md`. Figures: `report/figures/`.

## Code
| File | Step |
|---|---|
| `src/prepare_data.py` | official split + separate text/image labels, 5 classes, agreement, text cleaning |
| `src/data.py` | dataset, transforms, stress-test conditions |
| `src/models.py` | text-only, image-only, late fusion with modality dropout |
| `src/train.py` | training, early stopping, learning curves, resumable |
| `src/evaluate.py` | clean / text missing / image missing / text swapped / image swapped |
| `src/metrics.py` | macro-F1, critical-class recall, calibration (ECE), conflict-warning F1 / AUROC |
| `src/report.py`, `src/reproduce.py` | summary tables, figures, one-command pipeline |
