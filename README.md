# When Images and Text Disagree: Disaster Tweets (DLE-AI-202 Final Project, Track 2)

Multimodal classification of disaster tweets (CrisisMMD v2.0). One late-fusion
model (DistilBERT + MobileNetV3) predicts three things for every tweet-image pair:

1. the humanitarian category of the **text** (5 classes),
2. the humanitarian category of the **image** (5 classes),
3. a **humanitarian-label disagreement warning** for unequal merged modality labels.

The project compares label prediction and disagreement warnings on original,
missing-modality and synthetically swapped pairs. The target is an annotation-based
proxy for cross-modal disagreement.

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

## Versions
- **v1** (`runs/<run>/`, `report/figures/`): first full run, 8 epochs, no label smoothing, 4 models.
- **v2** (`runs/v2/`, `report/figures/v2/`): against overfitting (5 epochs, patience 2, label smoothing 0.1,
  head dropout 0.3) and a fifth model with swap augmentation for the conflict warning.
  Model choices are made on validation; the test split is evaluated once.
- **v3** (default recipe): v2 + stronger regularisation inside the networks
  (DistilBERT dropout and attention dropout 0.2, fusion dropout 0.2, weight decay 0.05).

## Reproduce

For the v2-compatible comparison, run plain fusion and caption swap at seeds
42, 43 and 44. Choose the worker count once and use the same arguments for
validation and final test.

```bash
python -m src.reproduce --recipe v2 --matrix fusion --seeds 42 43 44 --tag track2 --workers 6
```

The launcher prepares train/dev if needed. The `v2` recipe keeps five epochs,
patience two per objective, head dropout 0.3, label smoothing 0.1, weight decay 0.01,
the pretrained text model's default internal dropout, and no extra projection
dropout. DistilBERT, MobileNetV3 and head learning rates remain `2e-5`, `1e-4`
and `1e-3`. Plain fusion and caption swap differ only in swap augmentation
(`0` versus `0.25`).

CUDA uses native BF16 when available and FP16 with gradient scaling otherwise.
The run and evaluation manifests record the selected precision. CPU runs use FP32.

New run directories are under `runs/disagreement/<recipe>/` and include a
configuration fingerprint. A completed run requires its manifest, history and
hashed checkpoints. Use `--resume` for interrupted runs with a resumable
checkpoint or a new `--tag` for a separate run. Retain the selected checkpoints
and `last.pt` on the GPU machine.

The **classification** checkpoint uses validation label macro-F1
(mean of the two label heads for fusion, minimum improvement `1e-4`). The
**warning** checkpoint uses clean validation average precision (AP). Ties
keep the earlier checkpoint. Training stops early only when both active
objectives have exhausted their patience. If clean validation has just one
warning class, AP and AUROC are undefined and no warning checkpoint is selected.
Evaluation reports both checkpoints, using disagreement score `> 0.5` for
warning F1, precision and recall. The CSV identifies the checkpoint and seed.

After freezing the matrix and the reporting rule on validation, prepare the
official test split and evaluate the completed runs:

```bash
python -m src.prepare_data --splits test
python -m src.reproduce --recipe v2 --matrix fusion --seeds 42 43 44 --tag track2 --workers 6 --split test
```

Test mode inspects checkpoints before evaluating. Reports are written under
`report/generated/`. To rebuild archived tables from committed data without
checkpoints or model dependencies:

```bash
python -m src.report --archive v2 --split val
python -m src.report --archive v2 --split test
python -m src.report --archive v1 --split test
python -m src.report --archive v1 --split val --quick
```

For new runs whose results were shared without model weights, pass
`--run-dir <path>` once per run. Copy the run manifest and the evaluation
metrics, prediction archives and evaluation manifest for each checkpoint.

The v2 validation archive includes the historical independent text+image pair.
Its F1, precision and recall rule is hard class disagreement, and ranking
metrics use the soft score. Historical exports retain the metrics recorded at
the time of the experiment.

## Code
| File | Step |
|---|---|
| `src/prepare_data.py` | official split + separate text/image labels, 5 classes, agreement, text cleaning |
| `src/data.py` | dataset, transforms, stress-test conditions |
| `src/models.py` | text-only, image-only, late fusion with modality dropout |
| `src/train.py` | training, early stopping, learning curves, resumable |
| `src/evaluate.py` | clean / text missing / image missing / text swapped / image swapped |
| `src/metrics.py` | macro-F1, critical-class recall, calibration (ECE), disagreement F1, precision, recall, AP and AUROC |
| `src/report.py`, `src/reproduce.py` | summary tables and experiment pipeline |
