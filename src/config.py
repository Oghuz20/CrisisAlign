"""Project-wide paths, label set and defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "CrisisMMD"                     # hf download QCRI/CrisisMMD ... --local-dir CrisisMMD
EVENTS_DIR = RAW_DIR / "files_individual_events"
SPLIT_DIR = RAW_DIR / "humanitarian"             # official split: train.json / dev.json / test.json
PROCESSED_DIR = ROOT / "data" / "processed"
# v1: first full GPU run            -> runs/<run>/,    report/figures/
# v2: 5 epochs, patience 2, label smoothing 0.1, head dropout 0.3, + swap augmentation
#                                    -> runs/v2/,       report/figures/v2/
# v3: v2 + stronger regularisation inside the networks (encoder dropout 0.2,
#     fusion dropout 0.2, weight decay 0.05)
#                                    -> runs/v3/,       report/figures/v3/
EXPERIMENT = "v3"
ENCODER_DROPOUT = 0.2     # DistilBERT dropout and attention dropout (default 0.1)
FUSION_DROPOUT = 0.2      # dropout after each modality projection and before the image head
RUNS_DIR = ROOT / "runs" / EXPERIMENT
FIG_DIR = ROOT / "report" / "figures" / EXPERIMENT

# Official splits (tweet-level, no tweet appears in two splits).
SPLIT_FILES = {"train": "train.json", "val": "dev.json", "test": "test.json"}

# Five humanitarian classes: minority classes merged as in the CrisisMMD baseline.
MERGE = {
    "injured_or_dead_people": "affected_individuals",
    "missing_or_found_people": "affected_individuals",
    "vehicle_damage": "infrastructure_and_utility_damage",
}
CLASSES = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]
CRITICAL_CLASS = "affected_individuals"          # missing these posts is the costliest error

TEXT_MODEL = "distilbert-base-uncased"
IMAGE_SIZE = 224
MAX_TEXT_LEN = 64
