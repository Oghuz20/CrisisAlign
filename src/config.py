"""Project-wide paths, label set and defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "CrisisMMD"                     # hf download QCRI/CrisisMMD ... --local-dir CrisisMMD
EVENTS_DIR = RAW_DIR / "files_individual_events"
SPLIT_DIR = RAW_DIR / "humanitarian"             # official split: train.json / dev.json / test.json
PROCESSED_DIR = ROOT / "data" / "processed"
# Version 2 (anti-overfitting settings + swap augmentation). Version 1 results
# (first full GPU run) stay untouched in runs/ and report/figures/.
EXPERIMENT = "v2"
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
