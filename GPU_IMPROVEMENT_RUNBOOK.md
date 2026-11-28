# CrisisAlign GPU pilot for image-class weighting

The plain-fusion seed-42 warning checkpoint improved clean validation disagreement AP from .7981
to .8029. Its affected-individuals image recall fell from 35/87 to 24/87, while precision rose
from .467 to .545. Caption-swap fusion recovered more disagreements at the fixed threshold
but produced more false alarms and had lower clean AP than plain fusion. This pilot tests
whether a small change to the image loss recovers rare-class recall while preserving warning
quality. Disagreement means unequal humanitarian annotations.

Run both arms on the same GPU. Commit `5ce4a06` contains the original pilot archives, but their
manifests identify CPU/FP32 training and the checkout does not contain the weights. Those runs
are useful references. Use a fresh matched GPU control for the decision.

## Get the approved code

Wait until the reviewer pushes this change to `fix/overfitting-analysis`.

Fresh clone:

```powershell
git clone --branch fix/overfitting-analysis https://github.com/Oghuz20/CrisisAlign.git
Set-Location CrisisAlign
```

Existing clone:

```powershell
git status --short
git fetch origin
git switch fix/overfitting-analysis
git pull --ff-only origin fix/overfitting-analysis
git rev-parse HEAD
```

If switching is blocked by local edits, save them on a private branch or use another clone.
One alternative is `git stash push --include-untracked -m "Before CrisisAlign GPU pilot"`.
Inspect `git stash list` and restore with `git stash apply` on the branch where the edits belong.
Ignored local datasets and checkpoints stay on disk. If a branch exists only on the remote,
use `git switch --track origin/fix/overfitting-analysis`. If the fast-forward fails,
pause and coordinate with the reviewer.

## Environment and data

Use Python 3.12–3.14 with a CUDA-enabled matching PyTorch/torchvision pair.
The repository requirements exclude Python 3.14.1 for the pinned torchvision release.
In a fresh PowerShell session:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

Install the CUDA-enabled PyTorch and torchvision pair using the command from the
[official PyTorch installer](https://pytorch.org/get-started/locally/) for the machine.
Then install the remaining repository requirements without reinstalling the CUDA wheels:

```powershell
$packages = Get-Content requirements.txt | Where-Object { $_ -match '^[a-zA-Z]' -and $_ -notmatch '^torch(vision)?==' }
python -m pip install @packages
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('Native BF16:', torch.cuda.is_bf16_supported(including_emulation=False)); x=torch.ones((4,4),device='cuda'); exec('with torch.autocast(\x22cuda\x22,dtype=torch.float16):\n print(\x22FP16 CUDA matmul:\x22,(x@x).dtype)')"
git rev-parse HEAD
python -m unittest discover -s tests -v
python -m src.train --help
python -m src.evaluate --help
python -m src.diagnose --help
```

Check CUDA before starting. The pipeline uses native BF16 when available and FP16 otherwise,
with gradient scaling during FP16 training. The two arms must use the same hardware,
software, data, worker count, batch size, and mixed precision.

Place the existing CrisisMMD copy in `CrisisMMD/` as described in [TELIMAT.md](TELIMAT.md).
Verify `CrisisMMD/humanitarian/train.json`, `CrisisMMD/humanitarian/dev.json`,
`CrisisMMD/files_individual_events/` and `CrisisMMD/data_image/`. Reuse the existing processed
train/dev tables if present. For a fresh setup:

```powershell
python -m src.prepare_data --splits train val
python -c "from src.artifacts import dataset_metadata; print(dataset_metadata('train')); print(dataset_metadata('val'))"
```

Expect 13,608 train and 2,237 dev pairs. The original pilot hashes begin `c4d5ca190a25`
and `b9a2b96fb3bf`. Git commit, runtime precision, and complete hashes appear in the run manifests.

## Seed 42 only

```powershell
$common = @('--model','multimodal','--recipe','v2','--workers','6','--tag','rare-image','--device','cuda')
python -m src.train @common --seed 42
python -m src.train @common --seed 42 --image-class-weighting capped-sqrt
```

Both commands train plain fusion. The treatment caps inverse-square-root image-class
weights to `[0.75, 1.5]`, using only training labels. For the current train split,
the expected weights in label order are **1.5000, .9539, .7500, 1.2096, 1.2675**.
Only the smoothed image training cross-entropy receives these weights. Validation loss
and metrics, precision routing, the classification selection rule and the warning AP
selection rule retain their baseline behavior. Read the logged counts and weights in
the treatment’s `config.json`.

```powershell
$resolve = 'import sys; from src.artifacts import training_config, run_directory; print(run_directory(training_config(sys.argv[1:])[0]))'
$control = python -c $resolve @common --seed 42
$treatment = python -c $resolve @common --seed 42 --image-class-weighting capped-sqrt
python -m src.evaluate --run-dir "$control" --split val --workers 6
python -m src.evaluate --run-dir "$treatment" --split val --workers 6
python -m src.diagnose --run-dir "$control" --run-dir "$treatment" --tag rare-image-42 --audit-images
```

With the exact settings above, the directories end in `multimodal_seed42-97ef3b3f118e5d40`
and `multimodal_seed42-d23e0beabd47f133`. If rerunning evaluation or diagnosis, use the same
run directories. Use a fresh diagnostics tag if that report directory already exists.
Use `src.diagnose` for the comparison across different weighting settings. The
standard `src.report` command groups comparable settings and can summarize each arm separately.

## Decision before adding seeds

Compare warning checkpoints against warning checkpoints and classification checkpoints
against classification checkpoints. Report the full official dev results first.
Use AP from the warning checkpoint as the primary ranking metric, with F1, precision and
recall at the fixed score threshold `> 0.5`, and AUROC. Show both heads’ macro-F1, their
macro-F1 on unequal-label examples, critical image precision/recall and supports, and
text-swap and image-swap AP/AUROC alongside each condition’s prevalence.

Advance the seed-42 pilot if the warning-selected treatment recovers **at least five**
additional affected-individuals images out of 87, clean disagreement AP is within **.005**
of control or higher, and critical image precision falls by at most **.03**. Require overall
and unequal-label macro-F1 for each head to stay within **.01** of the corresponding control
on both checkpoints. Record any synthetic-swap AUROC drop above **.02** for review before
advancing. These are prospective engineering margins. Reject or defer if the margins fail,
the runs are mismatched, or the gain depends on overlap rows. Review duplicate-excluded
metrics as a secondary sensitivity check and avoid selection on test. One seed is a pilot.

If it passes, add two matched controls and two matched treatments:

```powershell
foreach ($seed in 43,44) {
    python -m src.train @common --seed $seed
    python -m src.train @common --seed $seed --image-class-weighting capped-sqrt
}
$diagnosticArgs = @('--run-dir',"$control",'--run-dir',"$treatment")
foreach ($seed in 43,44) {
    $c = python -c $resolve @common --seed $seed
    $t = python -c $resolve @common --seed $seed --image-class-weighting capped-sqrt
    python -m src.evaluate --run-dir "$c" --split val --workers 6
    python -m src.evaluate --run-dir "$t" --split val --workers 6
    $diagnosticArgs += @('--run-dir',"$c",'--run-dir',"$t")
}
python -m src.diagnose @diagnosticArgs --tag rare-image-42-44 --audit-images
```

Compare paired deltas by seed and report all three. Look for gains in at least two seeds
and a positive mean critical-recall delta with the same AP and precision guardrails.
Synthetic donor draws can vary across seeds. Compare within each seed and record that
the current archives store recipient IDs and effective targets, without donor IDs.

## Output and verification

Each `runs/disagreement/v2/<model>_seed<seed>-<fingerprint>/` should contain:

| File | Expected contents |
|---|---|
| `config.json`, `started.json` | Complete training configuration and provenance. The final config contains the full SHA-256 configuration fingerprint, dataset hashes and checkpoint hashes. |
| `history.csv`, `learning_curve.png` | Per-epoch loss, label and warning metrics, winning epochs, elapsed seconds, and training curves. |
| `best_classification_epoch*.pt`, `best_warning_epoch*.pt`, `last.pt` | Objective-specific weights and resume state. Use the filenames listed in `config.json`. |
| `eval_val.json`, `evaluation_val.json`, `pred_val_*.npz` | Classification checkpoint metrics, integrity metadata and the five validation prediction archives. |
| `warning/eval_val.json`, `warning/evaluation_val.json`, `warning/pred_val_*.npz` | Equivalent artifacts for the warning checkpoint. |

The comparison is under `report/diagnostics/rare-image-42/`: `analysis.json`,
`clean_precision_recall.png` and `paired_changes.csv` when decisions differ. The JSON
stores confusion matrices in the listed class order and precision, recall, F1 and supports.

Check the GPU and precision in the log, both checkpoint objectives in the final
config, and the expected weight vector in the treatment only. Verify checkpoint bytes:

```powershell
python -c "import sys; from pathlib import Path; from src.artifacts import validate_training; [print(p, validate_training(Path(p))['checkpoints']) for p in sys.argv[1:]]" "$control" "$treatment"
```

The diagnostics verify archive checksums, history-based checkpoint selection, stable IDs
and 2,237 unique dev rows in each condition. Confirm that the commands used `--split val`
and the reports contain only dev metrics.

### Troubleshooting

- **CUDA out of memory.** Close competing processes. If batch size must change, append
  `--batch-size 16` to `$common` and run both arms under a fresh shared tag on the new configuration.
- **Missing data.** Confirm the repository root and image directory, then prepare train/dev only.
- **Missing pretrained caches.** With network access on the teammate’s machine, allow the
  normal training initialization to retrieve DistilBERT and ImageNet MobileNetV3 weights.
  Use compatible cached copies on an offline machine. Keep both arms pretrained.
- **Mixed-precision error.** Check the CUDA-enabled torch/torchvision pair and driver, rerun
  the CUDA check, and record the exact error. Resume requires the same environment and precision.
- **Missing or mismatched checkpoints.** Check the exact training arguments, dataset hashes
  and file hashes. Restore the matching weights from local storage or use a fresh matched pair.
- **Interrupted training.** Append `--resume` to the exact interrupted `src.train` command
  when `started.json` and `last.pt` exist. Resume checks source, data and environment.
  Persistent-worker randomness can differ from uninterrupted execution. If resume is rejected,
  use a fresh matched tag.
- **Incomplete prediction archive.** Rerun the matching `src.evaluate --split val` command.
  It regenerates incomplete artifacts for the same checkpoint and dataset. If identity changed,
  keep the existing run and resolve the mismatch before proceeding.

## Share and report back

Both teammates should pull the latest branch before making commits. Share the scoped
diagnostics and small result metadata in Git. Keep `.pt`, datasets, and environment folders
local. Share NPZs via the team’s artifact storage with the corresponding manifests and hashes.
Review the staged paths before committing:

```powershell
git switch fix/overfitting-analysis
git pull --ff-only origin fix/overfitting-analysis
foreach ($run in @($control,$treatment)) {
    git add -- "$run/config.json" "$run/history.csv" "$run/learning_curve.png" "$run/eval_val.json" "$run/evaluation_val.json" "$run/warning/eval_val.json" "$run/warning/evaluation_val.json"
}
git add -- report/diagnostics/rare-image-42
git diff --cached --name-only
git commit -m "Add matched seed-42 image-weighting results"
git push origin HEAD:refs/heads/fix/overfitting-analysis
```

Keep existing code, historical outputs, and other runs intact. Send back one row per
arm and checkpoint with Git SHA, GPU and precision, recipe and changed factor, elapsed
time, both winning epochs, AP/F1/precision/recall/AUROC, text and image macro-F1, unequal-label
macro-F1s, critical image precision and recall with support, swap results, paired control
delta, and any errors or incomplete runs.

## Azərbaycanca göndəriləcək mesaj

Salam! Seed-42 nəticəsində warning checkpoint-in AP-si bir az artdı, amma
`affected_individuals` şəkillərinin recall-u 35/87-dən 24/87-yə düşdü. Buna görə
yalnız şəkil loss-u üçün yüngül class weighting sınayırıq. `fix/overfitting-analysis`
branch-in son versiyasını götür və `GPU_IMPROVEMENT_RUNBOOK.md` faylındakı
“Seed 42 only” əmrlərini işə sal. Birinci run adi v2 fusion, ikinci run
`--image-class-weighting capped-sqrt` variantıdır. Hər ikisini eyni GPU-da işlət.
Hər iki checkpoint-in validation nəticələrini və müqayisə hesabatını göndər.
Checkpoint-ləri özündə saxla. Test split-i işə salma, əvvəlki nəticələri dəyişmə
və 43/44 seed-lərinə keçməzdən əvvəl pilotu birlikdə yoxlayaq.
