# Validation under recording and operating-condition shift

The README's 98–100% scores come from historical random-window evaluation.
`load_seu.py` uses 50% overlapping windows; `train.py` shuffles windows rather
than source recordings. Such evaluation can reuse signal content and recording
conditions across train/test. It is not evidence for unseen-recording performance.

The [historical LOSO audit](model-audit-2026-05.md) already reports the loss of
accuracy across speed regimes. Its values were not rerun in this contribution.

## Reproduce a controlled comparison

From the repository root, with NumPy, SciPy and scikit-learn installed:

```bash
PYTHONPATH=python python python/validation_benchmark.py \
  --data-dir python/data/gear --max-windows-per-file 100 \
  --output reports/gear-validation.json
PYTHONPATH=python python -m pytest tests/test_validation_benchmark.py -q
```

The command uses one fixed Random Forest baseline and reports accuracy,
balanced accuracy, macro-F1 and confusion matrices for:

1. Random-window holdout, explicitly diagnostic and potentially dependent.
2. A complete source-file holdout per class, with class coverage checks.
3. Leave-one-speed-out with source-file disjointness checks.

File assignments apply to every window from the file. No imputation, feature
selection, scaling or model tuning reads test data. Parsing errors, missing
speeds, inadequate class coverage or fewer than two files per class fail with
an error; they never trigger fallback to random validation. No fitted browser
model is changed. The report records source-file SHA-256 and the window limit.
Exact duplicate file contents are rejected even if the filenames differ. The
selected signal channel is recorded per file. Feature extraction uses the same
fixed reference frequencies for all files (20 Hz rotation, 400 Hz gear mesh,
5120 Hz sampling), including the 30 Hz recordings. The speed-shift experiment
therefore measures this fixed historical feature basis; it does not claim the
best possible speed-conditioned feature engineering.

`--max-windows-per-file 0` uses all windows; the default samples evenly spaced
windows from each recording to keep the first comparison affordable. Do not
compare these numbers to a differently sized historical experiment as an
isolated model improvement. Repeated use of this benchmark becomes development.

## Remaining limitations

Source-file separation is a minimum. SEU filenames do not prove physical
bearing/session independence. A future dataset with bearing IDs should use
bearing-level groups and independent operating conditions. With few recordings,
thousands of windows do not yield thousands of independent observations.

This contribution verifies split invariants with fixtures. A new full-data
benchmark has not been executed; no new SEU scores are claimed.
