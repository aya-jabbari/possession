# Possession model: ground truth is evaluation-only

Run these commands from the `possession` repository root in PowerShell.
Use `python -X utf8` on Windows so diagnostic arrows print correctly.

## Current model

The default pipeline trains a NEW LightGBM model from automatic labels generated
from tracking and Smart Tagging. It does not read ground-truth CSVs, fit on their
labels, select parameters using them, or use ground-truth kickoff teams.

- `H` / `A`: visible ball near a player, with a clear distance advantage over
  opponents. Offensive Smart Tags can support a looser geometry threshold.
- `D`: visible ball clearly outside the pitch, or stationary and unattended for
  at least two seconds. Missing ball tracking alone never creates a `D` target.
- Uncertain frames: excluded from training; the fitted model still predicts them.

These are noisy pseudo-labels, not manual annotations or true possession labels.
Thresholds and model parameters are fixed heuristics. There is no ground-truth
cross-validation or early stopping. Training uses up to 20,000 examples per class,
selected with seed 42. This is weak supervision on the supplied match, not a
separate-match generalization experiment. All match features can be used; the
reserved answer-key labels are used only by the evaluator.

New artifacts are isolated in `pipeline/output/weak/`:

- `gbdt_model.txt`: newly trained model.
- `gbdt_meta.json`: feature order, label encoding, training source, rules and counts.
- `weak_labels.parquet`: automatic targets, confidence weights and rule per frame.
- `emissions.npy`: per-frame probabilities in `[H, A, D]` order.

The shared feature cache is `pipeline/output/features.parquet`, generated only
from the provided tracking and Smart Tagging inputs.

## Rebuild and generate predictions

Install dependencies:

```powershell
python -m pip install -r requirements-pipeline.txt
```

Rebuild features, train from scratch and generate the complete match CSV:

```powershell
python -X utf8 -m pipeline.run_pipeline --name weak_predictions
```

If the feature cache is current, skip feature extraction:

```powershell
python -X utf8 -m pipeline.run_pipeline --skip-features --name weak_predictions
```

To reuse the NEW model and generate a CSV named after your team:

```powershell
python -X utf8 -m pipeline.run_pipeline --skip-features --skip-training --name team_name
python -X utf8 evaluation/validate_submission.py evaluation/results/team_name.csv
```

`--skip-training` requires a model with automatic-label training provenance in
`output/weak`; it never falls back to the old model or silently retrains. Fresh
probabilities are computed even when reusing the model, avoiding stale emissions.
Viterbi starts each half with a uniform prior and removes very short possessions.

The equivalent individual steps are:

```powershell
python -X utf8 -m pipeline.features
python -X utf8 -m pipeline.gbdt_model
python -X utf8 -m pipeline.viterbi
```

## Verify without reading ground truth

```powershell
python -X utf8 -m unittest pipeline.test_weak_labels -v
python -X utf8 -m pipeline.check_gbdt
python -X utf8 evaluation/validate_submission.py evaluation/results/weak_predictions.csv
```

The artifact checker verifies model provenance, frame alignment, reproducible
weak targets, valid probability values, and agreement with saved emissions.
It never reads the ground truth. Training/save/load and generation were also
verified locally with file access to `evaluation/gt/` and old supervised model
artifacts blocked throughout the run.

## Evaluate separately

Only after training and generation, run:

```powershell
python -X utf8 evaluation/run_evaluation.py --no-detail
```

Read the `weak_predictions` (or your team name) row. The ranking metric is spell
F1; macro F1 measures per-frame classification. The evaluator reads the public
five-minute answer key, which the new model did not use as training labels.

Earlier supervised experiments already inspected this passage, so it is not a
pristine never-seen test set for the development process. Do not repeatedly tune
this new model against its score and still call that an independent test. The
public passage is small and cannot establish full-match accuracy.

uFirst evaluation after retraining on 2026-11-09 (39,196 sampled automatic labels):

| New automatic-label model | Score |
| --- | ---: |
| Public-window macro F1 | 0.6096 |
| Public-window spell F1 | 0.0207 |

The file passes format validation, but the ranking score is poor: the model
predicts 118 possessions in the public window, versus 17 true possessions.
It predicts 1,705 possessions across the full match. It is a working retraining
baseline, not a demonstrated improvement over the supplied smoothed baselines.
No changes were selected against these evaluation scores after this run.

## Historical supervised artifacts

The files directly in `pipeline/output/gbdt_model.txt`, `gbdt_meta.json`, and
`emissions.npy`, plus the old `my_predictions.csv`, `gbdt_checked.csv`, and
`gbdt_viterbi.csv`, come from the PREVIOUS model trained on public ground truth.
They are preserved for reference and are not loaded by the new pipeline.
Their perfect public-window scores are training-data fit, not independent tests.

The following measurements belong only to that previous supervised approach:

| Historical measurement | Score |
| --- | ---: |
| Random-frame CV macro F1 (optimistic neighboring-frame split) | 0.9977 |
| Chronological holdout GBDT macro F1 | 0.5821 |
| Chronological holdout GBDT + Viterbi macro F1 | 0.5961 |
| Chronological holdout GBDT + Viterbi spell F1 | 0.3667 |

That historical holdout trained on frames 11185-16184, skipped 16185-16259 and
tested 16260-18684. The old `check_gbdt --holdout` training mode has been removed
so the checker cannot train on the reserved ground-truth labels.
