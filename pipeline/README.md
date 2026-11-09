# Testing the possession model

Run these commands from the `possession` repository root in PowerShell.

Install the pipeline dependencies (the basic requirements only cover evaluation):

```powershell
python -m pip install -r requirements-pipeline.txt
```

Check the saved model, cached features and emissions:

```powershell
python -m pipeline.check_gbdt
```

This checks feature alignment, model loading, probability shape/range/sums, and
agreement between freshly predicted and cached emissions. Missing artifacts mean
you need to build features and train first.

Test on unseen frames:

```powershell
python -m pipeline.check_gbdt --holdout
```

This trains with the existing `train_gbdt` function on the first two thirds of
the public window, skips a 3-second gap, and scores the final third. It prints
per-class precision/recall/F1 for the GBDT and macro F1/spell F1 after Viterbi.
Training and save/load are checked using a temporary model; saved production
artifacts are preserved. Read the final **holdout** results: the inner random-frame
CV printed by `train_gbdt` is still optimistic because neighboring frames are
very similar. The holdout contains only a few possessions, so it cannot establish
full-match accuracy.

Verified locally on 2026-11-09 with the supplied tracking and saved artifacts:

| Measurement | Score |
| --- | ---: |
| Saved random-frame CV macro F1 | 0.9977 |
| Chronological holdout GBDT macro F1 | 0.5821 |
| Chronological holdout GBDT + Viterbi macro F1 | 0.5961 |
| Chronological holdout GBDT + Viterbi spell F1 | 0.3667 |

The holdout trains on frames 11185–16184, skips frames 16185–16259, and tests
frames 16260–18684 (97 seconds, four true possessions; eight predicted).
These are different evaluation procedures, not directly interchangeable scores.
Feature regeneration matched the cache, saved emissions matched fresh model
predictions, training/save/load passed, and the full submission passed validation.
Feature extraction emits warnings for all-missing tracking rows; the resulting
NaN features are accepted by LightGBM and are not invalid prediction values.

Generate a full submission from the saved model/emissions and validate its format:

```powershell
python -m pipeline.run_pipeline --skip-features --skip-training --name gbdt_checked
python evaluation/validate_submission.py evaluation/results/gbdt_checked.csv
python evaluation/run_evaluation.py --no-detail
```

The public scoreboard evaluates the same five minutes used for final training.
Its score is an in-sample fit check, not evidence of performance on unseen play.
The competition ranks on spell F1; inspect that alongside macro F1.

To rebuild everything from the tracking assets in `data/tactical/`:

```powershell
python -m pipeline.run_pipeline
```

This overwrites `pipeline/output/` features, model, metadata and emissions, and
generates `evaluation/results/gbdt_viterbi.csv`. Use `--skip-features` if cached
features are already current. Smart Tagging is used as an input feature; the
weak-supervision augmentation mentioned in `gbdt_model.py` is not implemented.
