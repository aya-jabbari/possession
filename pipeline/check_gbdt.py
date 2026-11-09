"""Check saved GBDT artifacts, optionally retraining on a chronological split.

Run from the repository root:
    python -m pipeline.check_gbdt
    python -m pipeline.check_gbdt --holdout

The holdout model is temporary; existing models and emissions are preserved.
"""

import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from games import GAME, GAMES
from evaluation import possession_metrics as pm
from pipeline.gbdt_model import (
    load_ground_truth, load_model, predict_emissions, save_model, train_gbdt,
)
from pipeline.viterbi import (
    build_transition_matrix, remove_short_segments, viterbi_vectorised,
)


def check_probabilities(values, n_frames):
    if values.shape != (n_frames, 3):
        raise ValueError(f"Expected {(n_frames, 3)} probabilities, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinity")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions are outside [0, 1]")
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-6)


def check_saved(output_dir):
    features = pd.read_parquet(output_dir / "features.parquet")
    model, names, scores = load_model(str(output_dir))
    n_frames = GAMES[GAME]["n_video_frames"]
    if not np.array_equal(features.index.to_numpy(), np.arange(n_frames)):
        raise ValueError("Feature index must contain every video frame in order")
    if len(names) != model.num_feature() or len(set(names)) != len(names):
        raise ValueError("Model and metadata feature counts disagree")
    if not features.columns.is_unique:
        raise ValueError("Feature columns contain duplicates")
    if np.isinf(features[names].to_numpy(dtype=np.float32)).any():
        raise ValueError("Features contain infinity (NaN is supported by LightGBM)")
    predictions = predict_emissions(model, features, names)
    check_probabilities(predictions, n_frames)
    cached = np.load(output_dir / "emissions.npy", allow_pickle=False)
    check_probabilities(cached, n_frames)
    np.testing.assert_allclose(cached, predictions, rtol=1e-6, atol=1e-7)
    frames, labels = load_ground_truth()
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Unexpected ground-truth labels")
    print("PASS: model loads, feature rows align, probabilities sum to 1,")
    print("      and cached emissions match fresh predictions.")
    print(f"Saved random-frame CV macro F1: {scores.get('cv_macro_f1', 'N/A')}")
    in_sample = f1_score(labels, predictions[frames].argmax(axis=1),
                         labels=[0, 1, 2], average="macro", zero_division=0)
    print(f"Training-window macro F1: {in_sample:.4f} (in-sample; not a test score)")
    return features, frames, labels


def check_holdout(features, frames, labels, output_dir):
    # Keep the final third unseen. The 3s gap is longer than the largest
    # centered rolling window (75 frames), preventing overlap at the split.
    fps = GAMES[GAME]["fps"]
    split = len(frames) * 2 // 3
    gap = 3 * fps
    train_frames, train_labels = frames[:split], labels[:split]
    test_mask = frames >= frames[split] + gap
    test_frames, test_labels = frames[test_mask], labels[test_mask]
    if not len(test_frames) or len(np.unique(train_labels)) != 3:
        raise ValueError("Holdout requires training examples for all classes and test frames")
    if not np.all(np.diff(frames) == 1):
        raise ValueError("This holdout decoder expects a contiguous public GT window")
    print(f"\nHoldout: {len(train_frames)} training frames, {gap}-frame gap,")
    print(f"         {len(test_frames)} unseen test frames.")
    print("Only the training portion is passed to train_gbdt.")
    model, names, scores = train_gbdt(features, train_frames, train_labels)
    test_features = features.loc[test_frames]
    probabilities = predict_emissions(model, test_features, names)
    check_probabilities(probabilities, len(test_frames))

    # Exercise the actual save/load functions without replacing user artifacts.
    with tempfile.TemporaryDirectory(prefix="gbdt-check-", dir=output_dir) as temp:
        save_model(model, names, scores, temp)
        reloaded, reloaded_names, _ = load_model(temp)
        restored = predict_emissions(reloaded, test_features, reloaded_names)
        np.testing.assert_allclose(restored, probabilities, rtol=1e-6, atol=1e-7)
    print("PASS: training and save/load prediction round trip.")
    predictions = probabilities.argmax(axis=1)
    print("\nUNSEEN HOLDOUT RESULTS (GBDT):")
    print(classification_report(test_labels, predictions, labels=[0, 1, 2],
                                target_names=["H", "A", "D"], digits=4,
                                zero_division=0))
    decoded = remove_short_segments(
        viterbi_vectorised(probabilities, build_transition_matrix()), min_frames=5,
    )
    result = pm.evaluate(test_labels, decoded)
    print(f"GBDT + Viterbi holdout macro F1: {result['macro_f1']:.4f}")
    print(f"GBDT + Viterbi holdout spell F1: {result['spell_f1_mean']:.4f}")
    print(f"Holdout possessions: {result['n_gt_spells']} true, "
          f"{result['n_pred_spells']} predicted")
    print("This small holdout measures one passage of play, not full-match accuracy.")
    print("The inner random-frame CV printed during training is still optimistic.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true",
                        help="also train a temporary model and test the unseen final third")
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parent / "output"
    features, frames, labels = check_saved(output_dir)
    if args.holdout:
        check_holdout(features, frames, labels, output_dir)


if __name__ == "__main__":
    main()
