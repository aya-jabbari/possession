"""Check automatic-label model artifacts without reading ground truth.

Run: python -m pipeline.check_gbdt
For accuracy, separately run: python evaluation/run_evaluation.py --no-detail
"""
from pathlib import Path
import numpy as np
import pandas as pd
from games import GAME, GAMES
from pipeline.gbdt_model import FEATURES_PATH, OUTPUT_DIR, load_model, predict_emissions
from pipeline.weak_labels import build_weak_labels


def check_probabilities(values, n_frames):
    if values.shape != (n_frames, 3):
        raise ValueError(f"Expected {(n_frames, 3)} probabilities, got {values.shape}")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must contain finite probabilities in [0, 1]")
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-6)


def check_saved(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    features = pd.read_parquet(FEATURES_PATH)
    model, names, info = load_model(output_dir)
    n_frames = GAMES[GAME]['n_video_frames']
    if not np.array_equal(features.index.to_numpy(), np.arange(n_frames)):
        raise ValueError("Feature index must contain every video frame in order")
    if not features.columns.is_unique or np.isinf(features[names].to_numpy(dtype=np.float32)).any():
        raise ValueError("Features contain duplicate names or infinity")
    predictions = predict_emissions(model, features, names)
    check_probabilities(predictions, n_frames)
    cached = np.load(output_dir / 'emissions.npy', allow_pickle=False)
    check_probabilities(cached, n_frames)
    np.testing.assert_allclose(cached, predictions, rtol=1e-6, atol=1e-7)
    saved_labels = pd.read_parquet(output_dir / 'weak_labels.parquet')
    pd.testing.assert_frame_equal(saved_labels, build_weak_labels(features))
    print('PASS: model provenance, frame alignment, automatic labels and emission probabilities.')
    print('Training examples:', info['training_counts'])
    print('Ground truth was not loaded. Evaluate with evaluation/run_evaluation.py.')
    return features


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_saved()
