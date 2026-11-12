"""
GBDT + Viterbi Possession Pipeline
===================================
More Than A Hack 2027

Complete pipeline: data → features → GBDT → Viterbi → submission.

Usage:
    python -m pipeline.run_pipeline

    # Or step by step:
    python -m pipeline.features          # Step 1: extract features
    python -m pipeline.gbdt_model        # Step 2: train GBDT + predict emissions
    python -m pipeline.viterbi           # Step 3: Viterbi decode + write submission

    # Score the result:
    python evaluation/run_evaluation.py
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.features import build_features
from pipeline.gbdt_model import (
    train_and_save, predict_emissions, load_model, OUTPUT_DIR, FEATURES_PATH,
)
from pipeline.viterbi import (
    build_transition_matrix, decode_match,
    remove_short_segments, labels_to_submission,
    write_submission,
)


def run_pipeline(
    # Transition matrix parameters — THE MAIN TUNING KNOBS
    stay_possession=0.998,
    possession_to_dead=0.001,
    possession_to_other=0.001,
    stay_dead=0.990,
    dead_to_home=0.005,
    dead_to_away=0.005,
    # Post-processing
    min_segment_frames=5,
    # Output
    submission_name="weak_predictions",
    skip_features=False,
    skip_training=False,
):
    """Run the full pipeline."""

    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()

    # ================================================================
    # STEP 1: Feature extraction
    # ================================================================
    feat_path = FEATURES_PATH

    if skip_features and os.path.exists(feat_path):
        print("=" * 60)
        print("STEP 1: Loading cached features")
        print("=" * 60)
        features_df = pd.read_parquet(feat_path)
    else:
        print("=" * 60)
        print("STEP 1: Extracting features from ATD + Smart Tagging")
        print("=" * 60)
        features_df = build_features(feat_path)

    t1 = time.time()
    print(f"\nStep 1 completed in {t1 - t0:.1f}s\n")

    # ================================================================
    # STEP 2: GBDT training + emission prediction
    # ================================================================
    emissions_path = os.path.join(output_dir, "emissions.npy")

    if skip_training:
        print("=" * 60)
        print("STEP 2: Loading model trained on automatic labels")
        print("=" * 60)
        model, feature_names, training_info = load_model(output_dir)
    else:
        print("=" * 60)
        print("STEP 2: Training GBDT model")
        print("=" * 60)
        model, feature_names, training_info = train_and_save(features_df, output_dir)

    # Predict again even when reusing the model, so changed features cannot use stale emissions.
    print("\nPredicting emission probabilities for all frames...")
    emissions = predict_emissions(model, features_df, feature_names)
    np.save(emissions_path, emissions)

    t2 = time.time()
    print(f"\nStep 2 completed in {t2 - t1:.1f}s\n")

    # ================================================================
    # STEP 3: Viterbi decoding
    # ================================================================
    print("=" * 60)
    print("STEP 3: Viterbi decoding")
    print("=" * 60)

    log_trans = build_transition_matrix(
        stay_possession=stay_possession,
        possession_to_dead=possession_to_dead,
        possession_to_other=possession_to_other,
        stay_dead=stay_dead,
        dead_to_home=dead_to_home,
        dead_to_away=dead_to_away,
    )

    # Uniform priors: no kickoff team is copied from ground truth.
    labels = decode_match(emissions, log_trans)

    # Post-process
    print(f"\nPost-processing: removing segments < {min_segment_frames} frames...")
    labels = remove_short_segments(labels, min_frames=min_segment_frames)

    # Convert to string labels
    label_strings = labels_to_submission(labels)

    t3 = time.time()
    print(f"\nStep 3 completed in {t3 - t2:.1f}s\n")

    # ================================================================
    # WRITE SUBMISSION
    # ================================================================
    sub_path = os.path.join(ROOT, "evaluation", "results",
                            f"{submission_name}.csv")
    write_submission(label_strings, sub_path)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time: {t3 - t0:.1f}s")
    print(f"Submission: {sub_path}")
    print(f"Training label source: {training_info['label_source']}")
    print("Ground truth was not used. Run the evaluator separately for F1 scores.")
    print(f"\nLabel distribution:")
    for label in ["H", "A", "D", "X"]:
        count = np.sum(label_strings == label)
        print(f"  {label}: {count:>7}")

    # Count possessions (H/A segments)
    n_possessions = 0
    for i in range(1, len(label_strings)):
        if label_strings[i] in ("H", "A") and (
            label_strings[i] != label_strings[i - 1]
            or label_strings[i - 1] == "D"
        ):
            n_possessions += 1
    print(f"\nEstimated possessions: ~{n_possessions}")

    print(f"\nScore it:")
    print(f"  python evaluation/run_evaluation.py")

    return label_strings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GBDT + Viterbi possession pipeline"
    )
    parser.add_argument("--stay-possession", type=float, default=0.998,
                        help="Self-transition prob for H/A states (default: 0.998)")
    parser.add_argument("--stay-dead", type=float, default=0.990,
                        help="Self-transition prob for D state (default: 0.990)")
    parser.add_argument("--min-segment", type=int, default=5,
                        help="Min frames for a possession segment (default: 5)")
    parser.add_argument("--name", default="weak_predictions",
                        help="Submission file name (default: weak_predictions)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Use cached features if available")
    parser.add_argument("--skip-training", action="store_true",
                        help="Require a saved automatic-label model; predict fresh emissions")

    args = parser.parse_args()

    run_pipeline(
        stay_possession=args.stay_possession,
        stay_dead=args.stay_dead,
        min_segment_frames=args.min_segment,
        submission_name=args.name,
        skip_features=args.skip_features,
        skip_training=args.skip_training,
    )

