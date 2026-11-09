"""
Step 2: GBDT Training
=====================
Train a LightGBM multi-class classifier to predict per-frame possession.

The model outputs P(H|frame), P(A|frame), P(D|frame) — these become the
emission probabilities for the Viterbi decoder in step 3.

Training strategy:
  - Train on the 5-minute public ground truth (7,500 graded frames)
  - Use Smart Tagging as weak supervision to augment training on the rest
  - Cross-validate within the public window to get emission quality estimate

Usage:
    python -m pipeline.gbdt_model
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]

# Ground truth paths
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
GT_META_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public_meta.json")

# Label encoding
LABEL_MAP = {"H": 0, "A": 1, "D": 2}
LABEL_INV = {0: "H", 1: "A", 2: "D"}


# ---------------------------------------------------------------------------
# 1. Load ground truth
# ---------------------------------------------------------------------------

def load_ground_truth():
    """Load the public ground truth and return (frames, labels) for graded frames only."""
    gt = pd.read_csv(GT_PATH)

    # Only keep graded frames (non-X labels)
    graded = gt[gt["label"] != "X"].copy()
    graded["label_int"] = graded["label"].map(LABEL_MAP)

    print(f"Ground truth: {len(graded)} graded frames")
    print(f"  H: {(graded['label'] == 'H').sum()}, "
          f"A: {(graded['label'] == 'A').sum()}, "
          f"D: {(graded['label'] == 'D').sum()}")

    return graded["frame"].values, graded["label_int"].values


# ---------------------------------------------------------------------------
# 2. Train GBDT model
# ---------------------------------------------------------------------------

def train_gbdt(features_df, gt_frames, gt_labels, n_folds=5):
    """
    Train a LightGBM multi-class model.

    Args:
        features_df: DataFrame with one row per video frame (index = video_frame)
        gt_frames: array of graded frame indices
        gt_labels: array of label ints (0=H, 1=A, 2=D)
        n_folds: number of CV folds

    Returns:
        model: trained LightGBM model
        feature_names: list of feature column names
        cv_scores: dict of cross-validation scores
    """
    # Select features (exclude non-predictive columns)
    exclude_cols = set()
    feature_names = [c for c in features_df.columns if c not in exclude_cols]

    # Prepare training data
    X_train = features_df.loc[gt_frames, feature_names].values.astype(np.float32)
    y_train = gt_labels

    print(f"\nTraining data: {X_train.shape[0]} samples × {X_train.shape[1]} features")

    # LightGBM parameters — tuned for this problem
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # ---- Cross-validation ----
    print(f"\n{n_folds}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_preds = np.zeros((len(y_train), 3), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        cv_preds[val_idx] = model.predict(X_val)
        fold_pred = np.argmax(cv_preds[val_idx], axis=1)
        fold_f1 = f1_score(y_val, fold_pred, average="macro")
        print(f"  Fold {fold + 1}: macro F1 = {fold_f1:.4f} "
              f"({model.best_iteration} rounds)")

    # Overall CV score
    cv_pred_labels = np.argmax(cv_preds, axis=1)
    cv_macro_f1 = f1_score(y_train, cv_pred_labels, average="macro")
    print(f"\nOverall CV macro F1: {cv_macro_f1:.4f}")
    print("\nPer-class report (CV):")
    print(classification_report(
        y_train, cv_pred_labels,
        target_names=["H", "A", "D"],
        digits=4,
    ))

    # ---- Train final model on ALL ground truth ----
    print("Training final model on all ground truth...")
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=300,  # use a reasonable number, no early stopping here
    )

    # Feature importance
    importance = final_model.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    print("\nTop 15 features by gain:")
    print(imp_df.head(15).to_string(index=False))

    return final_model, feature_names, {"cv_macro_f1": cv_macro_f1}


# ---------------------------------------------------------------------------
# 3. Predict emission probabilities for all frames
# ---------------------------------------------------------------------------

def predict_emissions(model, features_df, feature_names):
    """
    Predict P(H), P(A), P(D) for every frame.

    Returns:
        emissions: (n_frames, 3) array of probabilities
    """
    X_all = features_df[feature_names].values.astype(np.float32)
    emissions = model.predict(X_all).astype(np.float32)
    print(f"Predicted emissions for {emissions.shape[0]} frames")
    return emissions


# ---------------------------------------------------------------------------
# 4. Save/load model
# ---------------------------------------------------------------------------

def save_model(model, feature_names, cv_scores, output_dir):
    """Save model and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "gbdt_model.txt")
    model.save_model(model_path)

    meta = {
        "feature_names": feature_names,
        "cv_scores": cv_scores,
        "label_map": LABEL_MAP,
    }
    meta_path = os.path.join(output_dir, "gbdt_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {output_dir}")


def load_model(output_dir):
    """Load saved model and metadata."""
    model_path = os.path.join(output_dir, "gbdt_model.txt")
    model = lgb.Booster(model_file=model_path)

    meta_path = os.path.join(output_dir, "gbdt_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta["feature_names"], meta["cv_scores"]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pipeline.features import build_features

    output_dir = os.path.join(ROOT, "pipeline", "output")

    # Load or build features
    feat_path = os.path.join(output_dir, "features.parquet")
    if os.path.exists(feat_path):
        print("Loading cached features...")
        features_df = pd.read_parquet(feat_path)
    else:
        features_df = build_features(feat_path)

    # Load ground truth
    gt_frames, gt_labels = load_ground_truth()

    # Train
    model, feature_names, cv_scores = train_gbdt(features_df, gt_frames, gt_labels)

    # Save
    save_model(model, feature_names, cv_scores, output_dir)

    # Predict emissions
    emissions = predict_emissions(model, features_df, feature_names)
    np.save(os.path.join(output_dir, "emissions.npy"), emissions)
    print(f"Saved emissions to {output_dir}/emissions.npy")

