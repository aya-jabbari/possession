"""Fit LightGBM on automatic tracking/Smart Tagging labels only.

Ground truth is reserved for evaluation/run_evaluation.py. Old supervised
artifacts in pipeline/output are preserved; new artifacts live in output/weak.
"""
import hashlib
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from pipeline.weak_labels import LABEL_SOURCE, RULES, build_weak_labels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(ROOT, "pipeline", "output", "features.parquet")
OUTPUT_DIR = os.path.join(ROOT, "pipeline", "output", "weak")
LABEL_MAP = {"H": 0, "A": 1, "D": 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


def train_gbdt(features_df, weak_labels):
    """Fit fixed parameters on pseudo-labels; never accept a ground-truth path."""
    if not features_df.index.equals(weak_labels.index):
        raise ValueError("Features and weak labels must have identical frame indices")
    eligible = weak_labels[weak_labels["label"] >= 0]
    counts = eligible["label"].value_counts()
    if any(counts.get(label, 0) < 20 for label in LABEL_INV):
        raise ValueError(f"Need at least 20 automatic examples of each class; got {counts.to_dict()}")
    names = list(features_df.columns)
    rng = np.random.default_rng(42)
    selected = np.sort(np.concatenate([
        rng.choice(eligible.index[eligible["label"] == label].to_numpy(),
                   size=min(int(counts[label]), 20000), replace=False)
        for label in LABEL_INV
    ]))
    targets = weak_labels.loc[selected]
    params = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 50,
        "lambda_l2": 1.0, "feature_fraction": 0.8,
        "num_threads": 4, "seed": 42, "deterministic": True,
        "force_col_wise": True, "verbosity": -1,
    }
    print(f"Training from scratch on {len(selected)} automatic labels (no ground truth)")
    data = lgb.Dataset(features_df.loc[selected, names].to_numpy(dtype=np.float32),
                       label=targets["label"].to_numpy(), weight=targets["weight"].to_numpy(),
                       feature_name=names)
    model = lgb.train(params, data, num_boost_round=300)
    info = {
        "label_source": LABEL_SOURCE, "uses_ground_truth": False,
        "rules": RULES, "seed": 42, "num_boost_round": 300, "parameters": params,
        "eligible_counts": {LABEL_INV[int(k)]: int(v) for k, v in counts.items()},
        "training_counts": {LABEL_INV[int(k)]: int(v) for k, v in targets["label"].value_counts().items()},
        "unknown_frames": int((weak_labels["label"] < 0).sum()),
        "rule_counts": {k: int(v) for k, v in weak_labels["rule"].value_counts().items()},
        "selected_frames_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
        "weak_labels_sha256": hashlib.sha256(weak_labels["label"].to_numpy().tobytes()).hexdigest(),
        "validation": "No ground-truth-based fitting, early stopping or parameter search.",
    }
    return model, names, info


def predict_emissions(model, features_df, feature_names):
    """Predict columns P(H), P(A), P(D) in the saved feature order."""
    emissions = model.predict(features_df[feature_names].to_numpy(dtype=np.float32)).astype(np.float32)
    if emissions.shape != (len(features_df), 3) or not np.isfinite(emissions).all():
        raise ValueError("Invalid emission probabilities")
    if ((emissions < 0) | (emissions > 1)).any() or not np.allclose(emissions.sum(axis=1), 1):
        raise ValueError("Emission probabilities must be in [0, 1] and sum to one")
    print(f"Predicted emissions for {len(emissions)} frames")
    return emissions


def save_model(model, feature_names, training_info, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_model(os.path.join(output_dir, "gbdt_model.txt"))
    with open(os.path.join(output_dir, "gbdt_meta.json"), "w", encoding="utf-8") as handle:
        json.dump({"feature_names": feature_names, "label_map": LABEL_MAP,
                   "training": training_info}, handle, indent=2)
    print(f"Model saved to {output_dir}")


def load_model(output_dir):
    """Reject old models whose training used ground truth or lacks provenance."""
    with open(os.path.join(output_dir, "gbdt_meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    info = meta.get("training", {})
    if info.get("uses_ground_truth") is not False or info.get("label_source") != LABEL_SOURCE:
        raise ValueError("This is not a ground-truth-free model. Retrain with pipeline.run_pipeline.")
    model = lgb.Booster(model_file=os.path.join(output_dir, "gbdt_model.txt"))
    names = meta["feature_names"]
    if names != model.feature_name() or meta.get("label_map") != LABEL_MAP:
        raise ValueError("Model feature names or label encoding disagree with metadata")
    return model, names, info


def train_and_save(features_df, output_dir=OUTPUT_DIR):
    weak_labels = build_weak_labels(features_df)
    model, names, info = train_gbdt(features_df, weak_labels)
    save_model(model, names, info, output_dir)
    weak_labels.to_parquet(os.path.join(output_dir, "weak_labels.parquet"))
    return model, names, info


if __name__ == "__main__":
    from pipeline.features import build_features
    features = (pd.read_parquet(FEATURES_PATH) if os.path.exists(FEATURES_PATH)
                else build_features(FEATURES_PATH))
    model, names, _ = train_and_save(features)
    np.save(os.path.join(OUTPUT_DIR, "emissions.npy"), predict_emissions(model, features, names))
