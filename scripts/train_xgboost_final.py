"""
Trains the XGBoost sanity-baseline on ALL labeled clips (not just the locked
train split) and saves it for real-world/live inference use.

This is a deliberate departure from the held-out evaluation split: for
deployment testing you want the model to have seen as much of your own
motion data as possible, not to be evaluated. Held-out accuracy numbers
(94% test, 92%+-2.5% CV) already come from src/ notebooks -- this script
produces the actual artifact used by scripts/live_inference.py.

Usage: python scripts/train_xgboost_final.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xgboost as xgb
from tqdm import tqdm

from src.config import ACTION_TO_IDX, ROOT
from src.data.dataset import load_labels, _raw_landmarks
from src.data.preprocess import preprocess_clip
from src.features.handcrafted import extract_handcrafted_features

MODEL_DIR = ROOT / "models"


def build_matrix(df):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting features"):
        raw = _raw_landmarks(row)
        sample = preprocess_clip(raw["xyz"], raw["visibility"])
        X.append(extract_handcrafted_features(sample["xyz"]))
        y.append(ACTION_TO_IDX[row["action"]])
    return np.stack(X), np.array(y)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labels()
    print(f"training on all {len(df)} labeled clips (deployment model, not held-out eval)")

    X, y = build_matrix(df)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="multi:softmax", num_class=3, random_state=42,
    )
    model.fit(X, y)

    out_path = MODEL_DIR / "xgboost_baseline.json"
    model.save_model(out_path)
    print(f"saved -> {out_path}")

    # --- novelty/OOD gate calibration (src/inference.py) ---
    # Softmax confidence can't reject "not one of these 3 actions at all"
    # (waving, walking, anything) -- the model has no such class and will
    # always pick its closest match. A distance-based novelty check doesn't
    # need to have seen every possible non-action; it just measures whether
    # a live capture statistically resembles real training data at all.
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xz = (X - mean) / std  # standardize -- raw dims are on wildly different
                            # scales (speed ~0.02-0.5, ROM in degrees ~tens),
                            # distance would otherwise just measure ROM

    # calibrate the rejection threshold from the training data's own spread:
    # for each real clip, its distance to the nearest *other* real clip
    # (leave-one-out) -- a live capture farther than the 99th percentile of
    # that natural spread doesn't look like real data at all
    from scipy.spatial.distance import cdist
    dists = cdist(Xz, Xz)
    np.fill_diagonal(dists, np.inf)
    nn_dist = dists.min(axis=1)
    novelty_threshold = float(np.percentile(nn_dist, 99))

    feat_path = MODEL_DIR / "xgboost_train_features.npz"
    np.savez_compressed(
        feat_path, X=Xz, mean=mean, std=std, novelty_threshold=novelty_threshold,
    )
    print(f"saved -> {feat_path} (novelty gate, threshold={novelty_threshold:.3f})")

    train_acc = (model.predict(X) == y).mean()
    print(f"train accuracy (not held-out -- sanity check only): {train_acc:.3f}")


if __name__ == "__main__":
    main()
