"""
v2 of scripts/train_xgboost_final.py -- adds the starting-elbow-angle
feature (src/features/handcrafted_v2.py) on top of the original 15.
Completely separate script/model/output paths from the original so
models/xgboost_baseline.json and scripts/train_xgboost_final.py are
untouched -- this is an additional artifact, not a replacement.

Usage: python scripts/train_xgboost_final_v2.py
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
from src.features.handcrafted_v2 import extract_handcrafted_features_v2

MODEL_DIR = ROOT / "models"


def build_matrix(df):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting features (v2)"):
        raw = _raw_landmarks(row)
        sample = preprocess_clip(raw["xyz"], raw["visibility"])
        X.append(extract_handcrafted_features_v2(sample["xyz"]))
        y.append(ACTION_TO_IDX[row["action"]])
    return np.stack(X), np.array(y)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labels()
    print(f"training on all {len(df)} labeled clips (deployment model, not held-out eval) -- v2 features")

    X, y = build_matrix(df)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="multi:softmax", num_class=3, random_state=42,
    )
    model.fit(X, y)

    out_path = MODEL_DIR / "xgboost_v2.json"
    model.save_model(out_path)
    print(f"saved -> {out_path}")

    # --- novelty/OOD gate calibration (src/inference.py), same method as v1 ---
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xz = (X - mean) / std

    from scipy.spatial.distance import cdist
    dists = cdist(Xz, Xz)
    np.fill_diagonal(dists, np.inf)
    nn_dist = dists.min(axis=1)
    novelty_threshold = float(np.percentile(nn_dist, 99))

    feat_path = MODEL_DIR / "xgboost_v2_train_features.npz"
    np.savez_compressed(
        feat_path, X=Xz, mean=mean, std=std, novelty_threshold=novelty_threshold,
    )
    print(f"saved -> {feat_path} (novelty gate, threshold={novelty_threshold:.3f})")

    train_acc = (model.predict(X) == y).mean()
    print(f"train accuracy (not held-out -- sanity check only): {train_acc:.3f}")

    # feature importance -- confirm the new feature (index 15, last column)
    # actually earns real weight, not just added noise
    importances = model.feature_importances_
    print(f"\nstarting-elbow-angle feature importance: {importances[15]:.4f} "
          f"(rank {int((-importances).argsort().tolist().index(15)) + 1} of {len(importances)})")


if __name__ == "__main__":
    main()
