"""
Trains the XGBoost v3 deployment model (real MediaPipe Hands curl-angle hand
signal replacing v1's crude Pose-fingertip-distance hand-openness feature --
see src/features/handcrafted_v3.py and MODEL_JOURNEY.md section 7b for the
held-out evidence: test 0.98 vs v1's 0.94, wins every leave-participant-out
CV fold, the strongest offline result found so far).

Same convention as train_xgboost_final.py/train_xgboost_final_v2.py: trains
on ALL 642 labeled clips (deployment model, not held-out eval -- that number
already comes from scripts/eval_xgboost_v3_holdout.py). Does NOT touch
xgboost_baseline.json or xgboost_v2.json.

Usage: python scripts/train_xgboost_final_v3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xgboost as xgb
from tqdm import tqdm

from src.config import ACTION_TO_IDX, ROOT
from src.data.dataset import load_labels, _raw_landmarks
from src.data.graph_dataset import _raw_hand_landmarks
from src.data.hand_features import crop_and_resample, finger_curl_series
from src.data.preprocess import preprocess_clip
from src.features.handcrafted_v3 import extract_handcrafted_features_v3

MODEL_DIR = ROOT / "models"


def build_matrix(df):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting v3 features"):
        raw = _raw_landmarks(row)
        sample = preprocess_clip(raw["xyz"], raw["visibility"])

        hand_raw = _raw_hand_landmarks(row)
        curl, _presence = finger_curl_series(hand_raw)
        hand_curl = crop_and_resample(curl, None, sample["xyz"].shape[0])

        X.append(extract_handcrafted_features_v3(sample["xyz"], hand_curl))
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

    out_path = MODEL_DIR / "xgboost_v3.json"
    model.save_model(out_path)
    print(f"saved -> {out_path}")

    # novelty/OOD gate calibration -- same method as v1/v2 (see
    # train_xgboost_final.py for the full rationale)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xz = (X - mean) / std

    from scipy.spatial.distance import cdist
    dists = cdist(Xz, Xz)
    np.fill_diagonal(dists, np.inf)
    nn_dist = dists.min(axis=1)
    novelty_threshold = float(np.percentile(nn_dist, 99))

    feat_path = MODEL_DIR / "xgboost_v3_train_features.npz"
    np.savez_compressed(
        feat_path, X=Xz, mean=mean, std=std, novelty_threshold=novelty_threshold,
    )
    print(f"saved -> {feat_path} (novelty gate, threshold={novelty_threshold:.3f})")

    train_acc = (model.predict(X) == y).mean()
    print(f"train accuracy (not held-out -- sanity check only): {train_acc:.3f}")

    importances = model.feature_importances_
    print(f"real-curl hand-openness feature importance: {importances[4]:.4f}  "
          f"rank {sorted(importances, reverse=True).index(importances[4]) + 1} of {len(importances)}")


if __name__ == "__main__":
    main()
