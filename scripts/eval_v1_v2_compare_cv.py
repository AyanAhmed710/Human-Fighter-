"""
Direct v1 vs v2 comparison across MULTIPLE different held-out test participant
groups (not just the one locked test=[2,3] pair from eval_xgboost_v2_holdout.py) --
answers "was the v2 test-accuracy drop specific to participants 2,3, or does it
hold across different choices of who's held out?"

Uses the same leave-participant-out group k-fold split as
src.evaluate.group_kfold_splits (3 folds, GroupKFold over participant_id) so
every participant gets to be in a held-out group exactly once across the 3
folds -- a systematic sweep of "different participant for test", not one more
arbitrary single pick. Both v1 (15-feature) and v2 (16-feature, +elbow) models
are fit fresh per fold on the SAME train/held split so the comparison is
apples-to-apples per fold, not just pooled CV mean vs pooled CV mean.

Usage: python scripts/eval_v1_v2_compare_cv.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xgboost as xgb
from sklearn.metrics import recall_score
from tqdm import tqdm

from src.config import ACTION_TO_IDX, ACTIONS
from src.data.dataset import load_labels, _raw_landmarks
from src.data.preprocess import preprocess_clip
from src.evaluate import group_kfold_splits
from src.features.handcrafted import extract_handcrafted_features
from src.features.handcrafted_v2 import extract_handcrafted_features_v2


def build_matrices(df):
    X1, X2, y = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting v1+v2 features"):
        raw = _raw_landmarks(row)
        sample = preprocess_clip(raw["xyz"], raw["visibility"])
        X1.append(extract_handcrafted_features(sample["xyz"]))
        X2.append(extract_handcrafted_features_v2(sample["xyz"]))
        y.append(ACTION_TO_IDX[row["action"]])
    return np.stack(X1), np.stack(X2), np.array(y)


def new_model():
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="multi:softmax", num_class=3, random_state=42,
    )


def main():
    df = load_labels()
    X1_all, X2_all, y_all = build_matrices(df)
    groups_all = df["participant_id"].values
    punch_idx = ACTION_TO_IDX["punching"]

    print(f"\n{'held-out participants':<24}{'v1 acc':>8}{'v2 acc':>8}{'v1 punch recall':>18}{'v2 punch recall':>18}")
    v1_accs, v2_accs = [], []
    for fold, (train_idx, held_idx) in enumerate(group_kfold_splits(df, n_splits=3)):
        held_participants = sorted(set(groups_all[held_idx]))

        m1 = new_model()
        m1.fit(X1_all[train_idx], y_all[train_idx])
        pred1 = m1.predict(X1_all[held_idx])
        acc1 = (pred1 == y_all[held_idx]).mean()
        rec1 = recall_score(y_all[held_idx], pred1, labels=[punch_idx], average="micro")

        m2 = new_model()
        m2.fit(X2_all[train_idx], y_all[train_idx])
        pred2 = m2.predict(X2_all[held_idx])
        acc2 = (pred2 == y_all[held_idx]).mean()
        rec2 = recall_score(y_all[held_idx], pred2, labels=[punch_idx], average="micro")

        print(f"{str(held_participants):<24}{acc1:>8.3f}{acc2:>8.3f}{rec1:>18.3f}{rec2:>18.3f}")
        v1_accs.append(acc1)
        v2_accs.append(acc2)

    print(f"\n{'CV mean':<24}{np.mean(v1_accs):>8.3f}{np.mean(v2_accs):>8.3f}")
    print(f"{'CV std':<24}{np.std(v1_accs):>8.3f}{np.std(v2_accs):>8.3f}")


if __name__ == "__main__":
    main()
