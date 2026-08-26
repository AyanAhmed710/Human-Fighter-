"""
Held-out evaluation of the XGBoost v2 (16-feature, +starting-elbow-angle) model
on the LOCKED participant split -- mirrors notebooks/baseline/01_baseline_bilstm.ipynb
cell 8 (sanity baseline val/test) and cell 10 (leave-participant-out group k-fold
CV), swapped to handcrafted_v2 features. train_xgboost_final_v2.py trains on ALL
642 clips for deployment and only reports train accuracy (0.998, not a real eval
number) -- this script is the missing held-out number for v2, directly comparable
to v1's locked-split numbers (97% val / 94% test / 92%+-2.5% CV) already recorded
in the notebook.

Trains fresh models on train_df only (participants 4,5,6,7,8,9) -- does NOT touch
models/xgboost_v2.json (the all-642-clip deployment model) or xgboost_baseline.json.
Pure evaluation, no artifacts saved.

Usage: python scripts/eval_xgboost_v2_holdout.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.config import ACTION_TO_IDX, ACTIONS
from src.data.dataset import load_labels, split_dataframe, _raw_landmarks
from src.data.preprocess import preprocess_clip
from src.evaluate import group_kfold_splits
from src.features.handcrafted_v2 import extract_handcrafted_features_v2


def build_matrix(df):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting v2 features"):
        raw = _raw_landmarks(row)
        sample = preprocess_clip(raw["xyz"], raw["visibility"])
        X.append(extract_handcrafted_features_v2(sample["xyz"]))
        y.append(ACTION_TO_IDX[row["action"]])
    return np.stack(X), np.array(y)


def new_model():
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        objective="multi:softmax", num_class=3, random_state=42,
    )


def main():
    df = load_labels()
    train_df, val_df, test_df = split_dataframe(df)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    X_train, y_train = build_matrix(train_df)
    X_val, y_val = build_matrix(val_df)
    X_test, y_test = build_matrix(test_df)

    model = new_model()
    model.fit(X_train, y_train)

    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        acc = (preds == y).mean()
        print(f"\n=== XGBoost v2 (locked split, train-only fit) -- {name} -- acc={acc:.3f} ===")
        print(classification_report(y, preds, target_names=ACTIONS))
        print(confusion_matrix(y, preds))

    importances = model.feature_importances_
    print(f"\nstarting-elbow-angle feature importance (locked-split fit): {importances[15]:.4f}  "
          f"rank {sorted(importances, reverse=True).index(importances[15]) + 1} of {len(importances)}")

    # leave-participant-out group k-fold CV, same method as v1's notebook cell 10
    print("\n=== leave-participant-out group k-fold CV (v2 features) ===")
    X_all, y_all = build_matrix(df)
    groups_all = df["participant_id"].values
    accs = []
    for fold, (train_idx, held_idx) in enumerate(group_kfold_splits(df, n_splits=3)):
        m = new_model()
        m.fit(X_all[train_idx], y_all[train_idx])
        acc = (m.predict(X_all[held_idx]) == y_all[held_idx]).mean()
        held_participants = sorted(set(groups_all[held_idx]))
        print(f"  fold {fold}  held={held_participants}  acc={acc:.3f}")
        accs.append(acc)
    print(f"CV mean={np.mean(accs):.3f}  std={np.std(accs):.3f}")


if __name__ == "__main__":
    main()
