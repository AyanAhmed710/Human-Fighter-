"""
Evaluation utilities: TTA inference (addendum #6) and leave-participant-out
group k-fold CV (addendum #3) for a trustworthy accuracy estimate given only
9 participants.
"""
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from src.data.augment import mirror, rotate_y, time_warp
from src.data.preprocess import bone_vectors, joint_angles, normalize, resample_sequence, smooth, velocity


def tta_variants(xyz: np.ndarray, visibility: np.ndarray):
    """Original + mirrored + 2 mild time-warped copies (addendum #6)."""
    variants = [(xyz, visibility)]
    variants.append((mirror(xyz), visibility))
    for factor in (0.9, 1.1):
        variants.append(time_warp(xyz, visibility, factor))
    return variants


def to_feature_vector_from_xyz(xyz: np.ndarray, visibility: np.ndarray, seq_len: int) -> np.ndarray:
    """Mirrors src.data.preprocess.preprocess_clip + to_feature_vector but
    takes already-normalized-ready raw xyz -- used by TTA where we regenerate
    variants after the deterministic preprocess pass already ran once."""
    xyz_n = normalize(smooth(xyz))
    bones = bone_vectors(xyz_n)
    vel = velocity(xyz_n)
    angles = joint_angles(xyz_n)
    xyz_r = resample_sequence(xyz_n, seq_len)
    bones_r = resample_sequence(bones, seq_len)
    vel_r = resample_sequence(vel, seq_len)
    vis_r = resample_sequence(visibility, seq_len)
    angles_r = resample_sequence(angles, seq_len)
    T = xyz_r.shape[0]
    # order must match src.data.preprocess.to_feature_vector exactly
    return np.concatenate([
        xyz_r.reshape(T, -1), vel_r.reshape(T, -1), vis_r.reshape(T, -1),
        bones_r.reshape(T, -1), angles_r.reshape(T, -1),
    ], axis=-1).astype(np.float32)


@torch.no_grad()
def predict_with_tta(model, xyz: np.ndarray, visibility: np.ndarray, seq_len: int, device: str = "cpu") -> int:
    model.eval()
    probs = []
    for v_xyz, v_vis in tta_variants(xyz, visibility):
        feat = to_feature_vector_from_xyz(v_xyz, v_vis, seq_len)
        feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        logits, _ = model(feat_t)
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    avg_probs = np.mean(np.concatenate(probs, axis=0), axis=0)
    return int(np.argmax(avg_probs))


@torch.no_grad()
def evaluate_dataset(model, dataset, device: str = "cpu", class_names=None):
    """Plain (non-TTA) eval over a ClipSequenceDataset -- used for val/test
    once features are already cached, and for reporting confusion matrix."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    all_preds, all_labels = [], []
    for feat, label in loader:
        feat = feat.to(device)
        logits, _ = model(feat)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(label.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(labels, preds)
    return report, cm


def group_kfold_splits(df, n_splits: int = 3):
    """Leave-participant-out group k-fold (addendum #3) -- yields
    (train_idx, held_out_idx) over df grouped by participant_id, so no
    participant appears in both sides of a fold."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["participant_id"].values
    for train_idx, held_idx in gkf.split(df, groups=groups):
        yield train_idx, held_idx


def bilstm_group_cv(df, n_splits: int = 3, device: str = "cpu", epochs: int = 60, verbose: bool = True):
    """Leave-participant-out group k-fold CV for the BiLSTM baseline itself
    (addendum #3's deep-model follow-up). Trains a fresh model per fold on
    n-1 groups' participants (augmented), evaluates on the held-out group's
    side-only clips -- reports per-fold accuracy AND per-participant
    breakdown, so a single bad participant is visible instead of averaged
    away.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import DataLoader

    from src.config import ACTIONS, ACTION_TO_IDX, PROCESSED_DIR, SEED
    from src.data.dataset import build_feature_cache, ClipSequenceDataset
    from src.models.bilstm import BiLSTMBaseline
    from src.train import train_baseline
    from sklearn.metrics import confusion_matrix

    groups = df["participant_id"].values
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    for fold_i, (train_idx, held_idx) in enumerate(gkf.split(df, groups=groups)):
        train_participants = sorted(set(groups[train_idx]))
        held_participants = sorted(set(groups[held_idx]))

        fold_train_df = df.iloc[train_idx].reset_index(drop=True)
        fold_held_df = df.iloc[held_idx]
        fold_held_df = fold_held_df[fold_held_df["camera_angle"] != "front"].reset_index(drop=True)

        train_paths = build_feature_cache(fold_train_df, split=f"cv{fold_i}_train", augment=True)
        held_paths = build_feature_cache(fold_held_df, split=f"cv{fold_i}_held", augment=False)

        train_ds = ClipSequenceDataset(train_paths)
        held_ds = ClipSequenceDataset(held_paths)
        input_dim = train_ds[0][0].shape[-1]

        model = BiLSTMBaseline(input_dim=input_dim, hidden_dim=128, num_layers=2, num_classes=3)
        run_dir, _ = train_baseline(
            model, train_ds, held_ds, run_name=f"cv_fold{fold_i}", device=device, epochs=epochs,
            extra_config={"fold": fold_i, "held_participants": held_participants},
        )
        model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True))
        model.eval()

        loader = DataLoader(held_ds, batch_size=32, shuffle=False)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for feat, label in loader:
                logits, _ = model(feat.to(device))
                all_preds.append(logits.argmax(-1).cpu().numpy())
                all_labels.append(label.numpy())
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        acc = (preds == labels).mean()

        # per-participant breakdown within this held-out fold
        part_ids = fold_held_df["participant_id"].values
        per_part = {}
        for pid in held_participants:
            mask = part_ids == pid
            per_part[int(pid)] = float((preds[mask] == labels[mask]).mean())  # int() -- np.int64 keys break json.dump

        if verbose:
            print(f"fold {fold_i}: held-out {held_participants}  acc={acc:.3f}  per-participant={per_part}")
            print(confusion_matrix(labels, preds))

        fold_results.append({"fold": fold_i, "held_participants": held_participants,
                              "acc": acc, "per_participant": per_part})

    accs = [r["acc"] for r in fold_results]
    if verbose:
        print(f"\nBiLSTM leave-participant-out CV: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    return fold_results
