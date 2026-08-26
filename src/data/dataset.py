"""
Loads combined_labels.csv, builds participant-based train/val/test splits
(locked in config.py), runs each clip through extract -> preprocess -> feature
vector, and exposes a torch Dataset. Caches feature arrays to data/processed/
so re-running a notebook doesn't re-run MediaPipe every time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    ACTION_FOLDERS,
    ACTION_TO_IDX,
    AUG_MULTIPLIER,
    INTERIM_DIR,
    LABELS_CSV,
    PROCESSED_DIR,
    SEED,
    TEST_PARTICIPANTS,
    TRAIN_PARTICIPANTS,
    VAL_PARTICIPANTS,
)
from src.data.augment import augment_clip
from src.data.extract_landmarks import extract_and_cache
from src.data.preprocess import preprocess_clip, to_feature_vector


def _atomic_savez(path: Path, **arrays):
    """Write to a temp file then replace onto the final path -- a process
    killed mid-write can never leave a truncated file at `path` (the failure
    mode that caused a corrupt cache entry to silently pass the `.exists()`
    check and crash training later, see notebook incident).

    Temp name must itself end in .npz -- np.savez_compressed silently
    appends .npz to any path that doesn't already end with it, so a name
    like "x.npz.tmp" actually gets written as "x.npz.tmp.npz" and the
    subsequent replace() can't find what it thinks it just wrote."""
    tmp_path = path.parent / f"{path.stem}.tmp.npz"
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def _cached_and_valid(path: Path) -> bool:
    """Existence isn't enough -- verify the cache file actually loads."""
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            _ = data["feat"].shape
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    df["clip_stem"] = df["clip_id"].apply(lambda p: Path(p).stem)
    return df


def split_dataframe(df: pd.DataFrame):
    train_df = df[df["participant_id"].isin(TRAIN_PARTICIPANTS)].reset_index(drop=True)
    val_df = df[df["participant_id"].isin(VAL_PARTICIPANTS) & (df["camera_angle"] != "front")].reset_index(drop=True)
    test_df = df[df["participant_id"].isin(TEST_PARTICIPANTS) & (df["camera_angle"] != "front")].reset_index(drop=True)
    return train_df, val_df, test_df


def _clip_video_path(row) -> Path:
    return ACTION_FOLDERS[row["action"]] / f"{row['clip_stem']}.mp4"


def _raw_landmarks(row) -> dict:
    video_path = _clip_video_path(row)
    npz_path = extract_and_cache(video_path, row["action"], row["clip_stem"])
    data = np.load(npz_path)
    return {"xyz": data["xyz"], "visibility": data["visibility"]}


def build_feature_cache(df: pd.DataFrame, split: str, augment: bool = False, overwrite: bool = False):
    """Extracts + preprocesses every clip in df, writes one .npz per (clip,
    variant) under data/processed/<split>/. Deterministic pass always runs;
    if augment=True, AUG_MULTIPLIER extra augmented copies are added per clip
    (train split only -- caller controls this)."""
    out_dir = PROCESSED_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    paths = []
    for _, row in df.iterrows():
        raw = _raw_landmarks(row)

        # deterministic (non-augmented) sample
        det_path = out_dir / f"{row['action']}_{row['clip_stem']}_orig.npz"
        if overwrite or not _cached_and_valid(det_path):
            sample = preprocess_clip(raw["xyz"], raw["visibility"])
            feat = to_feature_vector(sample)
            _atomic_savez(det_path, feat=feat, label=ACTION_TO_IDX[row["action"]])
        paths.append(det_path)

        if augment:
            for k in range(AUG_MULTIPLIER):
                aug_path = out_dir / f"{row['action']}_{row['clip_stem']}_aug{k}.npz"
                if overwrite or not _cached_and_valid(aug_path):
                    aug_xyz, aug_vis, crop_window, _applied = augment_clip(raw["xyz"], raw["visibility"], rng=rng)
                    sample = preprocess_clip(aug_xyz, aug_vis, crop_window=crop_window)
                    feat = to_feature_vector(sample)
                    _atomic_savez(aug_path, feat=feat, label=ACTION_TO_IDX[row["action"]])
                paths.append(aug_path)

    return paths


class ClipSequenceDataset(Dataset):
    """Wraps a list of cached .npz feature-vector paths (from build_feature_cache)."""

    def __init__(self, npz_paths):
        self.paths = list(npz_paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # context manager -- always closes the file handle, even on error,
        # so a mid-batch exception can't leave a Windows file lock behind
        with np.load(self.paths[idx]) as data:
            feat = torch.from_numpy(data["feat"]).float()   # (SEQ_LEN, D)
            label = torch.tensor(int(data["label"]), dtype=torch.long)
        return feat, label
