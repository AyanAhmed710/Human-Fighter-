"""
Cache builder for the 2s-AGCN input -- mirrors src.data.dataset.build_feature_cache
exactly (same atomic-write/cache-validation conventions, same augment_clip
call) but saves the structured (SEQ_LEN, TOTAL_PACKED_NODES, 6) joint+bone+
hand-curl tensor (src.models.graph.joint_bone_from_sample) instead of the
flattened 279-dim BiLSTM feature vector. Saved under the same "feat"/"label"
npz keys, so src.data.dataset.ClipSequenceDataset can load these caches
unchanged -- no new Dataset class needed.

v4: also pulls in src.data.extract_hands' MediaPipe Hands cache and packs
its derived curl-angle signal (src.data.hand_features) alongside the Pose-
derived skeleton, replacing v3's Pose-fingertip-distance HandOpennessStream
input.
"""
from pathlib import Path

import numpy as np

from src.config import ACTION_TO_IDX, PROCESSED_DIR, SEED, SEQ_LEN, AUG_MULTIPLIER
from src.data.augment import augment_clip
from src.data.dataset import _atomic_savez, _clip_video_path, _raw_landmarks
from src.data.extract_hands import extract_and_cache_hands
from src.data.hand_features import apply_train_augment, crop_and_resample, finger_curl_series
from src.data.preprocess import preprocess_clip
from src.models.graph import TOTAL_PACKED_NODES, joint_bone_from_sample


def _raw_hand_landmarks(row) -> np.ndarray:
    """(T, 2, 21, 3) raw MediaPipe Hands landmarks for this clip's video,
    same clip_stem/video resolution as src.data.dataset._raw_landmarks."""
    video_path = _clip_video_path(row)
    npz_path = extract_and_cache_hands(video_path, row["action"], row["clip_stem"])
    with np.load(npz_path) as data:
        return data["landmarks"]


def _cached_and_valid_graph(path: Path) -> bool:
    """Same idea as src.data.dataset._cached_and_valid (file loads OK) plus
    a shape check that generic one doesn't do -- a graph-cache npz that
    loads fine but has a stale node count (e.g. 20 from before the hand-curl
    pseudo-nodes existed) used to pass as "valid" and silently never get
    rebuilt, wasting a full re-extraction pass computing data that then got
    thrown away (see the 46-minute stale-cache incident this traced back
    to). Any shape mismatch here means rebuild, not "close enough"."""
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            feat = data["feat"]
            if feat.shape != (SEQ_LEN, TOTAL_PACKED_NODES, 6):
                return False
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def build_graph_cache(df, split: str, augment: bool = False, overwrite: bool = False):
    """Same contract as build_feature_cache: one .npz per (clip, variant)
    under data/processed/graph_<split>/, deterministic pass always runs,
    +AUG_MULTIPLIER augmented copies if augment=True (train split only).

    Checks which output paths are already cached BEFORE touching raw
    landmarks -- _raw_hand_landmarks in particular runs a full MediaPipe
    Hands pass over the clip video on a cache miss (slow), so a row whose
    outputs are all already valid skips that entirely instead of paying the
    lookup cost just to then discard it (this used to run unconditionally
    every row, every rerun, regardless of whether anything needed it)."""
    out_dir = PROCESSED_DIR / f"graph_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    paths = []
    for _, row in df.iterrows():
        det_path = out_dir / f"{row['action']}_{row['clip_stem']}_orig.npz"
        aug_paths = [out_dir / f"{row['action']}_{row['clip_stem']}_aug{k}.npz"
                     for k in range(AUG_MULTIPLIER)] if augment else []

        need_det = overwrite or not _cached_and_valid_graph(det_path)
        need_aug = [overwrite or not _cached_and_valid_graph(p) for p in aug_paths]

        if not need_det and not any(need_aug):
            paths.append(det_path)
            paths.extend(aug_paths)
            continue

        raw = _raw_landmarks(row)
        hand_raw = _raw_hand_landmarks(row)
        curl, presence = finger_curl_series(hand_raw)  # (T0, 2, 5), (T0, 2)

        if need_det:
            sample = preprocess_clip(raw["xyz"], raw["visibility"])
            hand_curl = crop_and_resample(curl, None, SEQ_LEN)
            hand_presence = crop_and_resample(presence, None, SEQ_LEN)
            feat = joint_bone_from_sample(sample, hand_curl, hand_presence)
            _atomic_savez(det_path, feat=feat, label=ACTION_TO_IDX[row["action"]])
        paths.append(det_path)

        for k, aug_path in enumerate(aug_paths):
            if need_aug[k]:
                aug_xyz, aug_vis, crop_window, applied = augment_clip(
                    raw["xyz"], raw["visibility"], rng=rng)
                sample = preprocess_clip(aug_xyz, aug_vis, crop_window=crop_window)

                aug_curl = apply_train_augment(curl, applied)
                aug_presence = apply_train_augment(presence[:, :, None], applied)[:, :, 0]
                hand_curl = crop_and_resample(aug_curl, crop_window, SEQ_LEN)
                hand_presence = crop_and_resample(aug_presence, crop_window, SEQ_LEN)

                feat = joint_bone_from_sample(sample, hand_curl, hand_presence)
                _atomic_savez(aug_path, feat=feat, label=ACTION_TO_IDX[row["action"]])
            paths.append(aug_path)

    return paths


def predict_with_tta_agcn(model, xyz: np.ndarray, visibility: np.ndarray,
                           hand_landmarks: np.ndarray, device: str = "cpu") -> int:
    """Same TTA idea as src.evaluate.predict_with_tta (addendum #6: orig +
    mirror + 2 time-warps), rebuilt for the (SEQ_LEN, TOTAL_PACKED_NODES, 6)
    AGCN input instead of the flat BiLSTM feature vector. hand_landmarks:
    (T, 2, 21, 3) raw MediaPipe Hands output for the same clip, from
    _raw_hand_landmarks -- mirror/warp variants are replayed on the curl
    series to stay aligned with src.evaluate.tta_variants' xyz variants."""
    import torch

    from src.evaluate import tta_variants
    from src.data.hand_features import mirror_curl, warp_curl

    curl, presence = finger_curl_series(hand_landmarks)  # (T0, 2, 5), (T0, 2)
    hand_variants = [(curl, presence)]
    hand_variants.append((mirror_curl(curl), presence[:, ::-1]))
    for factor in (0.9, 1.1):
        hand_variants.append((warp_curl(curl, factor), warp_curl(presence[:, :, None], factor)[:, :, 0]))

    model.eval()
    probs = []
    with torch.no_grad():
        for (v_xyz, v_vis), (v_curl, v_presence) in zip(tta_variants(xyz, visibility), hand_variants):
            sample = preprocess_clip(v_xyz, v_vis)
            hand_curl = crop_and_resample(v_curl, None, SEQ_LEN)
            hand_presence = crop_and_resample(v_presence, None, SEQ_LEN)
            feat = joint_bone_from_sample(sample, hand_curl, hand_presence)
            feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            logits, _ = model(feat_t)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    avg_probs = np.mean(np.concatenate(probs, axis=0), axis=0)
    return int(np.argmax(avg_probs))


def predict_segment_agcn(model, xyz: np.ndarray, vis: np.ndarray, hand_landmarks: np.ndarray,
                          device: str = "cpu", debug: bool = False):
    """AGCN counterpart of src.inference.predict_segment -- same captured
    segment, same quality gate (visibility/motion), swaps the handcrafted-
    feature + XGBoost path for the joint+bone+hand-curl graph tensor +
    TwoStreamAGCN forward pass. hand_landmarks: (T, 2, 21, 3), same frame
    count/timeline as xyz/vis (see src.inference.ActionSegmenter's aux_frame
    buffering). Single-shot (no TTA) for low-latency live use;
    predict_with_tta_agcn above is for offline batch eval. Returns
    (label, probs[3], reason)."""
    import torch

    from src.config import ACTIONS
    from src.inference import _MOTION_JOINTS, frame_quality_gate

    sample = preprocess_clip(xyz, vis)
    peak_speed = float(np.max(np.linalg.norm(np.diff(sample["xyz"][:, _MOTION_JOINTS], axis=0), axis=-1)))

    ok, reason = frame_quality_gate(vis, sample["xyz"], peak_speed)
    if not ok:
        if debug:
            print(f"[gate] REJECT (visibility/motion): {reason}  peak_speed={peak_speed:.3f}")
        return "idle", np.array([1 / 3, 1 / 3, 1 / 3]), reason

    curl, presence = finger_curl_series(hand_landmarks)
    hand_curl = crop_and_resample(curl, None, SEQ_LEN)
    hand_presence = crop_and_resample(presence, None, SEQ_LEN)
    feat = joint_bone_from_sample(sample, hand_curl, hand_presence)
    feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(feat_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    label = ACTIONS[int(np.argmax(probs))]
    if debug:
        print(f"[gate] pass  peak_speed={peak_speed:.3f}")
        print(f"[predict] {label}  probs={np.round(probs, 3).tolist()}")
    return label, probs, None
