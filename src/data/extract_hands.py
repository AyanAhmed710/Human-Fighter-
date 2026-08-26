"""
MediaPipe Hands extraction -- dedicated 21-landmark-per-hand model, run
separately from src.data.extract_landmarks's Pose extraction.

Why: Pose's own fingertip landmarks (17-22) are a coarse full-body-scale
guess -- mean visibility 0.71-0.78, dipping as low as 0.06, even on the
curated recording setup (see the punch/shoot live-accuracy diagnosis).
Hands crops in on the hand region with its own dedicated model, giving real
per-knuckle landmarks for a proper fist-vs-open finger-curl signal instead
of a crude wrist-to-tip distance computed off noisy Pose points.

Per clip, produces data/interim_hands/<action>/<clip_stem>.npz:
    landmarks (T, 2, 21, 3) float32 -- [:, 0] = Left hand, [:, 1] = Right
                                        hand, NaN where that hand wasn't
                                        detected in a given frame
    fps        scalar

Handedness comes straight from MediaPipe's own classifier -- reliable here
since extraction runs on the original (non-mirrored) source video, unlike
the live script, which mirrors the display frame and has to resolve
handedness a different way (nearest wrist match, see src/inference.py).
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.config import INTERIM_HANDS_DIR, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE

mp_hands = mp.solutions.hands

N_HAND_LANDMARKS = 21


def extract_hand_clip(video_path: Path) -> dict:
    """Run MediaPipe Hands over every frame of one clip. Returns raw arrays.

    Frames/hands with no detection get NaN landmarks -- handled downstream
    by src.data.hand_features (interpolated, not silently skipped), same
    convention as src.data.extract_landmarks's Pose extraction.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            slot = np.full((2, N_HAND_LANDMARKS, 3), np.nan, dtype=np.float32)
            if result.multi_hand_landmarks and result.multi_handedness:
                for lm_set, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    idx = 0 if label == "Left" else 1
                    slot[idx] = np.array(
                        [[p.x, p.y, p.z] for p in lm_set.landmark], dtype=np.float32
                    )
            frames.append(slot)
    cap.release()

    return {
        "landmarks": np.stack(frames, axis=0),   # (T, 2, 21, 3)
        "fps": np.float32(fps),
    }


def _cached_and_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            _ = data["landmarks"].shape
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def extract_and_cache_hands(video_path: Path, action: str, clip_stem: str, overwrite: bool = False) -> Path:
    out_dir = INTERIM_HANDS_DIR / action
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_stem}.npz"

    if not overwrite and _cached_and_valid(out_path):
        return out_path

    data = extract_hand_clip(video_path)
    # temp name must end in .npz -- np.savez_compressed appends .npz to any
    # path that doesn't already end with it, which breaks a naive ".tmp" suffix
    tmp_path = out_path.parent / f"{out_path.stem}.tmp.npz"
    np.savez_compressed(tmp_path, **data)
    tmp_path.replace(out_path)  # atomic -- a killed process can't leave a truncated file at out_path
    return out_path
