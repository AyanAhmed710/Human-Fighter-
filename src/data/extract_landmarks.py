"""
MediaPipe pose extraction -- world landmarks (metric, hip-relative), not
image-plane normalized landmarks (protocol section 3, the highest-leverage
decision in the whole pipeline).

Per clip, produces a raw .npz cached under data/interim/<action>/<clip_stem>.npz:
    xyz        (T, 33, 3) float32  -- world coords, meters
    visibility (T, 33)    float32  -- per-landmark confidence, 0-1
    fps        scalar
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    INTERIM_DIR,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    N_POSE_LANDMARKS,
)

mp_pose = mp.solutions.pose


def extract_clip(video_path: Path) -> dict:
    """Run MediaPipe Pose over every frame of one clip. Returns raw arrays.

    Frames where pose detection fails entirely get NaN xyz + 0 visibility --
    handled downstream by preprocessing, not silently skipped here (skipping
    would desync the frame timeline).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    xyz_frames = []
    vis_frames = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_world_landmarks is None:
                xyz_frames.append(np.full((N_POSE_LANDMARKS, 3), np.nan, dtype=np.float32))
                vis_frames.append(np.zeros(N_POSE_LANDMARKS, dtype=np.float32))
                continue

            lms = result.pose_world_landmarks.landmark
            xyz = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            vis = np.array([lm.visibility for lm in lms], dtype=np.float32)
            xyz_frames.append(xyz)
            vis_frames.append(vis)

    cap.release()

    return {
        "xyz": np.stack(xyz_frames, axis=0),          # (T, 33, 3)
        "visibility": np.stack(vis_frames, axis=0),    # (T, 33)
        "fps": np.float32(fps),
    }


def _cached_and_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            _ = data["xyz"].shape
        return True
    except Exception:
        path.unlink(missing_ok=True)
        return False


def extract_and_cache(video_path: Path, action: str, clip_stem: str, overwrite: bool = False) -> Path:
    out_dir = INTERIM_DIR / action
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_stem}.npz"

    if not overwrite and _cached_and_valid(out_path):
        return out_path

    data = extract_clip(video_path)
    # temp name must end in .npz -- np.savez_compressed appends .npz to any
    # path that doesn't already end with it, which breaks a naive ".tmp" suffix
    tmp_path = out_path.parent / f"{out_path.stem}.tmp.npz"
    np.savez_compressed(tmp_path, **data)
    tmp_path.replace(out_path)  # atomic -- a killed process can't leave a truncated file at out_path
    return out_path
