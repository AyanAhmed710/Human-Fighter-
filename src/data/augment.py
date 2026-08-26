"""
Skeleton-level augmentation (protocol section 6 + addendum #6).
Applied to TRAIN split only, on the raw (pre-normalize) world xyz, before the
preprocess chain runs -- so normalization/bone/velocity are always derived
from the already-augmented pose, keeping streams consistent.
"""
import numpy as np

from src.config import (
    CROP_WINDOW_RANGE,
    JOINT_NOISE_STD,
    ROTATION_JITTER_DEG,
    TIME_WARP_RANGE,
)

# mirror swaps every LEFT_* <-> RIGHT_* MediaPipe pose landmark index
MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def rotate_y(xyz: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate around the vertical (y) axis -- camera-jitter simulation for a
    fixed side view, kept tight (+/-10 deg default) since deployment angle
    never actually varies."""
    theta = np.deg2rad(degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]], dtype=np.float32)
    return xyz @ rot.T


def time_warp(xyz: np.ndarray, visibility: np.ndarray, factor: float):
    """Speed up/slow down playback by resampling the time axis by `factor`."""
    T = xyz.shape[0]
    new_T = max(4, int(round(T * factor)))
    src_t = np.linspace(0, 1, T)
    dst_t = np.linspace(0, 1, new_T)

    flat = xyz.reshape(T, -1)
    warped = np.empty((new_T, flat.shape[1]), dtype=xyz.dtype)
    for k in range(flat.shape[1]):
        warped[:, k] = np.interp(dst_t, src_t, flat[:, k])
    xyz_out = warped.reshape((new_T,) + xyz.shape[1:])

    vis_out = np.empty((new_T, visibility.shape[1]), dtype=visibility.dtype)
    for k in range(visibility.shape[1]):
        vis_out[:, k] = np.interp(dst_t, src_t, visibility[:, k])

    return xyz_out, vis_out


def joint_noise(xyz: np.ndarray, std: float = JOINT_NOISE_STD, rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return xyz + rng.normal(0, std, size=xyz.shape).astype(np.float32)


def mirror(xyz: np.ndarray) -> np.ndarray:
    """Flip x-axis (left-right) and swap paired landmark indices. limb_used
    is not a tracked label (dropped per project decision), so no relabel step
    is needed -- this is pure augmentation, not a limb-balance fix."""
    out = xyz.copy()
    out[..., 0] *= -1
    for a, b in MIRROR_PAIRS:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def random_crop_window(rng: np.random.Generator = None) -> tuple:
    """Sample a (start_frac, end_frac) window covering CROP_WINDOW_RANGE
    (default 90-100%) of the clip, positioned randomly -- feeds into
    preprocess.preprocess_clip(crop_window=...)."""
    rng = rng or np.random.default_rng()
    keep_frac = rng.uniform(*CROP_WINDOW_RANGE)
    max_start = 1.0 - keep_frac
    start = rng.uniform(0, max_start) if max_start > 0 else 0.0
    return start, start + keep_frac


def augment_clip(xyz: np.ndarray, visibility: np.ndarray, rng: np.random.Generator = None):
    """One random augmented copy: stacks a random subset of 2-3 techniques
    (rotation, time-warp, noise, mirror) per addendum section 9. Random crop
    window is returned separately for the caller to pass into preprocess_clip.

    Also returns `applied` -- which of the timeline-changing techniques
    (mirror, warp) actually fired and their params, so a second, wholly
    separate landmark source for the same clip (src.data.hand_features'
    curl series) can be kept frame-aligned via
    src.data.hand_features.apply_train_augment instead of guessing. Rotation
    and joint-noise are Pose-xyz-only transforms and aren't reported since
    they don't apply to other landmark sources."""
    rng = rng or np.random.default_rng()
    techniques = rng.choice(["rotate", "warp", "noise", "mirror"], size=3, replace=False)

    out_xyz, out_vis = xyz.copy(), visibility.copy()
    applied = {"mirrored": False, "warp_factor": None}
    for tech in techniques:
        if tech == "rotate":
            deg = rng.uniform(-ROTATION_JITTER_DEG, ROTATION_JITTER_DEG)
            out_xyz = rotate_y(out_xyz, deg)
        elif tech == "warp":
            factor = rng.uniform(*TIME_WARP_RANGE)
            out_xyz, out_vis = time_warp(out_xyz, out_vis, factor)
            applied["warp_factor"] = factor
        elif tech == "noise":
            out_xyz = joint_noise(out_xyz, rng=rng)
        elif tech == "mirror":
            out_xyz = mirror(out_xyz)
            applied["mirrored"] = True

    crop_window = random_crop_window(rng)
    return out_xyz, out_vis, crop_window, applied
