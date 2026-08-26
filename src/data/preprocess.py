"""
Preprocessing chain (protocol section 4 + addendum #1/#2):
  soft-confidence weight -> moving-average smooth -> normalize (hip-center,
  shoulder-hip scale) -> bone-vector stream -> velocity stream -> resample to
  fixed SEQ_LEN.

Hard frame-dropping was rejected (addendum #2) in favor of carrying the
visibility array through as a per-frame weight -- short clips (~76 frames avg,
strike window inside) can't afford to lose the strike itself to a confidence
cutoff.
"""
import numpy as np

from src.config import ANGLE_TRIPLES, BONE_PAIRS, SEQ_LEN, SMOOTH_WINDOW, VISIBILITY_THRESHOLD

LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def interpolate_nans(xyz: np.ndarray) -> np.ndarray:
    """Fill NaN frames (failed detections) by linear interpolation along time,
    per landmark per axis. Edge NaNs are filled with the nearest valid value."""
    T, J, C = xyz.shape
    out = xyz.copy()
    t = np.arange(T)
    for j in range(J):
        for c in range(C):
            series = out[:, j, c]
            valid = ~np.isnan(series)
            if valid.sum() == 0:
                out[:, j, c] = 0.0
                continue
            if valid.sum() < T:
                out[:, j, c] = np.interp(t, t[valid], series[valid])
    return out


def smooth(xyz: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Moving-average filter along the time axis, edge-padded."""
    if window <= 1:
        return xyz
    pad = window // 2
    padded = np.pad(xyz, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    out = np.empty_like(xyz)
    T, J, C = xyz.shape
    for j in range(J):
        for c in range(C):
            out[:, j, c] = np.convolve(padded[:, j, c], kernel, mode="valid")[:T]
    return out


def normalize(xyz: np.ndarray) -> np.ndarray:
    """Center on hip midpoint, scale by shoulder-to-hip distance (per frame),
    removing camera-distance and body-size variation across participants."""
    hip_mid = (xyz[:, LEFT_HIP] + xyz[:, RIGHT_HIP]) / 2.0        # (T, 3)
    shoulder_mid = (xyz[:, LEFT_SHOULDER] + xyz[:, RIGHT_SHOULDER]) / 2.0  # (T, 3)

    centered = xyz - hip_mid[:, None, :]
    scale = np.linalg.norm(shoulder_mid - hip_mid, axis=-1)      # (T,)
    scale = np.clip(scale, 1e-6, None)
    return centered / scale[:, None, None]


def bone_vectors(xyz: np.ndarray) -> np.ndarray:
    """Adjacent-joint delta vectors (addendum #1) -- the 'bone' stream of a
    two-stream skeleton model, more invariant to body size than raw coords."""
    T = xyz.shape[0]
    bones = np.zeros((T, len(BONE_PAIRS), 3), dtype=np.float32)
    for i, (a, b) in enumerate(BONE_PAIRS):
        bones[:, i] = xyz[:, b] - xyz[:, a]
    return bones


def joint_angles(xyz: np.ndarray) -> np.ndarray:
    """Per-frame flexion angle at each ANGLE_TRIPLES vertex, normalized to
    [0, 1] (radians / pi). This is the explicit ROM signal that dominated the
    XGBoost sanity baseline (hipR_rom = 42% feature importance) -- giving it
    to the deep model as a continuous stream instead of forcing the model to
    re-derive it from raw coordinates on a small training set."""
    T = xyz.shape[0]
    angles = np.zeros((T, len(ANGLE_TRIPLES)), dtype=np.float32)
    for i, (a, b, c) in enumerate(ANGLE_TRIPLES):
        v1 = xyz[:, a] - xyz[:, b]
        v2 = xyz[:, c] - xyz[:, b]
        cos_ang = np.sum(v1 * v2, axis=-1) / (
            np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
        )
        angles[:, i] = np.arccos(np.clip(cos_ang, -1.0, 1.0))
    return angles / np.pi


def velocity(xyz: np.ndarray) -> np.ndarray:
    """First-order frame-to-frame delta, zero-padded at frame 0."""
    vel = np.zeros_like(xyz)
    vel[1:] = xyz[1:] - xyz[:-1]
    return vel


def resample_sequence(arr: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    """Interpolate along the time axis to a fixed length. Works for any
    (T, ...) array -- xyz, bones, velocity, visibility all resampled the same
    way so frames stay aligned across streams."""
    T = arr.shape[0]
    if T == target_len:
        return arr
    src_t = np.linspace(0, 1, T)
    dst_t = np.linspace(0, 1, target_len)
    flat = arr.reshape(T, -1)
    out = np.empty((target_len, flat.shape[1]), dtype=arr.dtype)
    for k in range(flat.shape[1]):
        out[:, k] = np.interp(dst_t, src_t, flat[:, k])
    return out.reshape((target_len,) + arr.shape[1:])


def preprocess_clip(xyz: np.ndarray, visibility: np.ndarray, crop_window: tuple = None) -> dict:
    """Full chain for one clip. crop_window=(start_frac, end_frac) applies the
    random-window crop augmentation (addendum, section 9) before resampling --
    leave None for the deterministic (non-augmented) pass.

    Returns dict of resampled streams, all shape (SEQ_LEN, ...):
        xyz, bones, velocity, visibility
    """
    xyz = interpolate_nans(xyz)
    xyz = smooth(xyz)
    xyz = normalize(xyz)

    if crop_window is not None:
        T = xyz.shape[0]
        start = int(crop_window[0] * T)
        end = max(start + 2, int(crop_window[1] * T))
        xyz = xyz[start:end]
        visibility = visibility[start:end]

    bones = bone_vectors(xyz)
    vel = velocity(xyz)
    angles = joint_angles(xyz)

    # soft-confidence: don't drop, just floor it so a fully-occluded joint
    # doesn't get a literal 0 weight (keeps gradient flowing)
    vis_weight = np.clip(visibility, VISIBILITY_THRESHOLD * 0.2, 1.0)

    return {
        "xyz": resample_sequence(xyz),
        "bones": resample_sequence(bones),
        "velocity": resample_sequence(vel),
        "visibility": resample_sequence(vis_weight),
        "angles": resample_sequence(angles),
    }


def to_feature_vector(sample: dict) -> np.ndarray:
    """Flatten the stream dict into the (SEQ_LEN, D) model input --
    xyz (99) + velocity (99) + visibility (33) + bones (len(BONE_PAIRS)*3)
    + angles (len(ANGLE_TRIPLES))."""
    T = sample["xyz"].shape[0]
    parts = [
        sample["xyz"].reshape(T, -1),
        sample["velocity"].reshape(T, -1),
        sample["visibility"].reshape(T, -1),
        sample["bones"].reshape(T, -1),
        sample["angles"].reshape(T, -1),
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)
