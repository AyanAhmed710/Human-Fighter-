"""
Derives a per-finger curl-angle signal from src.data.extract_hands's raw
21-landmark MediaPipe Hands output -- the replacement for the old crude
wrist-to-fingertip-distance HandOpennessStream input (src.models.graph's
Pose-fingertip-node proxy). Curl angle at each finger's PIP-equivalent joint
(small when curled/fist, large when extended/open) is a stronger, more
literal fist-vs-open signal than raw distance, and is invariant to wrist
rotation, which distance isn't.

Angles are computed from MediaPipe Hands' own local per-hand landmark
frame, so no cross-model coordinate alignment with Pose's metric world
coordinates is needed -- an angle between two vectors from the same
landmark set is valid regardless of what coordinate system that set lives
in.
"""
import numpy as np

from src.config import SEQ_LEN
from src.data.preprocess import resample_sequence

# (mcp-equivalent, pip-equivalent "vertex", tip) landmark-index triples per
# finger, MediaPipe Hands' 21-point layout (0=wrist, 1-4=thumb, 5-8=index,
# 9-12=middle, 13-16=ring, 17-20=pinky). Angle is measured at the middle
# index of each triple, same (a, vertex, c) convention as
# src.data.preprocess.joint_angles.
FINGER_TRIPLES = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
N_FINGERS = len(FINGER_ORDER)


def _angle_at_vertex(landmarks: np.ndarray, a: int, vertex: int, c: int) -> np.ndarray:
    """landmarks: (T, 21, 3) for one hand. Returns (T,) angle/pi at `vertex`,
    NaN where any of the 3 points is NaN (hand not detected that frame)."""
    v1 = landmarks[:, a] - landmarks[:, vertex]
    v2 = landmarks[:, c] - landmarks[:, vertex]
    cos_ang = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    )
    return np.arccos(np.clip(cos_ang, -1.0, 1.0)) / np.pi


def _interpolate_nan_1d(series: np.ndarray) -> np.ndarray:
    """Same convention as src.data.preprocess.interpolate_nans -- linear fill
    along time, edge-fill at the boundaries, all-NaN falls back to 0."""
    t = np.arange(len(series))
    valid = ~np.isnan(series)
    if valid.sum() == 0:
        return np.zeros_like(series)
    if valid.sum() < len(series):
        return np.interp(t, t[valid], series[valid])
    return series


def finger_curl_series(hand_landmarks: np.ndarray):
    """hand_landmarks: (T, 2, 21, 3) from src.data.extract_hands, slot 0 =
    Left, slot 1 = Right. Returns:
        curl     (T, 2, N_FINGERS) float32, 0=fully curled, 1=fully extended,
                 NaN frames interpolated (or 0 if a hand never appears at all)
        presence (T, 2) float32, 1.0 where that hand was actually detected
                 that frame, 0.0 where interpolated/missing -- soft-confidence
                 weight, same spirit as Pose's visibility array (not a hard
                 drop, addendum #2)."""
    T = hand_landmarks.shape[0]
    curl = np.zeros((T, 2, N_FINGERS), dtype=np.float32)
    presence = (~np.isnan(hand_landmarks[:, :, 0, 0])).astype(np.float32)  # (T, 2)

    for side in (0, 1):
        for f_idx, finger in enumerate(FINGER_ORDER):
            a, vertex, c = FINGER_TRIPLES[finger]
            raw = _angle_at_vertex(hand_landmarks[:, side], a, vertex, c)
            curl[:, side, f_idx] = _interpolate_nan_1d(raw)

    return curl, presence


def mirror_curl(curl: np.ndarray) -> np.ndarray:
    """Swap Left<->Right hand slot -- curl angle itself is left/right-
    symmetric (same triple convention both hands), so mirroring only needs
    the slot swap, no value transform."""
    return curl[:, ::-1, :]


def warp_curl(curl: np.ndarray, factor: float) -> np.ndarray:
    """Resample the time axis by `factor` -- mirrors
    src.data.augment.time_warp but for the (T, 2, N_FINGERS) curl array
    (no visibility companion to resample here)."""
    T = curl.shape[0]
    new_T = max(4, int(round(T * factor)))
    src_t = np.linspace(0, 1, T)
    dst_t = np.linspace(0, 1, new_T)
    flat = curl.reshape(T, -1)
    out = np.empty((new_T, flat.shape[1]), dtype=curl.dtype)
    for k in range(flat.shape[1]):
        out[:, k] = np.interp(dst_t, src_t, flat[:, k])
    return out.reshape((new_T,) + curl.shape[1:])


def apply_train_augment(curl: np.ndarray, applied: dict) -> np.ndarray:
    """Replays the mirror/warp parts of one src.data.augment.augment_clip()
    call on a curl series so it stays frame-aligned with the augmented Pose
    xyz it'll be packed alongside. Rotation and joint-noise augmentations
    are Pose-only transforms and don't touch a wholly separate landmark
    source, so they're deliberately not replayed here."""
    out = curl
    if applied.get("mirrored"):
        out = mirror_curl(out)
    factor = applied.get("warp_factor")
    if factor is not None:
        out = warp_curl(out, factor)
    return out


def crop_and_resample(curl: np.ndarray, crop_window, target_len: int = SEQ_LEN) -> np.ndarray:
    """Applies the same (start_frac, end_frac) crop_window used by
    src.data.preprocess.preprocess_clip, then resamples to target_len --
    keeps the curl series's timeline aligned with the xyz/bone stream it's
    packed alongside for the same clip."""
    if crop_window is not None:
        T = curl.shape[0]
        start = int(crop_window[0] * T)
        end = max(start + 2, int(crop_window[1] * T))
        curl = curl[start:end]
    return resample_sequence(curl, target_len)
