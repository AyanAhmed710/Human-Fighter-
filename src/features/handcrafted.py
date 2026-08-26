"""
Hand-crafted feature baseline (protocol addendum #5) -- a quick sanity check
run BEFORE the deep BiLSTM baseline. Cheap to compute, cheap to train
(XGBoost/SVM), and catches broken labels/pipeline bugs before spending time on
a deep model.

Features per clip, from the normalized (SEQ_LEN, 33, 3) xyz stream:
  - peak wrist/ankle velocity (L and R) -- strike speed/snap signal
  - joint-angle range of motion: elbow, knee, hip flexion (L and R)
  - strike-frame timing: frame index (0-1 normalized) of peak wrist/ankle speed
  - hand openness at each wrist's strike frame (L and R) -- fist vs open-palm
    proxy, added after live-testing showed punch/shoot collapsing together
    whenever legs aren't in play: MediaPipe Pose's wrist is a single point,
    it can't see finger shape, so elbow-ROM/wrist-timing alone are indirect
    proxies for the actual visual tell (closed fist vs flat open hand, per
    the frame-by-frame video review). Pose does include crude fingertip
    landmarks (pinky/index/thumb tips, 17-22) though -- mean distance from
    wrist to those three approximates open (large) vs closed (small).
"""
import numpy as np

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

_FINGERTIPS = {
    L_WRIST: (L_PINKY, L_INDEX, L_THUMB),
    R_WRIST: (R_PINKY, R_INDEX, R_THUMB),
}


def _joint_angle(a, b, c):
    """Angle at vertex b, given three (T, 3) joint position streams."""
    v1 = a - b
    v2 = c - b
    cos_ang = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def _peak_speed_and_timing(xyz: np.ndarray, joint_idx: int):
    pos = xyz[:, joint_idx]                  # (T, 3)
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=-1)  # (T-1,)
    peak_idx = int(np.argmax(speed))
    return float(speed[peak_idx]), peak_idx / max(1, len(speed) - 1), peak_idx


def _hand_openness(xyz: np.ndarray, wrist_idx: int, frame_idx: int) -> float:
    """Mean distance from wrist to pinky/index/thumb tips at one frame --
    small = fingers curled to the wrist (fist), large = fingers extended
    away from it (open hand)."""
    wrist_pos = xyz[frame_idx, wrist_idx]
    tips = _FINGERTIPS[wrist_idx]
    dists = [np.linalg.norm(xyz[frame_idx, t] - wrist_pos) for t in tips]
    return float(np.mean(dists))


def extract_handcrafted_features(xyz: np.ndarray) -> np.ndarray:
    """xyz: (SEQ_LEN, 33, 3) normalized world coords -> 1D feature vector."""
    feats = []

    wrist_stats = {}
    for wrist in (L_WRIST, R_WRIST):
        peak_speed, timing, peak_idx = _peak_speed_and_timing(xyz, wrist)
        feats += [peak_speed, timing]
        wrist_stats[wrist] = (peak_speed, peak_idx)

    # single hand-agnostic openness feature, using whichever wrist actually
    # struck (higher peak speed) -- which physical hand (L/R) strikes varies
    # per clip and isn't itself meaningful, so a per-hand L/R split dilutes
    # the real signal across two columns instead of exposing it in one
    striking_wrist = max(wrist_stats, key=lambda w: wrist_stats[w][0])
    _, strike_peak_idx = wrist_stats[striking_wrist]
    strike_frame = min(strike_peak_idx + 1, xyz.shape[0] - 1)
    feats += [_hand_openness(xyz, striking_wrist, strike_frame)]

    for ankle in (L_ANKLE, R_ANKLE):
        peak_speed, timing, _ = _peak_speed_and_timing(xyz, ankle)
        feats += [peak_speed, timing]

    elbow_l = _joint_angle(xyz[:, L_SHOULDER], xyz[:, L_ELBOW], xyz[:, L_WRIST])
    elbow_r = _joint_angle(xyz[:, R_SHOULDER], xyz[:, R_ELBOW], xyz[:, R_WRIST])
    knee_l = _joint_angle(xyz[:, L_HIP], xyz[:, L_KNEE], xyz[:, L_ANKLE])
    knee_r = _joint_angle(xyz[:, R_HIP], xyz[:, R_KNEE], xyz[:, R_ANKLE])
    hip_l = _joint_angle(xyz[:, L_SHOULDER], xyz[:, L_HIP], xyz[:, L_KNEE])
    hip_r = _joint_angle(xyz[:, R_SHOULDER], xyz[:, R_HIP], xyz[:, R_KNEE])

    for angle_series in (elbow_l, elbow_r, knee_l, knee_r, hip_l, hip_r):
        feats += [float(angle_series.max() - angle_series.min())]  # ROM

    return np.array(feats, dtype=np.float32)
