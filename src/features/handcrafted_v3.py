"""
v3: same 15-feature layout as handcrafted.py, but swaps the hand-openness
feature's input source -- was Pose's crude wrist-to-fingertip DISTANCE (3
fingertip landmarks, shown by the finger-gun hypothesis test to move as one
rigid blob, not real per-finger articulation). Now uses real MediaPipe Hands
per-finger CURL ANGLE (src.data.hand_features.finger_curl_series), the same
signal ported into the 2s-AGCN's hand stream in graph.py v4 -- a dedicated
hand-tracking model's actual finger-joint angles, not a Pose byproduct.

Isolated on purpose: only the hand-signal source changes here, elbow angle
(handcrafted_v2) is a separate, independent addition -- so a held-out eval of
this file in isolation tells us whether the better hand signal alone helps,
before stacking it with anything else.

Needs MediaPipe Hands landmarks for the clip (src.data.graph_dataset's
_raw_hand_landmarks / extract_and_cache_hands), not just Pose -- callers must
pass hand_curl, the (SEQ_LEN, 2, 5) crop_and_resample'd curl array already
aligned to the same 40-frame timeline as xyz (see build_matrix below for the
exact call sequence, mirrors src.data.graph_dataset.build_graph_cache's det
path).
"""
import numpy as np

from src.features.handcrafted import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
    _joint_angle, _peak_speed_and_timing,
)

_WRIST_SIDE = {L_WRIST: 0, R_WRIST: 1}  # Pose wrist index -> Hands landmark slot


def _hand_curl_openness(hand_curl: np.ndarray, wrist_idx: int, frame_idx: int) -> float:
    """hand_curl: (SEQ_LEN, 2, 5) real MediaPipe Hands curl angle (0=fully
    curled/fist, 1=fully extended/open), already frame-aligned to xyz. Mean
    across all 5 fingers at the striking wrist's strike frame -- same idea as
    handcrafted.py's _hand_openness, real curl angle instead of Pose distance."""
    side = _WRIST_SIDE[wrist_idx]
    return float(np.mean(hand_curl[frame_idx, side, :]))


def extract_handcrafted_features_v3(xyz: np.ndarray, hand_curl: np.ndarray) -> np.ndarray:
    """xyz: (SEQ_LEN, 33, 3) normalized world coords. hand_curl: (SEQ_LEN, 2, 5)
    real Hands curl, already crop_and_resample'd to the same SEQ_LEN timeline
    (see build_matrix in scripts/eval_xgboost_v3_holdout.py for how to build
    it from a raw clip). -> 1D 15-dim feature vector, same layout as v1."""
    feats = []

    wrist_stats = {}
    for wrist in (L_WRIST, R_WRIST):
        peak_speed, timing, peak_idx = _peak_speed_and_timing(xyz, wrist)
        feats += [peak_speed, timing]
        wrist_stats[wrist] = (peak_speed, peak_idx)

    striking_wrist = max(wrist_stats, key=lambda w: wrist_stats[w][0])
    _, strike_peak_idx = wrist_stats[striking_wrist]
    strike_frame = min(strike_peak_idx + 1, xyz.shape[0] - 1)
    feats += [_hand_curl_openness(hand_curl, striking_wrist, strike_frame)]

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
        feats += [float(angle_series.max() - angle_series.min())]

    return np.array(feats, dtype=np.float32)
