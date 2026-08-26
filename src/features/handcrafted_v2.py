"""
v2 of the hand-crafted feature baseline -- adds ONE new feature on top of
src/features/handcrafted.py's 15: starting elbow angle at the striking arm.

Kept as a fully separate module (not an edit to handcrafted.py) so the
original XGBoost model/pipeline stays untouched -- see
scripts/train_xgboost_final_v2.py for the matching separate training script.

Why this feature: live-testing + a live back-and-forth with the user
surfaced it, verified against the full 642-clip dataset (not a hand-picked
sample) before adding here:
  punching  starting elbow angle: mean 105.5 deg, std 30.2 (bent guard pose)
  shooting  starting elbow angle: mean 148.8 deg, std  8.1 (already near-
            extended, consistent with "raise a straight arm and point")
  effect size 2.27, only 33.6% range overlap -- the strongest single
  separator found this session, well above hand-openness's 1.2-1.3 and
  elbow-ROM's more diffuse signal. Crucially it's built entirely from
  shoulder/elbow/wrist -- Pose's best-tracked joints, not the fragile
  fingertip landmarks (17-22) that made the hand-openness feature and the
  AGCN hand-openness stream unreliable live -- so this one should hold up
  outside the curated recording setup too, not just on the locked test set.
"""
import numpy as np

from src.features.handcrafted import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    _hand_openness,
    _joint_angle,
    _peak_speed_and_timing,
)

START_FRAMES = 3  # frames averaged for the "starting" angle -- denoises a single-frame read

_ELBOW_OF = {L_WRIST: (L_SHOULDER, L_ELBOW), R_WRIST: (R_SHOULDER, R_ELBOW)}


def _starting_elbow_angle(xyz: np.ndarray, wrist_idx: int) -> float:
    """Mean elbow angle (degrees) over the first START_FRAMES frames of the
    striking arm -- before any wind-up/extension motion has happened, so it
    reads the ready/guard-vs-already-extended pose the clip started from."""
    shoulder_idx, elbow_idx = _ELBOW_OF[wrist_idx]
    angle_series = _joint_angle(xyz[:, shoulder_idx], xyz[:, elbow_idx], xyz[:, wrist_idx])
    return float(np.mean(angle_series[:START_FRAMES]))


def extract_handcrafted_features_v2(xyz: np.ndarray) -> np.ndarray:
    """Same 15 features as src.features.handcrafted.extract_handcrafted_features,
    plus 1 new one appended at the end (index 15): starting elbow angle of
    the striking arm. xyz: (SEQ_LEN, 33, 3) normalized world coords."""
    feats = []

    wrist_stats = {}
    for wrist in (L_WRIST, R_WRIST):
        peak_speed, timing, peak_idx = _peak_speed_and_timing(xyz, wrist)
        feats += [peak_speed, timing]
        wrist_stats[wrist] = (peak_speed, peak_idx)

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

    # v2 addition -- index 15
    feats += [_starting_elbow_angle(xyz, striking_wrist)]

    return np.array(feats, dtype=np.float32)
