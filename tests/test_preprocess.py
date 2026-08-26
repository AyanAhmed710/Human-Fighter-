"""Unit tests for preprocessing math -- run with: pytest tests/ (or plain python).
No video/MediaPipe dependency; synthetic arrays only.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import (
    bone_vectors,
    interpolate_nans,
    joint_angles,
    normalize,
    resample_sequence,
    smooth,
    to_feature_vector,
    preprocess_clip,
    velocity,
)
from src.config import ANGLE_TRIPLES, BONE_PAIRS, N_POSE_LANDMARKS, SEQ_LEN


def _dummy_clip(T=76, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, N_POSE_LANDMARKS, 3)).astype(np.float32)


def test_interpolate_nans_fills_all():
    xyz = _dummy_clip(20)
    xyz[5, 3, 1] = np.nan
    xyz[0, 0, 0] = np.nan  # edge case
    out = interpolate_nans(xyz)
    assert not np.isnan(out).any()


def test_smooth_preserves_shape():
    xyz = _dummy_clip(30)
    out = smooth(xyz, window=5)
    assert out.shape == xyz.shape


def test_normalize_centers_hip_at_origin():
    xyz = _dummy_clip(10)
    out = normalize(xyz)
    hip_mid = (out[:, 23] + out[:, 24]) / 2.0
    assert np.allclose(hip_mid, 0.0, atol=1e-5)


def test_resample_hits_target_length():
    xyz = _dummy_clip(76)
    out = resample_sequence(xyz, SEQ_LEN)
    assert out.shape[0] == SEQ_LEN
    assert out.shape[1:] == xyz.shape[1:]


def test_resample_short_clip_upsamples():
    xyz = _dummy_clip(29)  # min observed clip length
    out = resample_sequence(xyz, SEQ_LEN)
    assert out.shape[0] == SEQ_LEN


def test_velocity_first_frame_zero():
    xyz = _dummy_clip(10)
    vel = velocity(xyz)
    assert np.allclose(vel[0], 0.0)


def test_bone_vectors_shape():
    xyz = _dummy_clip(10)
    bones = bone_vectors(xyz)
    assert bones.shape[0] == xyz.shape[0]
    assert bones.shape[2] == 3


def test_joint_angles_shape_and_range():
    xyz = _dummy_clip(10)
    angles = joint_angles(xyz)
    assert angles.shape == (10, len(ANGLE_TRIPLES))
    assert (angles >= 0).all() and (angles <= 1).all()  # normalized by /pi


def test_joint_angles_known_configuration():
    # straight arm (shoulder-elbow-wrist collinear) -> angle ~pi -> normalized ~1.0
    xyz = np.zeros((1, N_POSE_LANDMARKS, 3), dtype=np.float32)
    xyz[0, 11] = [0, 0, 0]   # shoulder
    xyz[0, 13] = [1, 0, 0]   # elbow
    xyz[0, 15] = [2, 0, 0]   # wrist -- straight line
    angles = joint_angles(xyz)
    assert np.isclose(angles[0, 0], 1.0, atol=1e-3)  # first triple is left elbow

    # fully bent elbow (folded back on itself) -> angle ~0
    xyz[0, 15] = [0, 0, 0]  # wrist folds back onto shoulder
    angles = joint_angles(xyz)
    assert np.isclose(angles[0, 0], 0.0, atol=1e-3)


def test_to_feature_vector_dim_matches_streams():
    xyz = _dummy_clip(76)
    vis = np.ones((76, N_POSE_LANDMARKS), dtype=np.float32)
    sample = preprocess_clip(xyz, vis)
    feat = to_feature_vector(sample)
    expected_dim = 99 + 99 + 33 + len(BONE_PAIRS) * 3 + len(ANGLE_TRIPLES)
    assert feat.shape == (SEQ_LEN, expected_dim)


if __name__ == "__main__":
    import inspect
    fns = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
